"""Entry point: mic -> VAD -> Whisper -> WebSocket, with the UI served over HTTP."""

from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import json
import socket
import threading
import time
from pathlib import Path

from . import cuda_setup  # noqa: F401  (must precede ctranslate2 import)
from .audio import MicrophoneOpenError, MicrophoneStream, WavFileStream, print_input_devices
from .config import Settings
from .paths import bundled_model, data_dir, speaker_profiles_path, ui_dir
from .pipeline import CaptionPipeline, SessionController

UI_DIR = ui_dir()


def _local_ip() -> str:
    """Best-effort LAN address, so the UI can be opened from another device."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))  # no packets sent; just picks the route
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


class _UiRequestHandler(http.server.SimpleHTTPRequestHandler):
    ws_port: int = 8766

    def end_headers(self) -> None:
        # The UI is served from disk and edited in place; browser caching would silently
        # serve a stale page after every change.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/config.json":
            self._json({"wsPort": self.ws_port})
            return
        if path == "/devices.json":
            # Lets the UI populate a microphone picker without shelling out to the CLI.
            from .audio import list_input_devices

            try:
                devices = list_input_devices()
                for d in devices:
                    d["loopback"] = False
            except Exception as exc:
                self._json({"error": str(exc), "devices": []})
                return
            # WASAPI is the modern Windows path; surface it first and hide duplicates
            # of the same device exposed through legacy host APIs.
            devices.sort(key=lambda d: (d["hostapi"] != "Windows WASAPI", d["name"]))

            # Output endpoints, so what is being played can be captioned too. Appended after
            # the microphones and flagged, so the UI can group them rather than mixing two
            # very different things in one flat list.
            try:
                from .loopback import list_loopback_devices

                devices.extend(list_loopback_devices())
            except Exception:
                # Loopback is an enhancement; its absence must not break the picker.
                pass

            self._json({"devices": devices})
            return
        super().do_GET()

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # silence per-request logging
        pass


def _serve_ui(host: str, port: int, ws_port: int) -> None:
    handler = functools.partial(_UiRequestHandler, directory=str(UI_DIR))
    _UiRequestHandler.ws_port = ws_port
    httpd = http.server.ThreadingHTTPServer((host, port), handler)
    threading.Thread(target=httpd.serve_forever, name="ui-http", daemon=True).start()


def parse_args() -> tuple[Settings, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Offline live captioning server")
    parser.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    parser.add_argument("--device", default=None, help="input device index or name substring")
    parser.add_argument(
        "--loopback-device", type=int, default=None,
        help="WASAPI output endpoint index to capture instead of the microphone, so system "
             "audio (calls, video) is transcribed",
    )
    parser.add_argument("--wav", default=None, help="replay a WAV file instead of the mic")
    parser.add_argument(
        "--fast", action="store_true", help="with --wav, replay as fast as possible"
    )
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    parser.add_argument("--compute-type", default="float16", help="ctranslate2 compute type")
    parser.add_argument("--language", default="en", help="forced language, or 'auto'")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 for LAN)")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--ws-port", type=int, default=8766)
    parser.add_argument("--end-silence-ms", type=int, default=520)
    parser.add_argument("--partial-interval-ms", type=int, default=450)
    parser.add_argument(
        "--start-stopped",
        action="store_true",
        help="launch paused; press Start in the UI to begin capturing",
    )
    parser.add_argument(
        "--no-speakers", action="store_true", help="disable speaker labelling"
    )
    parser.add_argument(
        "--vocabulary",
        default="",
        help="comma-separated names/places to bias transcription (e.g. 'Hyderabad,Priya')",
    )
    args = parser.parse_args()

    device: int | str | None = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    settings = Settings(
        model_size=args.model,
        compute_type=args.compute_type,
        language=None if args.language == "auto" else args.language,
        host=args.host,
        http_port=args.http_port,
        ws_port=args.ws_port,
        input_device=device,
        loopback_device=args.loopback_device,
        end_silence_ms=args.end_silence_ms,
        partial_interval_ms=args.partial_interval_ms,
        vocabulary=tuple(v.strip() for v in args.vocabulary.split(",") if v.strip()),
    )
    return settings, args


async def run(settings: Settings, args: argparse.Namespace) -> None:
    from websockets.asyncio.server import serve

    loop = asyncio.get_running_loop()
    clients: set = set()
    events: asyncio.Queue[dict] = asyncio.Queue()
    latest_status: dict = {"type": "status", "state": "starting", "running": False}
    controller = SessionController(running=not args.start_stopped)

    speaker = None
    if settings.enable_speakers and not args.no_speakers:
        from .speaker import SpeakerIdentifier

        model_file = bundled_model(settings.speaker_model)
        try:
            speaker = SpeakerIdentifier(
                model_file,
                threshold=settings.speaker_threshold,
                max_speakers=settings.max_speakers,
                min_identify_s=settings.speaker_min_identify_s,
                min_new_speaker_s=settings.speaker_min_new_s,
                profile_path=speaker_profiles_path(),
            )
            print(f"Speaker labelling: on ({settings.speaker_model})")
        except Exception as exc:
            print(f"Speaker labelling: off ({exc})")

    def emit(event: dict) -> None:
        """Thread-safe bridge from pipeline threads into the asyncio loop."""
        loop.call_soon_threadsafe(events.put_nowait, event)

    async def handler(ws) -> None:  # noqa: ANN001
        clients.add(ws)
        await ws.send(json.dumps(latest_status))
        if speaker is not None:
            await ws.send(json.dumps({"type": "roster", "speakers": speaker.roster()}))
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                cmd = msg.get("cmd")
                if cmd == "start":
                    controller.start()
                elif cmd == "stop":
                    controller.pause()
                elif cmd == "toggle":
                    controller.toggle()
                elif cmd == "download_model":
                    requested = str(msg.get("model") or settings.model_size)
                    loop.create_task(ensure_model(requested))
                elif cmd == "list_models":
                    # The first-run gate only advertises the catalogue when a model is missing.
                    # The switcher needs it whenever asked, including once one is already loaded.
                    async def send_catalog() -> None:
                        from . import models as model_catalog

                        emit({
                            "type": "model_catalog",
                            "current": settings.model_size,
                            "catalog": await asyncio.to_thread(model_catalog.catalog_with_status),
                        })

                    loop.create_task(send_catalog())
                elif cmd == "rename_speaker" and speaker is not None:
                    if speaker.rename(int(msg.get("id", -1)), str(msg.get("name", ""))):
                        emit({"type": "roster", "speakers": speaker.roster()})
                elif cmd == "set_self" and speaker is not None:
                    if speaker.set_self(int(msg.get("id", -1)), bool(msg.get("value", True))):
                        emit({"type": "roster", "speakers": speaker.roster()})
                elif cmd == "merge_speakers" and speaker is not None:
                    if speaker.merge(int(msg.get("source", -1)), int(msg.get("target", -1))):
                        emit({"type": "roster", "speakers": speaker.roster()})
                elif cmd == "reset_speakers" and speaker is not None:
                    speaker.reset()
                    emit({"type": "roster", "speakers": speaker.roster()})
        finally:
            clients.discard(ws)

    async def broadcaster() -> None:
        nonlocal latest_status
        while True:
            event = await events.get()
            kind = event.get("type")
            if kind == "status":
                latest_status = event
            elif kind == "final" and event.get("text"):
                who = event.get("speaker")
                prefix = f"{who}: " if who else ""
                clarity = event.get("clarity")
                meta = f"{event['latency_ms']:>6.0f} ms"
                if clarity is not None:
                    meta += f", clarity {clarity:>3}%"
                print(f"  [{meta}] {prefix}{event['text']}", flush=True)
            elif kind == "error":
                print(f"  [error] {event.get('message')}", flush=True)
            if not clients:
                continue
            message = json.dumps(event)
            for ws in list(clients):
                try:
                    await ws.send(message)
                except Exception:
                    clients.discard(ws)

    _serve_ui(settings.host, settings.http_port, settings.ws_port)
    ui_host = _local_ip() if settings.host == "0.0.0.0" else settings.host
    print(f"  UI:        http://{ui_host}:{settings.http_port}")
    print(f"  WebSocket: ws://{ui_host}:{settings.ws_port}")

    model_ready = asyncio.Event()
    chosen_model = settings.model_size
    downloading = False

    async def ensure_model(model_id: str) -> None:
        """Download a model if it isn't cached, reporting progress, then release the loader."""
        nonlocal chosen_model, downloading
        if downloading:
            return
        downloading = True
        chosen_model = model_id
        try:
            from . import models as model_catalog

            if not model_catalog.is_available(model_id).available:
                emit({"type": "download_started", "model": model_id})
                print(f"Downloading {model_id} ...", flush=True)

                last = [0.0]

                def on_progress(done: int, total: int) -> None:
                    # Throttle: the hub reports in small chunks and the UI only needs ~10 Hz.
                    now = time.monotonic()
                    if done < total and now - last[0] < 0.1:
                        return
                    last[0] = now
                    emit({
                        "type": "download_progress",
                        "model": model_id,
                        "downloaded": done,
                        "total": total,
                        "percent": round(100 * done / total, 1) if total else 0.0,
                    })

                await asyncio.to_thread(model_catalog.download, model_id, on_progress)
                print(f"Downloaded {model_id}", flush=True)

            emit({"type": "download_complete", "model": model_id})
            model_ready.set()
        except Exception as exc:
            emit({"type": "download_failed", "model": model_id, "message": str(exc)})
            print(f"[error] model download failed: {exc}", flush=True)
        finally:
            downloading = False

    async with serve(handler, settings.host, settings.ws_port):
        asyncio.create_task(broadcaster())

        # Decide up front whether we can load, or must ask the user to pick and download.
        from . import models as model_catalog

        if model_catalog.is_available(settings.model_size).available:
            model_ready.set()
        else:
            catalog = await asyncio.to_thread(model_catalog.catalog_with_status)
            latest_status = {
                "type": "model_required",
                "requested": settings.model_size,
                "catalog": catalog,
            }
            emit(dict(latest_status))
            print(f"Model '{settings.model_size}' not downloaded yet; waiting for a choice.")

        await model_ready.wait()
        settings.model_size = chosen_model

        # Fail loudly and early if the CUDA payload is mis-staged, rather than surfacing an
        # opaque ctranslate2 DLL load error once the user is already mid-conversation.
        if settings.device == "cuda":
            from .cuda_setup import register_cuda_dlls

            try:
                register_cuda_dlls(required=True)
            except RuntimeError as exc:
                emit({"type": "error", "message": str(exc), "running": False})
                print(f"[fatal] {exc}", flush=True)
                raise

        print(f"\nLoading Whisper {settings.model_size} ({settings.compute_type}) on "
              f"{settings.device} ...")
        emit({"type": "status", "state": "loading", "model": settings.model_size})

        from .asr import WhisperEngine

        engine = await asyncio.to_thread(WhisperEngine, settings)
        warmup_ms = await asyncio.to_thread(engine.warmup)
        print(f"Model ready (warmup {warmup_ms:.0f} ms)")

        pipeline = CaptionPipeline(
            settings, engine, emit,
            should_run=lambda: controller.is_running,
            speaker=speaker,
        )

        def make_source():
            if args.wav:
                return WavFileStream(args.wav, realtime=not args.fast)
            # A loopback endpoint captures what is being played rather than what is being
            # said. Selected by index like any other device, so the UI needs no separate
            # control — it just marks which entries are outputs.
            if settings.loopback_device is not None:
                from .loopback import LoopbackStream

                return LoopbackStream(settings.loopback_device)
            return MicrophoneStream(settings.input_device)

        def pump() -> None:
            """One capture session per start/stop cycle.

            The microphone is opened on start and closed on stop, so Windows' mic-in-use
            indicator reflects reality. The model and ASR worker persist across cycles.
            """
            announced_idle = False
            try:
                while not controller.is_shutdown:
                    if not controller.is_running:
                        if not announced_idle:
                            emit({"type": "status", "state": "stopped", "running": False})
                            print("Stopped. Microphone released.", flush=True)
                            announced_idle = True
                        controller.wait_for_start(timeout=0.25)
                        continue

                    announced_idle = False
                    try:
                        with make_source() as stream:
                            emit(
                                {
                                    "type": "status",
                                    "state": "listening",
                                    "running": True,
                                    "model": settings.model_size,
                                    "device": stream.device_name,
                                }
                            )
                            print(f"Listening on: {stream.device_name}", flush=True)
                            pipeline.run(stream.frames(lambda: controller.is_running))
                    except MicrophoneOpenError as exc:
                        # Surface a distinguishable code so the UI can offer the right fix
                        # rather than showing a wall of PortAudio diagnostics.
                        emit({
                            "type": "error",
                            "code": "mic_denied" if exc.access_denied else "mic_unavailable",
                            "message": str(exc),
                            "detail": exc.detail(),
                            "running": False,
                        })
                        print(f"[error] {exc}\n  {exc.detail()}", flush=True)
                        controller.pause()
                        continue
                    except Exception as exc:
                        emit({"type": "error", "message": str(exc), "running": False})
                        print(f"[error] {exc}", flush=True)
                        controller.pause()
                        continue

                    if args.wav and controller.is_running:
                        pipeline.drain()  # let in-flight transcriptions finish
                        break  # file exhausted
            finally:
                pipeline.close()

        print("Press Ctrl+C to stop.\n")
        worker = asyncio.to_thread(pump)
        try:
            await worker
        except asyncio.CancelledError:
            controller.shutdown()
            pipeline.stop()
            raise


def main() -> None:
    settings, args = parse_args()
    if args.list_devices:
        print_input_devices()
        return
    try:
        asyncio.run(run(settings, args))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
