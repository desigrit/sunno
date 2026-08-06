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
        "--loopback-device", default=None,
        help="WASAPI output endpoint id to capture instead of the microphone, so system "
             "audio (calls, video) is transcribed",
    )
    parser.add_argument("--wav", default=None, help="replay a WAV file instead of the mic")
    parser.add_argument(
        "--fast", action="store_true", help="with --wav, replay as fast as possible"
    )
    parser.add_argument("--model", default=None,
                        help="Whisper model size; defaults to the best one this hardware "
                             "can keep up with")
    parser.add_argument("--compute-type", default=None,
                        help="ctranslate2 compute type; defaults to float16 on GPU, int8 on CPU")
    parser.add_argument("--compute-device", default="auto", choices=("auto", "cuda", "cpu"),
                        help="where to run the model; 'auto' uses the GPU when one is usable")
    parser.add_argument("--language", default="en", help="forced language, or 'auto'")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 for LAN)")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--ws-port", type=int, default=8766)
    parser.add_argument("--end-silence-ms", type=int, default=520)
    parser.add_argument("--partial-interval-ms", type=int, default=450)
    parser.add_argument(
        "--echo-transcript",
        action="store_true",
        help="print recognised words to the console. Off by default, and deliberately so: "
             "the desktop app captures this process's stdout into a permanent log file, so "
             "echoing transcripts here would leave a plaintext record of every conversation "
             "the app has ever heard, including from people who never consented to it",
    )
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

    # Resolve the compute device before anything imports CTranslate2 for real. Most Windows
    # PCs have no usable NVIDIA GPU, and this used to be hardcoded to CUDA — which turned
    # every such machine into an install that raised at startup and never captioned.
    from . import hardware, models as model_catalog
    from .engine import resolve_engine

    engine_kind = resolve_engine()
    compute_device = "cpu" if engine_kind == "onnx" else hardware.resolve_device(args.compute_device)
    compute_type = args.compute_type or hardware.compute_type_for(compute_device)
    model = args.model or hardware.default_model(
        [entry["id"] for entry in model_catalog.catalog_for(engine_kind)],
        compute_device,
        engine_kind,
    )

    settings = Settings(
        engine=engine_kind,
        model_size=model,
        device=compute_device,
        compute_type=compute_type,
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
    if settings.engine == "ct2" and settings.enable_speakers and not args.no_speakers:
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
                            "device": settings.device,
                            "catalog": await asyncio.to_thread(
                                model_catalog.catalog_with_status, settings.device, settings.engine
                            ),
                        })

                    loop.create_task(send_catalog())
                elif cmd == "rename_speaker" and speaker is not None:
                    if speaker.rename(int(msg.get("id", -1)), str(msg.get("name", ""))):
                        emit({"type": "roster", "speakers": speaker.roster()})
                elif cmd == "set_self" and speaker is not None:
                    if speaker.set_self(int(msg.get("id", -1)), bool(msg.get("value", True))):
                        emit({"type": "roster", "speakers": speaker.roster()})
                elif cmd == "merge_speakers" and speaker is not None:
                    source = int(msg.get("source", -1))
                    target = int(msg.get("target", -1))
                    if speaker.merge(source, target):
                        # Announced before the roster, and it has to stay that way. Speaker
                        # ids are durable now, so a merged-away id simply vanishes from the
                        # roster and the UI's relabel pass skips the captions carrying it -
                        # they would keep the old name for good. This tells the UI to move
                        # those lines onto the survivor first; the roster that follows then
                        # names them correctly.
                        emit({"type": "speaker_merged", "from": source, "into": target})
                        emit({"type": "roster", "speakers": speaker.roster()})
                elif cmd == "delete_speaker" and speaker is not None:
                    target_id = int(msg.get("id", -1))
                    fallback = speaker.delete(target_id)
                    if fallback is not None:
                        # Like speaker_merged, this goes out before the roster and for the
                        # same reason: the deleted id is gone from the roster, so the UI's
                        # relabel pass skips those captions and they would keep the name of
                        # someone the user just asked the app to forget.
                        emit({"type": "speaker_deleted", "id": target_id, "label": fallback})
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
                clarity = event.get("clarity")
                meta = f"{event['latency_ms']:>6.0f} ms"
                if clarity is not None:
                    meta += f", clarity {clarity:>3}%"
                if args.echo_transcript:
                    who = event.get("speaker")
                    prefix = f"{who}: " if who else ""
                    print(f"  [{meta}] {prefix}{event['text']}", flush=True)
                else:
                    # Neither the words nor the speaker's name go on stdout. The app captures
                    # this stream into %LOCALAPPDATA%\Sunno\backend.log, which has no rotation
                    # and no size cap, so anything printed here is kept forever in plaintext.
                    # That is the opposite of what this app promises the room it is sitting in.
                    # The numeric id and length still prove captions are flowing and how fast,
                    # which is all the log is for.
                    sid = event.get("speaker_id")
                    # Not `or "-"`: speaker 0 is a real speaker and is falsy.
                    print(f"  [{meta}] speaker {'-' if sid is None else sid}, "
                          f"{len(event['text'])} chars", flush=True)
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

    from . import hardware as _hw

    if not _hw.engine_importable(settings.engine):
        raise RuntimeError(f"The {settings.engine} speech engine is not available")

    if settings.engine == "ct2":
        # Prism emulation is only relevant to the x64 CT2 build. A native ARM64 ONNX process
        # should not warn about the architecture it was built for.
        if _hw.is_emulated():
            print("  Note:      x64 engine running under ARM64 emulation; expect it to be slow")

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

            if not model_catalog.is_available(model_id, settings.engine).available:
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

                await asyncio.to_thread(
                    model_catalog.download, model_id, on_progress, settings.engine
                )
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

        if model_catalog.is_available(settings.model_size, settings.engine).available:
            model_ready.set()
        else:
            catalog = await asyncio.to_thread(
                model_catalog.catalog_with_status, settings.device, settings.engine
            )
            latest_status = {
                "type": "model_required",
                "requested": settings.model_size,
                "device": settings.device,
                "catalog": catalog,
            }
            emit(dict(latest_status))
            print(f"Model '{settings.model_size}' not downloaded yet; waiting for a choice.")

        await model_ready.wait()
        settings.model_size = chosen_model

        # A mis-staged CUDA payload should not be fatal any more: the CPU path works, just
        # slower, and a captioning app that runs behind is far better than one that refuses
        # to start. hardware.resolve_device() has already proved CUDA loads, so reaching
        # here means the payload broke between that check and now.
        if settings.device == "cuda":
            from . import hardware
            from .cuda_setup import register_cuda_dlls

            try:
                register_cuda_dlls(required=True)
            except RuntimeError as exc:
                settings.device = "cpu"
                settings.compute_type = hardware.compute_type_for("cpu")
                print(f"[error] GPU unavailable ({exc}); falling back to CPU", flush=True)

        print(f"\nLoading Whisper {settings.model_size} ({settings.compute_type}) on "
              f"{settings.device} ...")
        emit({"type": "status", "state": "loading", "model": settings.model_size})

        from .engine import create_engine

        engine = await asyncio.to_thread(create_engine, settings, settings.engine)
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
                            print("Listening.", flush=True)
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
                        # Carries a code and a human sentence, like MicrophoneOpenError above.
                        # This used to emit str(exc) as the message, which put raw PortAudio
                        # text on the user's screen - "[Errno -9996] Invalid device info" was
                        # the banner a user actually saw. Nearly everything that lands here is
                        # a capture device that could not be opened, and the remedy is the
                        # same, so say that and keep the diagnostics in detail.
                        loopback = settings.loopback_device is not None
                        emit({
                            "type": "error",
                            "code": "loopback_unavailable" if loopback else "capture_failed",
                            "message": (
                                "Sunno could not start capturing this system-audio source."
                                if loopback
                                else "Sunno could not start listening on this microphone."
                            ),
                            "detail": str(exc),
                            "running": False,
                        })
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
