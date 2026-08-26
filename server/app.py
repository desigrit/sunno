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


def _fresh_devices() -> list[dict] | None:
    """Re-read the device list in a child process, or None if that could not be done.

    PortAudio fixes its device list at Pa_Initialize, so this process cannot see anything
    plugged in since it started, and it cannot simply look again: re-initialising while a
    capture stream is open invalidates that stream and the next read fails. A child process
    starts its own PortAudio, reads the current hardware, and exits without ever touching
    this one's capture.

    Returning None rather than raising, because the caller's answer to "the refresh did not
    work" is to serve the cached list, which is still usable.
    """
    import os
    import subprocess
    import sys

    root = str(Path(__file__).resolve().parents[1])
    # cwd alone would work here, because Python puts the working directory on sys.path for
    # -m. It is not worth depending on: an embeddable interpreter with a ._pth file does not,
    # and the failure is a ModuleNotFoundError inside a child whose only visible symptom is a
    # refresh button that quietly does nothing. PYTHONPATH says it outright. Prepended rather
    # than assigned so an interpreter that needs its own entries keeps them.
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "server.enum_devices"],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=root,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:  # noqa: BLE001  (spawn failure, timeout, missing interpreter)
        print(f"[devices] refresh could not run ({type(exc).__name__}), serving cached list",
              flush=True)
        return None

    if proc.returncode != 0:
        # stderr carries the child's traceback, and also its own diagnostics, so only the
        # last line goes to the log. Device names are never in either.
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no detail"]
        print(f"[devices] refresh failed (exit {proc.returncode}): {tail[0]}", flush=True)
        return None

    try:
        payload = json.loads(proc.stdout)
        devices = payload["devices"]
    except Exception as exc:  # noqa: BLE001
        print(f"[devices] refresh returned unreadable output ({type(exc).__name__})", flush=True)
        return None

    if not isinstance(devices, list):
        print("[devices] refresh returned the wrong shape", flush=True)
        return None
    return devices


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

            # ?fresh=1 is the refresh button. Without it this serves what PortAudio cached at
            # startup, which is right for the startup call and wrong for every later one.
            fresh = "fresh=1" in self.path.partition("?")[2].split("&")
            if fresh:
                devices = _fresh_devices()
                if devices is not None:
                    self._json({"devices": devices})
                    return
                # Falling through to the cached list rather than erroring: a slightly stale
                # picker still lets someone choose a microphone, and an empty one does not.
                # Flagged so the UI can say so in its log instead of quietly believing this
                # was a real refresh.

            try:
                devices = list_input_devices()
                for d in devices:
                    d["loopback"] = False
            except Exception as exc:
                self._json({"error": str(exc), "devices": []})
                return
            # One entry per connected device by the time it gets here: list_input_devices
            # narrows to the WASAPI enumeration, which is where the legacy host APIs' stale
            # and duplicate entries were coming from. Plain alphabetical order is all that
            # is left to do. Sorting WASAPI first, as this did, is now either a no-op or,
            # on the fallback path, a sort by an API that returned nothing.
            devices.sort(key=lambda d: d["name"])

            # Output endpoints, so what is being played can be captioned too. Appended after
            # the microphones and flagged, so the UI can group them rather than mixing two
            # very different things in one flat list.
            try:
                from .loopback import list_loopback_devices

                devices.extend(list_loopback_devices())
            except Exception:
                # Loopback is an enhancement; its absence must not break the picker.
                pass

            payload = {"devices": devices}
            if fresh:
                payload["stale"] = True
            self._json(payload)
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

    compute_device = hardware.resolve_device(args.compute_device)
    compute_type = args.compute_type or hardware.compute_type_for(compute_device)
    model = args.model or hardware.default_model(
        [entry["id"] for entry in model_catalog.CATALOG], compute_device
    )

    settings = Settings(
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
                            "device": settings.device,
                            "gpu_payload_needed": gpu_payload_needed(),
                            "gpu_payload_mb": round(cuda_bytes() / 1e6),
                            "catalog": await asyncio.to_thread(
                                model_catalog.catalog_with_status, catalog_device()
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

    # Probed unconditionally, not inside the CUDA branch. Force CPU skips the GPU probe
    # entirely, so an ARM user with that setting on would otherwise get no report at all -
    # exactly the machine where "the engine will not load" is the whole story. Cached, so the
    # auto path that already probed during argument parsing does not report twice.
    from . import hardware as _hw

    _hw.engine_importable()

    # Recorded plainly rather than as [error], because emulation is not by itself a failure:
    # Prism emulates AVX2 on recent Windows, so the x64 engine may load and run, just slowly.
    # If it does not load, engine_importable() reports the real failure with the same machine name.
    if _hw.is_emulated():
        print("  Note:      x64 engine running under ARM64 emulation; expect it to be slow")

    model_ready = asyncio.Event()
    chosen_model = settings.model_size
    downloading = False

    def gpu_payload_needed() -> bool:
        """Whether this machine wants the CUDA libraries and does not have them.

        Both halves matter. An AMD or Intel machine must never be asked to fetch 455 MB it
        can never load, and a machine that already has them must never fetch them twice.
        """
        from . import hardware

        return hardware.gpu_capable() and not hardware.payload_ready()

    def payload_needed_for(model_id: str) -> bool:
        """Whether choosing this model should pull the CUDA libraries down.

        The streaming models pin themselves to CPU providers, so they never need it however
        good the graphics card is.
        """
        from . import models as model_catalog

        if model_catalog.is_stream_model(model_id):
            return False
        return gpu_payload_needed()

    def cuda_bytes() -> int:
        try:
            from . import cuda_download

            return cuda_download.compressed_total()
        except Exception:
            return 0

    def catalog_device() -> str:
        """Which device the picker should quote timings for.

        Not settings.device. On a machine with an NVIDIA GPU but no libraries yet, the
        engine really is on the processor, but choosing a model installs the libraries, so
        the speed the user is deciding against is the GPU one. Quoting processor timings
        here would push a first-run RTX owner towards Whisper base and mark large-v3 as
        several seconds behind, on a machine that will run it in half a second.

        The number is wrong for one case: someone whose library download fails or is
        cancelled stays on the processor while this still promises GPU speed. Accepted
        deliberately. The alternative mis-sells every fresh install to protect a few.
        """
        from . import hardware

        return "cuda" if hardware.gpu_capable() else settings.device

    async def ensure_model(model_id: str) -> None:
        """Install whatever `model_id` needs, then release the loader.

        Two things can be missing: the model itself, and the CUDA libraries the Whisper
        engine needs on a machine that has an NVIDIA GPU. They are reported as one download
        against one total, because from the user's side it is a single act — they picked a
        model and it is being set up. Two bars, or a bar that finishes and then starts again,
        describes our packaging rather than their intent.

        Model first, then the payload. Either leg can be interrupted by the user closing the
        app, and the order decides what they are left with: model-first leaves a working
        install running on the processor, payload-first leaves 455 MB of libraries and no way
        to caption anything.

        A payload failure is deliberately not fatal and does not even mark the operation
        failed. The model is there, captions work, they are just slower. Saying "couldn't
        download that model" over a model that downloaded perfectly would be a lie, and the
        next model switch retries the payload anyway.
        """
        nonlocal chosen_model, downloading
        if downloading:
            return
        downloading = True
        chosen_model = model_id
        try:
            from . import models as model_catalog

            need_payload = payload_needed_for(model_id)
            model_missing = not model_catalog.is_available(model_id).available
            payload_bytes = cuda_bytes() if need_payload else 0
            model_bytes = 0

            if model_missing or need_payload:
                emit({"type": "download_started", "model": model_id})

            last = [0.0]

            def report(done: int, total: int, force: bool = False) -> None:
                # Throttle: the hub reports in small chunks and the UI only needs ~10 Hz.
                now = time.monotonic()
                if not force and done < total and now - last[0] < 0.1:
                    return
                last[0] = now
                emit({
                    "type": "download_progress",
                    "model": model_id,
                    "downloaded": done,
                    "total": total,
                    "percent": round(100 * done / total, 1) if total else 0.0,
                })

            if model_missing:
                print(f"Downloading {model_id} ...", flush=True)

                def on_model(done: int, total: int) -> None:
                    nonlocal model_bytes
                    model_bytes = total
                    report(done, total + payload_bytes)

                await asyncio.to_thread(model_catalog.download, model_id, on_model)
                print(f"Downloaded {model_id}", flush=True)

            if need_payload:
                # Announced to the log, not to the screen. The user asked for a model; the
                # libraries it happens to need are our business, and a second named download
                # appearing halfway through would only raise a question they cannot act on.
                print("Downloading CUDA support libraries ...", flush=True)
                base = model_bytes
                try:
                    from . import cuda_download

                    def on_payload(done: int, total: int) -> None:
                        report(base + done, base + total)

                    await asyncio.to_thread(cuda_download.download_payload, on_payload)
                    print("CUDA support libraries installed", flush=True)
                except Exception as exc:
                    print(f"[error] CUDA library download failed: {exc}", flush=True)

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
            # The model is on disk, so nothing above will ever ask the user to choose, which
            # means ensure_model never runs and the CUDA libraries would never be fetched.
            # That is the state every existing GPU user lands in after updating to the build
            # that stopped shipping them: cached model, vanished payload, and a permanent
            # drop from 455 ms to about 4.1 s on large-v3 — which this app's own `responsive`
            # threshold classifies as too slow to follow a conversation. Not a first-launch
            # blip; without this it never repairs itself.
            #
            # Repaired in the background, on purpose. Blocking here would hold captions
            # hostage to a 455 MB transfer on a machine that was captioning fine a minute
            # ago, and for a deaf user no captions is worse than slow ones. So the engine
            # starts on the processor immediately and the libraries land behind it.
            #
            # Silent, as the product intends: no prompt, no progress, nothing to dismiss.
            # It applies at the next launch, because the engine is already built against the
            # device resolved above and swapping it mid-conversation is a worse failure than
            # waiting for a restart.
            if payload_needed_for(settings.model_size):
                async def repair_payload() -> None:
                    from . import cuda_download

                    try:
                        print("GPU libraries are missing; fetching them in the background. "
                              "Captions run on the processor until the next restart.",
                              flush=True)
                        await asyncio.to_thread(cuda_download.download_payload, None)
                        print("GPU libraries installed; they take effect next launch.",
                              flush=True)
                    except Exception as exc:
                        # Never fatal, never surfaced. Captions are already running.
                        print(f"[error] background GPU library fetch failed: {exc}",
                              flush=True)

                asyncio.create_task(repair_payload())
        else:
            catalog = await asyncio.to_thread(
                model_catalog.catalog_with_status, catalog_device()
            )
            latest_status = {
                "type": "model_required",
                "requested": settings.model_size,
                "device": settings.device,
                "gpu_payload_needed": gpu_payload_needed(),
                "gpu_payload_mb": round(cuda_bytes() / 1e6),
                "catalog": catalog,
            }
            emit(dict(latest_status))
            print(f"Model '{settings.model_size}' not downloaded yet; waiting for a choice.")

        await model_ready.wait()
        settings.model_size = chosen_model

        # Re-resolve the device now the download has finished.
        #
        # settings.device was fixed in parse_args, before anything was downloaded. On a PC
        # with an NVIDIA card and no libraries yet, that read "cpu" — correctly at the time.
        # If the user has just picked a model and we have since installed the libraries, that
        # answer is stale, and leaving it stale is the worst outcome this change can produce:
        # the machine downloads the payload and then spends the entire session on the
        # processor anyway. The picker quoted GPU speed to get them here, so a first run that
        # ends on CPU is both slow and a broken promise.
        #
        # Only ever an upgrade. args.compute_type is honoured if the user set it explicitly,
        # and a forced --compute-device cpu still resolves to cpu, so this cannot override
        # either. The create_engine/warmup guard below catches it if the upgrade is wrong.
        if settings.device != "cuda":
            from . import hardware

            upgraded = hardware.resolve_device(args.compute_device)
            if upgraded != settings.device:
                settings.device = upgraded
                settings.compute_type = args.compute_type or hardware.compute_type_for(upgraded)
                print(f"GPU libraries are now installed; using {upgraded}", flush=True)

        # A broken CUDA payload should not be fatal: the CPU path works, just slower, and a
        # captioning app that runs behind is far better than one that refuses to start.
        # resolve_device() has already checked the payload is complete, so reaching here with
        # a fault means it broke between that check and now.
        if settings.device == "cuda":
            from . import hardware
            from .cuda_setup import register_cuda_dlls

            try:
                register_cuda_dlls(required=True)
            except RuntimeError as exc:
                settings.device = "cpu"
                settings.compute_type = hardware.compute_type_for("cpu")
                print(f"[error] GPU unavailable ({exc}); falling back to CPU", flush=True)

        # Not every model is a Whisper checkpoint any more, and this line ends up in the
        # backend log a user attaches to a bug report, where "Loading Whisper stream-en"
        # would send whoever reads it looking for a Whisper model that does not exist.
        from .models import is_stream_model

        if is_stream_model(settings.model_size):
            print(f"\nLoading {settings.model_size} on cpu ...")
        else:
            print(f"\nLoading Whisper {settings.model_size} ({settings.compute_type}) on "
                  f"{settings.device} ...")
        emit({"type": "status", "state": "loading", "model": settings.model_size})

        from .engine import create_engine

        # warmup is inside the guard, not just create_engine, and that distinction is the
        # whole point. CTranslate2 resolves cuBLAS lazily: constructing a CUDA engine with
        # no libraries present succeeds, and the failure only appears when something is
        # actually decoded. warmup is that first decode. Guarding only the constructor would
        # have caught nothing and left the backend dying here, outside every handler, on a
        # machine whose payload was incomplete.
        try:
            engine = await asyncio.to_thread(create_engine, settings)
            warmup_ms = await asyncio.to_thread(engine.warmup)
        except Exception as exc:
            if settings.device == "cpu":
                raise
            from . import hardware

            print(f"[error] GPU engine failed to start ({exc}); falling back to CPU",
                  flush=True)
            settings.device = "cpu"
            settings.compute_type = hardware.compute_type_for("cpu")
            emit({"type": "status", "state": "loading", "model": settings.model_size})
            engine = await asyncio.to_thread(create_engine, settings)
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
                        emit({
                            "type": "error",
                            "code": "capture_failed",
                            "message": "Sunno could not start listening on this microphone.",
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
