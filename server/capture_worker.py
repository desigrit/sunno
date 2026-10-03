"""Small disposable audio process. stdout is a framed JSON protocol, never a log.

Only this process enters audio-driver calls. Closing the parent's stdin ends the
child even if a driver blocks, and the packaged UI's job also covers the child.
"""
from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
import threading

from .capture_target import AudioTarget
from .coreaudio import CaptureError


def main():
    target = AudioTarget.from_dict(json.loads(sys.argv[1]))
    probe = "--probe" in sys.argv[2:]
    output = sys.stdout
    stop = threading.Event()

    def watch_parent():
        while True:
            command = sys.stdin.readline()
            if not command:
                os._exit(0)  # A dead parent must not leave a blocked capture process alive.
            stop.set()  # Keep watching EOF even if native shutdown is blocked.

    threading.Thread(target=watch_parent, daemon=True, name="capture-parent").start()

    def send(message):
        output.write(json.dumps(message, ensure_ascii=True) + "\n")
        output.flush()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            if sys.platform == "win32":
                from .coreaudio import EndpointStream, list_endpoints

                devices = list_endpoints(target.kind)
                legacy = None
                if not target.endpoint_id and not target.name and not target.follow_default:
                    if target.kind == "loopback":
                        from .loopback import list_loopback_devices
                        legacy = list_loopback_devices()
                    else:
                        from .audio import list_input_devices
                        legacy = list_input_devices()
                selected = target.resolve(devices, legacy)
                source = lambda: EndpointStream(selected.endpoint_id, selected.kind)
            else:
                from .audio import MicrophoneStream, list_input_devices
                selected = target.resolve(list_input_devices())
                source = lambda: MicrophoneStream(selected.index)
            if probe:
                send({"type": "ready", "target": selected.to_dict(), "probe": True})
                return
            if stop.is_set():
                return
            with source() as stream:
                ready = False
                for frame in stream.frames(lambda: not stop.is_set()):
                    if stop.is_set():
                        break
                    if not ready:
                        send({"type": "ready", "target": selected.to_dict(),
                              "rate": stream.capture_rate, "channels": stream.capture_channels})
                        ready = True
                    send({"type": "audio", "data": base64.b64encode(frame.astype("<f4").tobytes()).decode("ascii")})
                if not stop.is_set():
                    raise CaptureError("capture_stalled", "The selected input stopped sending audio.")
    except Exception as exc:
        denied = getattr(exc, "access_denied", False)
        missing = isinstance(exc, ImportError)
        code = getattr(exc, "code", "capture_denied" if denied else "capture_failed")
        if missing:
            code = "capture_dependency"
        send({"type": "error", "code": code,
              "retryable": getattr(exc, "retryable", not denied and not missing),
              "message": str(exc) if isinstance(exc, CaptureError) else
                         "Part of Sunno's audio capture is missing. Reinstalling Sunno should repair it." if missing else
                         "Sunno could not open this input. It will try again."})


if __name__ == "__main__":
    main()
