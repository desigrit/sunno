"""Enumerate capture devices in a fresh process, so a running capture is left alone.

PortAudio builds its device list at Pa_Initialize and never revisits it, which is why a
backend that has been up for an hour still cannot see a microphone plugged in ten minutes
ago. Re-initialising in the running process is not an option: with a stream open it
invalidates that stream, and the next read raises
`PortAudioError: Invalid stream pointer [PaErrorCode -9988]` — captions would stop
mid-sentence, which is the one failure this app cannot have.

Both enumerations have to happen here, and the reason for the second one is easy to get
wrong. list_loopback_devices() builds a fresh pyaudiowpatch instance per call, which looks
like it re-enumerates on its own. It does not, whenever it matters: Pa_Initialize is
reference counted, so while a loopback capture is running the "fresh" instance is a no-op
that hands back the list cached when capture started. Measured on one machine, the same
call took 107 ms with nothing else holding PortAudio open and 0.01 ms while a capture was
live — a ten-thousandfold gap that is the library declining to look. A separate process
holds no reference count, so both lists are genuinely re-read here.

The output contract is deliberately narrow, because the caller's fallback is silent by
design and a half-broken payload would be worse than an obvious failure:

    success  ->  {"devices": [...]} on stdout, exit 0
    failure  ->  traceback on stderr, non-zero exit, nothing on stdout

Errors are never encoded into stdout. An empty device list is a valid success: it means a
machine with no capture hardware, which is a real state and not an error.
"""

from __future__ import annotations

import contextlib
import json
import sys


def collect() -> list[dict]:
    """Every device the picker can offer, microphones first, then output endpoints.

    Mirrors what the /devices.json handler builds in-process, including the sort, so that a
    refreshed list and a startup list differ only in how recently they were read.
    """
    from .audio import list_input_devices

    # list_input_devices prints a diagnostic count line, and stdout here belongs to the JSON
    # payload. Sending it to stderr keeps it in the parent's log where it is useful.
    with contextlib.redirect_stdout(sys.stderr):
        devices = list_input_devices()
        for d in devices:
            d["loopback"] = False
        devices.sort(key=lambda d: d["name"])

        try:
            from .loopback import list_loopback_devices

            devices.extend(list_loopback_devices())
        except Exception:
            # Loopback is an enhancement. A machine without it should still get a refreshed
            # microphone list rather than a failed refresh.
            pass

    return devices


def main() -> int:
    json.dump({"devices": collect()}, sys.stdout)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
