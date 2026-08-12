"""Refreshing the device list must never be able to break the picker or the running capture.

The refresh runs the enumeration in a child process, because PortAudio fixes its device list
at Pa_Initialize and re-initialising in this process invalidates any open capture stream.
That makes the child's output a contract between two processes, and contracts between
processes fail quietly: a child that prints a stray line, exits non-zero, or hangs would
otherwise leave the picker showing stale hardware with nothing to say why.

These cover the child's side of that contract and the parent's handling of every way it can
go wrong. Run directly, matching the other tests here.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import app as server_app  # noqa: E402
from server import enum_devices  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"PASS  {name}")
    else:
        failures.append(f"{name}: {detail}")
        print(f"FAIL  {name}  {detail}")


@contextlib.contextmanager
def patched(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


def fake_run(*, stdout="", stderr="", returncode=0, raises=None):
    def run(*a, **kw):
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(a[0] if a else [], returncode, stdout, stderr)
    return run


# ---------------------------------------------------------------- the child's contract

devices = enum_devices.collect()
check("the child enumerates something", isinstance(devices, list), f"got {type(devices)}")

mics = [d for d in devices if not d.get("loopback")]
check(
    "microphones come before output endpoints",
    all(not d.get("loopback") for d in devices[: len(mics)]),
    "the two groups are interleaved",
)
check(
    "microphones are alphabetical",
    [d["name"] for d in mics] == sorted(d["name"] for d in mics),
    f"got {[d['name'] for d in mics]}",
)
check(
    "every entry carries the loopback flag",
    all("loopback" in d for d in devices),
    "an entry would land in neither picker group",
)

# The whole reason the child exists is that it must also re-read the output endpoints.
# list_loopback_devices looks self-refreshing because it builds a fresh pyaudiowpatch
# instance per call, but Pa_Initialize is reference counted: while a loopback capture is
# running that "fresh" instance is a no-op returning the list cached when capture started.
# Measured on one machine, 107 ms with nothing holding PortAudio open against 0.01 ms while
# a capture was live. A separate process holds no reference count, which is why both
# enumerations belong here rather than only the microphone one.
check(
    "the child re-reads output endpoints too, not just microphones",
    any(d.get("loopback") for d in devices) or not mics,
    "no loopback entries: a refresh would delete the System Audio group",
)

# stdout belongs to the payload. list_input_devices prints a diagnostic count line, so a
# regression here means every refresh silently falls back to the cached list forever.
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    enum_devices.main()
raw = buf.getvalue()
try:
    parsed = json.loads(raw)
    ok, detail = isinstance(parsed.get("devices"), list), ""
except Exception as exc:  # noqa: BLE001
    ok, detail = False, f"{exc} -- stdout began {raw[:60]!r}"
check("the child writes nothing to stdout but JSON", ok, detail)

# End to end through a real process, which is the only way to catch an import-time print.
proc = subprocess.run(
    [sys.executable, "-m", "server.enum_devices"],
    capture_output=True, text=True, cwd=str(ROOT), timeout=60,
)
check("a real child exits 0", proc.returncode == 0, f"exit {proc.returncode}: {proc.stderr[-200:]}")
try:
    real = json.loads(proc.stdout)["devices"]
    ok, detail = isinstance(real, list), ""
except Exception as exc:  # noqa: BLE001
    ok, detail = False, f"{exc} -- stdout began {proc.stdout[:60]!r}"
check("a real child's stdout parses", ok, detail)

# Device names are health information here: "Headset (R-Phonak hearing aid)" says someone
# wears a hearing aid, and stderr is forwarded into backend.log, which users are asked to
# send when something breaks.
leaked = [d["name"] for d in real if d["name"] in proc.stderr] if isinstance(real, list) else []
check("no device name reaches the child's stderr", not leaked, f"leaked {leaked}")


# ------------------------------------------------------- the parent's failure handling
#
# Each of these must return None, which the handler turns into "serve the cached list and
# mark it stale". Any of them returning a value would put fabricated hardware in the picker.

cases = [
    ("the interpreter cannot be spawned", fake_run(raises=OSError("no such file"))),
    ("the child times out", fake_run(raises=subprocess.TimeoutExpired("cmd", 20))),
    ("the child exits non-zero", fake_run(returncode=1, stderr="Traceback...\nValueError: x")),
    ("the child prints nothing", fake_run(stdout="")),
    ("the child prints a diagnostic before its JSON", fake_run(stdout='[audio] 3 connected\n{"devices": []}')),
    ("the child prints valid JSON of the wrong shape", fake_run(stdout='{"devices": {"a": 1}}')),
    ("the child omits the devices key", fake_run(stdout='{"ok": true}')),
    ("the child prints a bare list", fake_run(stdout='[{"index": 1}]')),
]
for label, runner in cases:
    with patched(subprocess, "run", runner), contextlib.redirect_stdout(io.StringIO()):
        got = server_app._fresh_devices()
    check(f"refresh gives up cleanly when {label}", got is None, f"returned {got!r}")

with patched(subprocess, "run", fake_run(stdout='{"devices": [{"index": 7, "name": "Mic"}]}')), \
        contextlib.redirect_stdout(io.StringIO()):
    got = server_app._fresh_devices()
check(
    "a well-formed child is passed through untouched",
    got == [{"index": 7, "name": "Mic"}],
    f"got {got!r}",
)

with patched(subprocess, "run", fake_run(stdout='{"devices": []}')), \
        contextlib.redirect_stdout(io.StringIO()):
    got = server_app._fresh_devices()
check(
    "an empty list is a success, not a failure",
    got == [],
    "a machine with no capture hardware is a real state, and falling back would hide it",
)

# The give-up path is logged, because a refresh that quietly does nothing is the failure
# mode this whole file exists to prevent. Counts and reasons only, never device names.
buf = io.StringIO()
with patched(subprocess, "run", fake_run(returncode=1, stderr="ValueError: boom")), \
        contextlib.redirect_stdout(buf):
    server_app._fresh_devices()
check("giving up is logged", "refresh failed" in buf.getvalue(), f"got {buf.getvalue()!r}")

# The child is found by module name, which only resolves if the backend root is importable.
# Relying on cwd alone works with a normal interpreter and fails with an embeddable one that
# ships a ._pth file, and the symptom would be a refresh button that quietly does nothing.
captured: dict = {}


def recording_run(*a, **kw):
    captured.update(kw)
    return subprocess.CompletedProcess(a[0] if a else [], 0, '{"devices": []}', "")


with patched(subprocess, "run", recording_run), contextlib.redirect_stdout(io.StringIO()):
    server_app._fresh_devices()

check(
    "the child is told where to import the backend from",
    str(ROOT) in (captured.get("env") or {}).get("PYTHONPATH", ""),
    f"PYTHONPATH was {(captured.get('env') or {}).get('PYTHONPATH')!r}",
)
check(
    "the child keeps the rest of the environment",
    "PATH" in (captured.get("env") or {}),
    "dropping PATH would stop the child loading its audio libraries",
)
check(
    "the child cannot hang the request forever",
    isinstance(captured.get("timeout"), (int, float)) and captured["timeout"] > 0,
    f"timeout was {captured.get('timeout')!r}",
)

print()
if failures:
    print(f"{len(failures)} FAILED")
    for f in failures:
        print(f"  {f}")
    raise SystemExit(1)

print("ALL PASS")
print(f"({len(devices)} devices enumerated in a child, {len(cases)} failure modes handled)")
