"""Windows-on-ARM: the x64 build must resolve its own binaries, not the host's.

What broke
----------
1.0.77 and 1.0.79 do not start on any ARM64 PC. ``sounddevice`` chooses its bundled
PortAudio DLL from ``platform.machine()``, and on Windows CPython answers that from WMI's
``Win32_Processor.Architecture`` - falling back to ``PROCESSOR_ARCHITEW6432`` only when
that query raises - so every path describes the *host* CPU. An x64 build on a Snapdragon
therefore asked for ``libportaudioarm64.dll``, a file the ``win_amd64`` wheel does not
contain and an x64 process could not load anyway, and the backend died at
``server/audio.py`` import time.

What these checks pin
---------------------
The binding assertion is the resolved path that sounddevice itself reports through
``_libname`` - not a reimplementation of its resolution logic. A test that models lines
77-85 of one sounddevice version would stay green while the product broke on an upgrade
that moved them, which is the "tested the library instead of the product" mistake this
project has already made once.

Crucially that assertion is made against ``server.audio`` as well as against the helper.
An earlier version of this file exercised ``native.py`` alone, and reverting
``server/audio.py`` to a bare ``import sounddevice as sd`` - the exact regression this
suite exists to prevent - left all 11 suites green.

The second group guards the fix's own worst failure mode. ``platform.machine()`` says
"AMD64" and ``sysconfig.get_platform()`` says "win-amd64"; comparing them unnormalised
reports every native x64 machine as emulated, which would show a "expect it to be slow"
notice to the entire x64 user base and make ``bench/bench_arm.py`` refuse to run on the
machine that produced its own x64 baseline.
"""

from __future__ import annotations

import contextlib
import io
import os
import platform
import re
import shutil
import sys
import sysconfig
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tests._isolate  # noqa: F401,E402

from server import hardware, native  # noqa: E402

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    if not ok:
        failures += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  -> ' + detail) if detail else ''}")


def _forget_sounddevice() -> None:
    """Drop the cached modules so the next import re-runs its DLL resolution."""
    for name in ("sounddevice", "_sounddevice_data", "server.audio"):
        sys.modules.pop(name, None)


def _with_host(machine_name: str, fn):
    """Run ``fn`` with ``platform.machine()`` claiming ``machine_name``.

    That is precisely what a Snapdragon does to an emulated x64 process, so forcing it
    here reproduces the shipped crash on x64 hardware.
    """
    _forget_sounddevice()
    original = platform.machine
    platform.machine = lambda: machine_name
    try:
        return fn()
    finally:
        platform.machine = original
        _forget_sounddevice()


# The DLL that belongs to *this* interpreter, whichever SKU is running the suite.
expected_dll = (
    "libportaudioarm64.dll"
    if sysconfig.get_platform() == "win-arm64"
    else "libportaudio64bit.dll"
)

print(f"\ninterpreter: {sysconfig.get_platform()}   host: {platform.machine()}")
print(f"expecting:   {expected_dll}\n")

print("-- PortAudio resolution (the shipped crash) --")

resolved = _with_host("ARM64", lambda: native.import_sounddevice()._libname)
check(
    "a host claiming ARM64 still resolves this process's PortAudio",
    os.path.basename(resolved).lower() == expected_dll,
    resolved,
)
check("the resolved PortAudio exists on disk", os.path.isfile(resolved), resolved)


# The product, not the helper. Without this, reverting server/audio.py to a bare
# ``import sounddevice as sd`` leaves the whole suite green - which is exactly what
# happened when this file was first written.
def _product_libname():
    import server.audio

    return server.audio.sd._libname


product_resolved = _with_host("ARM64", _product_libname)
check(
    "server.audio itself resolves this process's PortAudio",
    os.path.basename(product_resolved).lower() == expected_dll,
    product_resolved,
)

# The same forcing without the shim. While this raises, the workaround is still load
# bearing; if sounddevice ever fixes its own resolution this check fails and server/native.py
# can be deleted, which is the only signal that would tell us so.
def _bare_import():
    try:
        import sounddevice  # noqa: F401

        return None
    except Exception as exc:  # noqa: BLE001 - the failure is the subject
        return exc


bare = _with_host("ARM64", _bare_import)
if sysconfig.get_platform() == "win-amd64":
    check(
        "a bare import still fails, so the workaround is still needed",
        isinstance(bare, Exception),
        "sounddevice resolves correctly now - server/native.py may be removable"
        if bare is None
        else str(bare)[:90],
    )

before_fn = platform.machine
native.import_sounddevice()
check(
    "platform.machine is restored by identity, not merely by value",
    platform.machine is before_fn,
    "a leaked patch still returns the right string on this machine, so comparing "
    "values can only fail on the Snapdragon that never runs this suite",
)

print("\n-- architecture normalisation (the fix's own failure mode) --")

check(
    "AMD64 and x64 are the same architecture",
    hardware._canonical("AMD64") == hardware._canonical("x64"),
    hardware._canonical("AMD64"),
)
check(
    "ARM64 and aarch64 are the same architecture",
    hardware._canonical("ARM64") == hardware._canonical("AARCH64"),
    hardware._canonical("ARM64"),
)
check(
    "this machine is not reported as emulated",
    hardware.is_emulated() is False,
    f"process={hardware.process_machine()} host={hardware.native_machine()}",
)
check(
    "process and host agree on a native machine",
    hardware.process_machine() == hardware.native_machine(),
    f"{hardware.process_machine()} vs {hardware.native_machine()}",
)

# The other half of the same coin, and the one the Snapdragon actually exercises: an x64
# interpreter whose host reports ARM64 really is emulated and must say so, or the "expect
# it to be slow" notice never appears on the machine it was written for. Normalising both
# sides must not flatten this case away.
if sysconfig.get_platform() == "win-amd64":
    _saved_machine = platform.machine
    platform.machine = lambda: "ARM64"
    try:
        check(
            "an x64 interpreter on an ARM64 host IS reported as emulated",
            hardware.is_emulated() is True,
            f"process={hardware.process_machine()} host={hardware.native_machine()}",
        )
    finally:
        platform.machine = _saved_machine

print("\n-- the contested API stays out of control flow --")


def _must_not_be_called():
    raise AssertionError("IsWow64Process2 must not carry control flow")


_original_machines = hardware._machines
hardware._machines = _must_not_be_called
try:
    check(
        "process_machine does not consult IsWow64Process2",
        hardware.process_machine() == hardware._canonical(native.process_machine_name()),
    )
    check("native_machine does not consult IsWow64Process2", bool(hardware.native_machine()))
    check("is_emulated does not consult IsWow64Process2", hardware.is_emulated() is False)
finally:
    hardware._machines = _original_machines

print("\n-- platform tag mapping --")

_original_get_platform = sysconfig.get_platform
for tag, machine in (("win-amd64", "AMD64"), ("win-arm64", "ARM64"), ("win32", "x86")):
    sysconfig.get_platform = lambda t=tag: t
    try:
        check(f"{tag} -> {machine}", native.process_machine_name() == machine)
    finally:
        sysconfig.get_platform = _original_get_platform

sysconfig.get_platform = lambda: "some-future-tag"
try:
    check(
        "an unknown tag falls back to a non-ARM64 answer",
        native.process_machine_name() == "AMD64",
        "so resolution falls through to platform.architecture(), which is process-relative",
    )
finally:
    sysconfig.get_platform = _original_get_platform

print("\n-- the preflight probe must agree with the product --")

# The probe exists to be believed. Importing sounddevice raw here would report FAIL on
# exactly the machine the shim fixes, sending the next round-trip chasing a fix that
# already works.
probes = _with_host("ARM64", lambda: native.probe(include_heavy=False))
by_name = {p.name: p for p in probes}

check("the probe covers sounddevice", "sounddevice" in by_name)
check(
    "the probe reports sounddevice loading under a claimed ARM64 host",
    by_name["sounddevice"].ok,
    by_name["sounddevice"].detail[:90],
)
for lazy in ("soxr", "pyaudiowpatch"):
    check(f"the probe covers {lazy}, which the product imports lazily", lazy in by_name)

_missing = native._probe_one("onnxruntime_genai")
check(
    "an allow-listed absent module is reported, never raised",
    _missing.status in ("absent", "ok"),
    _missing.detail,
)
check(
    "an allow-listed absent module is not called a failure",
    _missing.ok is True,
    "an x64 build ships no onnxruntime_genai; flagging that red trains readers to skim",
)

_unexpected = native._probe_one("a_module_that_is_not_installed_anywhere")
check(
    "a module missing WITHOUT being allow-listed is a failure",
    _unexpected.status == "FAIL",
    "otherwise a packaging regression that dropped faster_whisper renders as a benign gap",
)

# A module that exists but blows up on import is the case that actually matters, and it is
# not the same code path as one that is simply absent.
_boomdir = tempfile.mkdtemp(prefix="sunno-probe-")
Path(_boomdir, "sunno_probe_boom.py").write_text(
    "raise OSError(\"simulated DLL load failure\")", encoding="utf-8"
)
sys.path.insert(0, _boomdir)
try:
    _boom = native._probe_one("sunno_probe_boom")
    check("a module that fails to load is reported as FAIL", _boom.status == "FAIL", _boom.detail)
    check("a failing module reports ok False", _boom.ok is False)
finally:
    sys.path.remove(_boomdir)
    sys.modules.pop("sunno_probe_boom", None)
    shutil.rmtree(_boomdir, ignore_errors=True)
check(
    "the heavy set is excluded unless asked for",
    all(p.name not in native.HEAVY for p in probes),
)
check(
    "the heavy set is included when asked for",
    {p.name for p in native.probe(include_heavy=True)} >= set(native.HEAVY),
)

print("\n-- the failure report must survive the frontend's filter --")


def _is_diagnostic(line: str) -> bool:
    """A faithful port of BackendHost.IsDiagnostic (app/Services/BackendHost.cs:490).

    Duplicated here deliberately. The report is written in Python and filtered in C#, and
    nothing else makes that contract fail loudly: the first version of this report used no
    marker at all, so every one of its lines was dropped and the user would have seen the
    same three lines 1.0.79 showed him.
    """
    if line.startswith('  File "'):
        return True
    text = line.lstrip()
    return (
        text.lower().startswith("[fatal]")
        or text.lower().startswith("[error]")
        or text.startswith("Traceback")
        or re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt)\s*:", text) is not None
    )


_buffer = io.StringIO()
_boomdir2 = tempfile.mkdtemp(prefix="sunno-report-")
Path(_boomdir2, "sunno_report_boom.py").write_text(
    "raise OSError(\"cannot load library 'libportaudioarm64.dll': error 0x7e\")",
    encoding="utf-8",
)
sys.path.insert(0, _boomdir2)
_saved_cheap, _saved_heavy = native.CHEAP, native.HEAVY
native.CHEAP, native.HEAVY = ("numpy", "sunno_report_boom"), ()
try:
    with contextlib.redirect_stdout(_buffer):
        native.print_startup_failure(
            "Traceback (most recent call last):\n"
            '  File "server/audio.py", line 21, in <module>\n'
            "OSError: cannot load library 'libportaudioarm64.dll': error 0x7e\n"
        )
finally:
    native.CHEAP, native.HEAVY = _saved_cheap, _saved_heavy
    sys.path.remove(_boomdir2)
    sys.modules.pop("sunno_report_boom", None)
    shutil.rmtree(_boomdir2, ignore_errors=True)

_lines = _buffer.getvalue().splitlines()
_diagnostic = [ln for ln in _lines if _is_diagnostic(ln)]

check("the report emits something at all", len(_lines) > 10, f"{len(_lines)} lines")
check(
    "some lines survive the frontend's diagnostic filter",
    len(_diagnostic) > 0,
    f"{len(_diagnostic)} of {len(_lines)} lines pass IsDiagnostic",
)

# RecentDiagnostics keeps only the last three that pass the filter, so those three are the
# entire message the user gets. They must be the verdict, not the tail of the probe list.
# RecentDiagnostics keeps only the last three that pass the filter, so those three are the
# entire message the user gets. The contract is not "all three are mine" - the traceback's
# own final line is the single most useful thing there and it is welcome. What must hold is
# that the summary is not pushed out of the window by anything printed after it.
_window = _diagnostic[-3:]
_marked = [ln for ln in _diagnostic if ln.lstrip().lower().startswith("[error]")]
check(
    "every marked summary line lands inside the three the user sees",
    all(ln in _window for ln in _marked),
    f"{len(_marked)} marked, window holds {sum(1 for ln in _window if ln in _marked)}",
)
check(
    "the report's final line is the marked summary, so nothing follows it",
    _diagnostic[-1].lstrip().lower().startswith("[error]"),
    _diagnostic[-1].strip()[:70],
)
check(
    "the summary names the dependency that failed",
    any("sunno_report_boom" in ln for ln in _window),
    " | ".join(ln.strip()[:70] for ln in _window),
)
check(
    "the summary points at the full report",
    any("backend.log" in ln for ln in _window),
)

print(f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILURE(S)'}")
sys.exit(1 if failures else 0)
