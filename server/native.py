"""Loading the native extensions, with the architecture question answered in one place.

Why this exists
---------------
Sunno ships as an x64 package. On a Windows-on-ARM PC that package runs under Prism
emulation, and one dependency resolves a bundled binary by asking which machine this is -
and gets an answer that is true of the machine but false of the process.

``sounddevice`` 0.5.5 picks its PortAudio DLL like this (lines 77-85)::

    if _platform.machine().lower() in ('arm64', 'aarch64'):
        _platform_suffix = 'arm64'
    else:
        _platform_suffix = _platform.architecture()[0]      # -> '64bit'
    _libname = 'libportaudio' + _platform_suffix + '.dll'

``platform.machine()`` is not the process's architecture. On Windows, CPython 3.12's
``_get_machine_win32()`` queries WMI ``Win32_Processor.Architecture`` first - its own
comment says "WOW64 processes mask the native architecture" - and maps index 12 to
``ARM64``; only if that raises ``OSError`` does it fall back to
``PROCESSOR_ARCHITEW6432 or PROCESSOR_ARCHITECTURE``. Every one of those paths describes
the **host CPU**. So on a Snapdragon our x64 build asks for ``libportaudioarm64.dll``. The
``win_amd64`` wheel does not contain that file - it ships only ``libportaudio64bit.dll``
and its ASIO twin - and an x64 process could not load an ARM64 DLL even if it did. The
import raises ``OSError`` 0x7e (ERROR_MOD_NOT_FOUND) at ``audio.py`` import time, so the
backend dies before it can report anything.

That is not hypothetical: it is what 1.0.77 and 1.0.79 do on every ARM64 PC, and the
user's crash report named exactly this file.

The signal used here instead is ``sysconfig.get_platform()``. It is derived from the
``[MSC v.1943 64 bit (AMD64)]`` literal baked into ``sys.version`` when CPython was
compiled, so no amount of emulation can move it, and it is the same tag pip matched when
it chose the wheels now sitting on disk. That makes it self-consistent by construction: it
names the architecture of the binaries actually present, which is the only question being
asked. It is also correct for a native ARM64 build with no special case, because that
build's interpreter reports ``win-arm64`` and its wheel does ship the ARM64 DLL.

Deliberately NOT ``IsWow64Process2``. Microsoft documents ``pProcessMachine`` as
IMAGE_FILE_MACHINE_UNKNOWN when the process "is not a WOW64 process", and first-hand
reports disagree over what an emulated x64 process on ARM64 returns - some say UNKNOWN
with a native of ARM64, which is indistinguishable from a native ARM64 process. A fix
resting on it could compute the same broken filename and waste the round-trip. It survives
in ``hardware.py`` as supplementary diagnostic output, explicitly not load-bearing.
"""

from __future__ import annotations

import platform
import sysconfig
import threading
from dataclasses import dataclass

# sysconfig's platform tag -> the value platform.machine() would report on a machine whose
# *native* architecture matched this interpreter. Only the ARM64 entry changes sounddevice's
# behaviour; the rest fall through to platform.architecture()[0], which is process-relative
# and therefore already correct. They are spelled out anyway so the mapping reads as a fact
# about architectures rather than as a special case for one library.
_TAG_TO_MACHINE = {
    "win-amd64": "AMD64",
    "win-arm64": "ARM64",
    "win32": "x86",
}

# platform.machine is process-global, so two threads importing at once could otherwise save
# each other's lambda as "the original" and leave it patched for the life of the process.
# Deadlock is not reachable through this: the only waiter is another import_sounddevice(),
# which holds no module lock this one needs.
_PATCH_LOCK = threading.Lock()


def process_tag() -> str:
    """The platform tag this interpreter was compiled for, e.g. ``win-amd64``."""
    return sysconfig.get_platform()


def process_machine_name() -> str:
    """This *process's* architecture, in ``platform.machine()`` vocabulary.

    Unknown tags fall back to AMD64 rather than raising. A wrong-but-non-ARM64 answer
    still resolves through ``platform.architecture()[0]``, which is process-relative and
    correct, so the failure mode of this default is "no change", not "broken import".
    """
    return _TAG_TO_MACHINE.get(process_tag(), "AMD64")


def import_sounddevice():
    """Import ``sounddevice`` with its architecture probe answered for this process.

    The patch is narrow on purpose: one attribute, restored in ``finally`` whether the
    import succeeds or raises. sounddevice's module body is small and does not itself
    branch on the machine name for anything except the DLL suffix.

    Safe to call repeatedly - the second call gets the module from ``sys.modules`` and the
    patch is inert. Callers should use this rather than a bare ``import sounddevice`` so
    that the diagnostic probe and the product exercise identical code; a probe that
    imports it raw reports a failure on precisely the machine the shim exists to fix.
    """
    forced = process_machine_name()
    with _PATCH_LOCK:
        original = platform.machine
        platform.machine = lambda: forced
        try:
            import sounddevice

            return sounddevice
        finally:
            platform.machine = original


@dataclass(frozen=True)
class Probe:
    """One native dependency and how it answered.

    Three outcomes, not two. "absent" is not "broken": an x64 build does not ship
    ``onnxruntime_genai``, and reporting that as FAIL would put a red line in every report
    and teach the reader to skim past them.
    """

    name: str
    status: str  # "ok" | "FAIL" | "absent"
    detail: str

    @property
    def ok(self) -> bool:
        return self.status != "FAIL"


# Cheap, and none of them faults in a heavyweight native runtime. soxr and pyaudiowpatch
# are lazy in the product, which is exactly why they are probed here: a "does it start?"
# check never touches them, so system-audio capture could be broken on a machine where
# startup looks perfectly healthy.
CHEAP = ("numpy", "sounddevice", "soxr", "pyaudiowpatch")

# Deferred by design in the product, so a launch that never transcribes never pays for
# them. Importing the set eagerly measured +236 ms on x64, which is why it happens only
# once something has already failed.
#
# These are the modules the product actually imports, which is not the same as the
# libraries underneath: asr.py:117 imports faster_whisper (whose __init__ executes
# ctranslate2 and pulls a native tokenizers.pyd), and asr_onnx.py:44 imports
# onnxruntime_genai. Probing only ctranslate2 could report "ok" while the engine still
# failed to load - the opposite of what this report is for.
HEAVY = (
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "onnxruntime_genai",
    "sherpa_onnx",
    "av",
)


# Modules legitimately not shipped in every build. Anything else that fails to import is a
# real failure, not a benign gap: a packaging regression that dropped faster_whisper would
# otherwise render as a grey "--" and read as intentional.
EXPECTED_ABSENT = frozenset({"onnxruntime_genai"})


def _probe_one(name: str) -> Probe:
    """Import one module, converting any failure into a result rather than an exception.

    ``BaseException`` is not caught: a KeyboardInterrupt during startup should still stop
    the process. Everything else, including the ``OSError`` a missing DLL raises, is a
    finding to report.
    """
    try:
        if name == "sounddevice":
            module = import_sounddevice()
            # The resolved DLL is the fact worth reporting; "imported" alone would not
            # distinguish a correct load from a lucky one.
            return Probe(name, "ok", getattr(module, "_libname", "loaded"))
        __import__(name)
        return Probe(name, "ok", "loaded")
    except ModuleNotFoundError as exc:
        # Only an allow-listed name may be reported as a benign gap. A module that IS
        # installed but whose own dependency has gone also raises ModuleNotFoundError, and
        # calling that "absent" would hide it.
        if name in EXPECTED_ABSENT and exc.name == name:
            return Probe(name, "absent", f"not shipped in this build ({exc.name})")
        return Probe(name, "FAIL", f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        return Probe(name, "FAIL", f"{type(exc).__name__}: {exc}")


def probe(include_heavy: bool = False) -> list[Probe]:
    """Load-test each native dependency independently.

    Independently matters. The failure this was written for stopped the backend at the
    first bad import, so the crash report named one library and said nothing about the
    ones behind it - and the machine that produced it was a plane ride away.
    """
    names = list(CHEAP) + (list(HEAVY) if include_heavy else [])
    return [_probe_one(name) for name in names]


def format_probe(p: Probe, width: int = 18) -> str:
    """One probe as a printable line."""
    label = {"ok": "ok  ", "FAIL": "FAIL", "absent": "--  "}[p.status]
    return f"  {label}  {p.name.ljust(width)}  {p.detail}"


def format_report(probes: list[Probe]) -> list[str]:
    """Probe results as printable lines, longest-name-aligned."""
    if not probes:
        return []
    width = max(len(p.name) for p in probes)
    return [format_probe(p, width) for p in probes]


def architecture_lines() -> list[str]:
    """The architecture facts, as printable lines."""
    from . import hardware

    return [
        f"  process        {hardware.process_machine()}  (tag {process_tag()})",
        f"  host           {hardware.native_machine()}",
        f"  emulated       {hardware.is_emulated()}",
        f"  wow64 probe    {hardware.wow64_machines()}  (diagnostic only, see hardware.py)",
    ]


def print_startup_failure(traceback_text: str) -> None:
    """Print everything known about a failed start, ordered for who reads what.

    Printed and flushed line by line rather than assembled and returned. The deferred
    dependencies fault in native loaders, and a structured exception or access violation
    inside one of those is not catchable by ``except Exception`` - it would take the
    process down mid-report. Building the whole list first would mean that on the one
    machine this handler exists for, the log could end up with no diagnostics at all,
    which is worse than the bare traceback it replaced.

    Order is load-bearing. ``BackendHost.RecentDiagnostics`` keeps only the **last three**
    lines that pass ``BackendHost.IsDiagnostic``, and that is the only sanctioned route
    from here to the InfoBar and the diagnostics export. So the detail goes out first, for
    ``backend.log``, and the two ``[error]`` lines that carry the verdict go out last,
    where the frontend will actually pick them up. Prefixing every line instead would put
    the tail of the probe list in front of the user and drop the verdict.

    The traceback is printed here rather than left to the interpreter for the same reason:
    re-raising would emit it *after* these lines and push the summary out of the window.
    """

    def emit(line: str) -> None:
        print(line, flush=True)

    for line in traceback_text.rstrip().splitlines():
        emit(line)

    emit("")
    emit("  A native dependency failed to load. Architecture first, then every dependency")
    emit("  probed on its own so this names all of them, not just the first.")
    emit("")
    for line in architecture_lines():
        emit(line)
    emit("")

    results = list(probe(include_heavy=False))
    for p in results:
        emit(format_probe(p))
    emit("")
    emit("  -- deferred dependencies (these load native runtimes) --")
    for name in HEAVY:
        p = _probe_one(name)
        results.append(p)
        emit(format_probe(p))

    # Last, and marked, so RecentDiagnostics' final-three window catches these rather than
    # the tail of the list above. See IsDiagnostic in app/Services/BackendHost.cs.
    broken = [p for p in results if p.status == "FAIL"]
    emit("")
    if broken:
        first = broken[0]
        emit(f"[error] Sunno's engine could not start: {first.name} - {first.detail}")
        others = ", ".join(p.name for p in broken[1:]) or "none"
        emit(f"[error] Other failures: {others}. Full report in backend.log.")
    else:
        emit("[error] Sunno's engine could not start, but every dependency probed loaded.")
        emit("[error] The traceback above is the only evidence; see backend.log.")
