"""Which compute device to use, and how far behind captions will run on it.

Two jobs:

1. Pick a device. Most Windows PCs have no NVIDIA GPU, and this app previously hardcoded
   CUDA and raised at startup without one — an install that could never caption anything.

2. Predict decode lag per model, so the picker can say what each choice costs *before* the
   user spends several gigabytes finding out. On this hardware the spread is large: about
   0.6 s for large-v3 on a GPU versus about 4.4 s for the same model on CPU. A four-second
   lag is fine for captioning a recorded video and useless for following a conversation,
   and only the user knows which they are doing.

What the number means, precisely: the time to decode one utterance on the *provisional*
pass, which is greedy (beam 1) and is what puts words on screen first. It is not the whole
end-to-end delay — endpointing waits ``end_silence_ms`` before finalising, and the final
pass re-decodes with beam search — but those add roughly the same amount for every model on
every machine, so they would shift all the figures without changing any comparison. Decode
time is the part that actually differs between a choice and its alternatives.

The numbers below are measured, not estimated. Notably the lag barely varies with how long
someone spoke (2 s and 8 s utterances land within noise of each other) because Whisper pads
every window to 30 s, so one figure per model per device is an honest summary.
"""

from __future__ import annotations

import functools
import os

# Measured on a Quadro RTX 8000 (float16) and an i9-14900K (int8), faster-whisper 1.x,
# greedy decode, best of three, utterances of 2/4/8 s averaged. Rerun bench/bench_latency.py
# to regenerate.
_LAG_MS_CUDA: dict[str, int] = {
    "small": 250,
    "medium": 550,
    "distil-large-v3": 500,
    "large-v3": 650,
}

# CPU lag at two thread counts on the reference machine. Scaling with cores is strongly
# sublinear — quadrupling threads takes large-v3 from 4.5 s only to 4.4 s — so these are
# interpolated on a log scale rather than divided by core count.
_LAG_MS_CPU_4: dict[str, int] = {
    "small": 1450,
    "medium": 3460,
    "distil-large-v3": 4370,
    "large-v3": 4540,
}
_LAG_MS_CPU_16: dict[str, int] = {
    "small": 810,
    "medium": 2260,
    "distil-large-v3": 3610,
    "large-v3": 4400,
}

# BLAS matmul score of the reference CPU, from the same benchmark. A machine that scores
# half this is assumed to take roughly twice as long.
_REFERENCE_CPU_SCORE = 73.0

_UNKNOWN_MODEL_LAG_MS = 5000


_MACHINE_NAMES: dict[int, str] = {
    0x0000: "unknown",
    0x014C: "x86",
    0x8664: "x64",
    0x01C4: "ARM32",
    0xAA64: "ARM64",
}


def _machines() -> tuple[str, str]:
    """(process machine, native machine) as Windows reports them.

    Deliberately not ``platform.machine()``. That resolves through PROCESSOR_ARCHITECTURE and
    PROCESSOR_ARCHITEW6432, both process-relative, so an x64 build running under emulation on
    an ARM64 PC reports "AMD64" - precisely the blind spot this exists to close.

    IsWow64Process2 answers both halves at once, and the pair is what carries the meaning:
    pProcessMachine is IMAGE_FILE_MACHINE_UNKNOWN when the process is *not* emulated, so
    native alone cannot distinguish an emulated x64 build from a native ARM64 one. Both run on
    an ARM64 machine; only one of them is slow.
    """
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.IsWow64Process2.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.USHORT),
            ctypes.POINTER(wintypes.USHORT),
        ]
        kernel32.IsWow64Process2.restype = wintypes.BOOL

        process = wintypes.USHORT()
        native = wintypes.USHORT()
        if not kernel32.IsWow64Process2(
            kernel32.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(native)
        ):
            return "unknown", "unknown"

        native_name = _MACHINE_NAMES.get(native.value, f"0x{native.value:04x}")
        # UNKNOWN here means "not running under WOW64", i.e. the process is native - so the
        # process machine is the native one.
        if process.value == 0:
            return native_name, native_name
        return _MACHINE_NAMES.get(process.value, f"0x{process.value:04x}"), native_name
    except Exception:
        # Older Windows, a non-Windows host, or a blocked call. An unknown answer is fine:
        # every caller treats this as context for a log line, never as control flow.
        return "unknown", "unknown"


def native_machine() -> str:
    """The machine Windows is really running on, not the one this process is emulating."""
    return _machines()[1]


def process_machine() -> str:
    """The machine this process is built for."""
    return _machines()[0]


def is_emulated() -> bool:
    """True when this process is being emulated, e.g. an x64 build on an ARM64 PC.

    Compares the pair rather than testing for ARM64. A native ARM64 build also runs on a
    machine whose native architecture is ARM64, and reporting that as emulation would put a
    "this will be slow" notice on the one build that is not.
    """
    process, native = _machines()
    if "unknown" in (process, native):
        return False
    return process != native


@functools.lru_cache(maxsize=1)
def engine_importable() -> bool:
    """Whether the inference engine's native extension loads at all.

    Deliberately separate from :func:`has_cuda`. Loadability is not a GPU question, but it used
    to be observable only as a side effect of the CUDA probe - which the user can switch off with
    Force CPU, silencing the answer on the one machine that most needs it. CTranslate2 publishes
    no win_arm64 wheel, so an ARM PC is the likely cause of a failure here.

    Cached so the report is made once per process rather than on every device resolution.
    """
    try:
        import ctranslate2  # noqa: F401

        return True
    except (ImportError, OSError) as exc:
        # Split out from the catch-all below on purpose. This is the one failure here that does
        # not mean "no GPU": the extension itself would not load, so the CPU path is dead too and
        # the process will die loading the model. Reporting a healthy-looking "cpu" and saying
        # nothing is what made that death unexplainable.
        print(
            f"[error] ctranslate2 could not be loaded (native machine {native_machine()}): {exc}",
            flush=True,
        )
        return False
    except Exception:
        return False


def has_cuda() -> bool:
    """Whether CTranslate2 can actually run on a GPU here.

    Deliberately not a check for an NVIDIA card: the card can be present while the CUDA
    payload we ship is missing or unloadable, and the only answer that matters is whether
    a model would load.
    """
    if not engine_importable():
        return False

    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() <= 0:
            return False
        # Device count is reported by the driver and says nothing about our own DLLs, so
        # confirm the compute type we would actually request is offered.
        return "float16" in ctranslate2.get_supported_compute_types("cuda")
    except Exception:
        return False


def resolve_device(preference: str = "auto") -> str:
    """Turn a preference into a device that will really load."""
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        # An explicit request still gets checked: failing here with a clear device string
        # beats failing later inside CTranslate2 with a missing-DLL error.
        return "cuda" if has_cuda() else "cpu"
    return "cuda" if has_cuda() else "cpu"


def compute_type_for(device: str) -> str:
    """Best compute type for the device.

    float16 on CUDA. int8 on CPU, where it is the only quantisation that actually pays —
    on Turing GPUs int8 measured ~15% *slower* than float16, but on CPU it is the
    difference between usable and not.
    """
    return "float16" if device == "cuda" else "int8"


def cpu_threads() -> int:
    """Threads CTranslate2 will get. Mirrors its own default so estimates match reality."""
    override = os.environ.get("OMP_NUM_THREADS")
    if override and override.isdigit() and int(override) > 0:
        return int(override)
    return os.cpu_count() or 4


@functools.lru_cache(maxsize=1)
def cpu_score() -> float:
    """Cheap BLAS throughput probe, used to rescale the reference CPU timings.

    Cached to disk and kept at the best value ever seen, because what we want is the
    machine's capability rather than how busy it happens to be right now. Measured live on
    a loaded machine the score sags enough to move an estimate across a round number, which
    would make the picker's figures wander between launches for no real reason.

    Falls back to the reference score so a probe failure degrades to "assume average"
    rather than to a confidently wrong estimate.
    """
    cached = _read_cached_score()
    measured = _measure_cpu_score()
    best = max(cached or 0.0, measured)
    if best > (cached or 0.0):
        _write_cached_score(best)
    return best or _REFERENCE_CPU_SCORE


def _measure_cpu_score() -> float:
    try:
        import time

        import numpy as np

        a = np.random.rand(512, 512).astype(np.float32)
        b = np.random.rand(512, 512).astype(np.float32)
        a @ b  # let BLAS spin up its threads before timing
        best = None
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(20):
                a @ b
            elapsed = time.perf_counter() - start
            best = elapsed if best is None else min(best, elapsed)
        return 1.0 / best if best else _REFERENCE_CPU_SCORE
    except Exception:
        return _REFERENCE_CPU_SCORE


def _read_cached_score() -> float | None:
    score = float(_read_state().get("cpu_score", 0) or 0)
    return score if score > 0 else None


def _write_cached_score(score: float) -> None:
    state = _read_state()
    state["cpu_score"] = round(score, 2)
    _write_state(state)


def _score_path():
    from pathlib import Path

    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / "Sunno" / "hardware.json"


def _read_state() -> dict:
    try:
        import json

        return json.loads(_score_path().read_text())
    except Exception:
        return {}


def _write_state(state: dict) -> None:
    try:
        import json

        path = _score_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state))
    except Exception:
        pass   # losing the cache costs a little accuracy, never correctness


# Observed decode times, keyed "<device>:<model>". Kept in memory as a short window and
# flushed to disk as a median, so a single slow decode — a background build, a GPU busy with
# a game — can't move the figure the picker shows.
_OBSERVED: dict[str, list[float]] = {}
_OBSERVE_WINDOW = 15
_OBSERVE_MIN = 5


def record_latency(model_id: str, device: str, ms: float) -> None:
    """Note how long a real decode actually took on this machine.

    Called for finalised utterances only. Partials are decoded greedily and would report a
    faster figure than the one the user waits for at the end of a sentence, which is what
    the picker's number claims to describe.
    """
    if ms <= 0 or ms > 60_000:
        return   # a wild value means something else went wrong; don't poison the estimate

    key = f"{device}:{model_id}"
    samples = _OBSERVED.setdefault(key, [])
    samples.append(float(ms))
    if len(samples) > _OBSERVE_WINDOW:
        del samples[0]
    if len(samples) < _OBSERVE_MIN:
        return

    ordered = sorted(samples)
    median = ordered[len(ordered) // 2]
    state = _read_state()
    observed = state.setdefault("observed_lag_ms", {})
    previous = observed.get(key)
    # Only rewrite when it has actually moved, to avoid a disk write per utterance.
    if previous is None or abs(previous - median) > 50:
        observed[key] = int(round(median))
        _write_state(state)


def measured_lag_ms(model_id: str, device: str) -> int | None:
    """What this machine has been seen doing, or None if it has never run this model."""
    key = f"{device}:{model_id}"
    samples = _OBSERVED.get(key)
    if samples and len(samples) >= _OBSERVE_MIN:
        return int(round(sorted(samples)[len(samples) // 2]))
    value = _read_state().get("observed_lag_ms", {}).get(key)
    return int(value) if value else None


def estimated_lag_ms(model_id: str, device: str | None = None) -> int:
    """Roughly how long after someone stops speaking their words appear.

    Prefers what this machine has actually been measured doing, and falls back to the
    shipped table only for a model that has never run here. That matters most on a GPU,
    where the table cannot adapt: CPU figures are rescaled by a benchmark, but every CUDA
    machine would otherwise be quoted the numbers from the one card these were recorded on,
    and the spread across NVIDIA generations is far wider than the spread between models.
    """
    device = device or resolve_device()

    measured = measured_lag_ms(model_id, device)
    if measured is not None:
        return measured

    if device == "cuda":
        return _LAG_MS_CUDA.get(model_id, _UNKNOWN_MODEL_LAG_MS)

    low = _LAG_MS_CPU_4.get(model_id)
    high = _LAG_MS_CPU_16.get(model_id)
    if low is None or high is None:
        return _UNKNOWN_MODEL_LAG_MS

    # Interpolate between the two measured thread counts, then clamp: below 4 threads the
    # curve is unmeasured, and above 16 more cores stop helping.
    threads = max(1, cpu_threads())
    if threads <= 4:
        base = low
    elif threads >= 16:
        base = high
    else:
        # Log interpolation: 4->16 is two doublings, and each buys less than the last.
        import math

        span = (math.log2(threads) - 2.0) / 2.0
        base = low + (high - low) * span

    # Rescale for a CPU faster or slower than the one these numbers came from.
    score = cpu_score()
    if score > 0:
        base *= _REFERENCE_CPU_SCORE / score

    return int(round(base))


def describe_lag(lag_ms: int) -> str:
    """Human phrasing for the picker. Deliberately coarse — it is an estimate, and a figure
    quoted to three digits would imply a precision that a busy machine does not have."""
    if lag_ms < 1000:
        return f"about {lag_ms / 1000:.1f}s behind"
    return f"about {round(lag_ms / 1000)}s behind"


def default_model(catalog_ids: list[str], device: str | None = None) -> str:
    """The model to start on when the user has never chosen one.

    The most accurate model that still keeps up with conversation, falling back to the
    fastest available when nothing does. Catalog order is best-accuracy-first, so this is
    a scan for the first entry inside the budget.

    Not simply "the fastest": on a GPU every model is comfortably inside the budget, and
    starting such a machine on the least accurate model would be a downgrade for no reason.
    """
    device = device or resolve_device()
    for model_id in catalog_ids:
        if estimated_lag_ms(model_id, device) <= RESPONSIVE_LAG_MS:
            return model_id
    return min(catalog_ids, key=lambda m: estimated_lag_ms(m, device))


# Above this, captions arrive too late to follow a live conversation. Chosen to match the
# measured gap rather than a round number that sounds nice: on a GPU every model lands
# under 0.7 s, while the slowest CPU configurations sit above 4 s, so anything in between
# separates the two cleanly.
RESPONSIVE_LAG_MS = 1000
