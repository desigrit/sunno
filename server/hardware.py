"""Which compute device to use, and how far behind captions will run on it.

Two jobs:

1. Pick a device. Most Windows PCs have no NVIDIA GPU, and this app previously hardcoded
   CUDA and raised at startup without one, an install that could never caption anything.

2. Predict decode lag per model, so the picker can say what each choice costs *before* the
   user spends several gigabytes finding out. On this hardware the spread is large: about
   0.6 s for large-v3 on a GPU versus about 4.4 s for the same model on CPU. A four-second
   lag is fine for captioning a recorded video and useless for following a conversation,
   and only the user knows which they are doing.

What the number means, precisely: the time to decode one utterance. It is not the whole
end-to-end delay, since endpointing waits ``end_silence_ms`` before finalising.

Which decode, though, is where this gets slippery, so be exact. The tables below come from
``bench/bench_latency.py``, which asks for beam 1 and a language and takes everything else
at faster-whisper's defaults. That is closer to the provisional pass than to the final one,
but it is neither pass as shipped: the library default ``temperature`` already carries the
full fallback ladder that the provisional pass explicitly switches off, the provisional
pass carries an initial prompt once any vocabulary or context exists where the bench never
does, and the bench conditions on previous text where the app never does.

Two cautions, both measured, both currently unhandled.

Lag is **not** flat with utterance length. An earlier version of this comment claimed 2 s
and 8 s utterances land within noise of each other because Whisper pads every window to
30 s. Only the encoder behaves that way. The decoder is autoregressive, so it scales with
how many tokens come out, and on a Quadro RTX 8000 the base model measures 44, 43, 67, 99
and 155 ms for 1, 2, 5, 10 and 20 s of speech. That is 3.6x across the measured range, and
``config.py`` permits utterances shorter still, down to ``min_utterance_ms``. So one figure
per model per device summarises a spread rather than reporting a constant.

The tables and ``record_latency`` also describe different work. The tables are beam 1
without word timestamps. ``record_latency`` is called for finals only, and the final pass
raises the beam to 5 and turns word timestamps on. Those two changes on their own, timed
against the same clip with both arms otherwise at faster-whisper's defaults, so that
temperature, prompt and previous-text conditioning were identical in each, measured 1.4x to
3.4x more expensive on the same card, the multiplier growing with utterance length.

The consequence is visible without re-running anything. On the dev box ``hardware.json``
holds a measured ``cuda:base`` of 249 ms against a table entry of 55 ms, a 4.5x gap on one
model on one machine. A measured figure and a table figure are not on one scale, and
``estimated_lag_ms`` can hand back one of each inside a single list.
"""

from __future__ import annotations

import functools
import os

# Measured on a Quadro RTX 8000 (float16) and an i9-14900K (int8), faster-whisper 1.x,
# greedy decode, best of three, utterances of 2/4/8 s averaged. Rerun bench/bench_latency.py
# to regenerate.
#
# base was added later, and its run does not reconcile cleanly with the four rows above. The
# same pass measured small at 1213 ms against the 810 filed here - about 1.5x slow - while
# cpu_score moved only 73.0 to 67.32, or 8%. So the probe caught a sixth of the drift and the
# rest is unexplained load.
#
# Anchoring base to small within that run instead (both measured in the same pass, so load
# cancels) puts it near 294 ms; an earlier run measured 350; the raw figure scaled by cpu_score
# alone gives 440. 405 is a deliberately conservative pick from that range, because telling
# someone a model keeps up when it does not is the failure that matters here.
#
# Its 4-thread figure is derived from small's 4/16 ratio rather than measured: base's 4-thread
# pass came out faster than its 16-thread pass, which cannot be right, and the same pass has
# visible garbage elsewhere (distil-large-v3 at 8.36 s against 3.79 and 3.71).
_LAG_MS_CUDA: dict[str, int] = {
    "base": 55,
    "small": 250,
    "medium": 550,
    "distil-large-v3": 500,
    "large-v3": 650,
    # Same figures as the processor rows, because it is the same work: these models run on
    # the processor whatever else the machine has (asr_stream.py pins provider="cpu"), so
    # a graphics card does not change them. Without a row here they fell through to the
    # unknown-model default and the picker told everyone with a GPU that a model which
    # decodes in well under a second was five seconds behind.
    "stream-en": 135,
    "stream-en-kroko": 120,
}

# CPU lag at two thread counts on the reference machine. Scaling with cores is strongly
# sublinear — quadrupling threads takes large-v3 from 4.5 s only to 4.4 s — so these are
# interpolated on a log scale rather than divided by core count.
#
# The two stream- rows are streaming transducers rather than Whisper checkpoints, measured
# by bench/bench_stream_latency.py over the same 2, 4 and 8 second clips as the rows around
# them, on testdata/3-two-speakers-en.wav at four threads: stream-en 135 ms, Kroko 120 ms.
# Both were restated together in one run, because two figures in the same column measured
# different ways are worse than either being stale, and an earlier stream-en figure of 180
# no longer reproduced on the same script and clip.
#
# Eight consecutive passes, not three, and the difference mattered. A first attempt filed
# Kroko at 80 ms from a median of three, which a review could not reproduce: the model has
# an occasional fast mode and short runs land in one mode or the other, so three passes
# pinned nothing. Over eight, Kroko sits at 120-125 and stream-en at 132-144. Filing the
# common mode rather than the best one also matters beyond honesty: at 80 ms Kroko was the
# quickest entry in the whole catalogue by a wide margin, which is what let it win the
# picker's fastest-first branch outright.
#
# They carry the same value at BOTH thread counts, and the reason is not that they measured
# the same. They did not; more threads are worse for these models, because the work per
# chunk is too small to spread and the coordination costs more than it returns. The reason
# is that asr_stream._threads() caps the recogniser at four whatever the machine has, so the
# sixteen-thread column is unreachable for these models and filing a sixteen-thread
# measurement there would describe something the app never does. That couples this table to
# that cap, so tests/test_stream_engine.py asserts the coupling rather than leaving it to
# this comment: raise the cap and the test fails.
#
# Note the shape as well as the number. Both have almost no fixed cost and scale close to
# linearly, about 30 ms per second of speech, so the means above cover 2 to 8 second
# utterances and a 20 second one, which config.py still permits, costs nearer 610 ms.
# Whisper is the other way round, mostly fixed cost, so the same single figure hides a
# different spread for each.
#
# Deliberately NOT the streaming figure. Fed audio continuously these models put words on
# screen while someone is still speaking, which measured near zero lag, but the pipeline
# waits for an endpoint before it decodes anything, so nobody would see that today. Filing
# it here would make them win on every machine, including ones with a graphics card,
# on the strength of a number the app cannot yet deliver.
_LAG_MS_CPU_4: dict[str, int] = {
    "stream-en": 135,
    "stream-en-kroko": 120,
    "base": 730,
    "small": 1450,
    "medium": 3460,
    "distil-large-v3": 4370,
    "large-v3": 4540,
}
_LAG_MS_CPU_16: dict[str, int] = {
    "stream-en": 135,
    "stream-en-kroko": 120,
    "base": 405,
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
    """Where the measured cpu_score and observed lags are cached.

    Goes through paths.data_dir() rather than reading LOCALAPPDATA directly, so it honours
    the Sunno_DATA_DIR override the rest of the Python side respects. It did not, and the
    consequence was not theoretical: the test suite could not be isolated from the real
    profile, so running the tests wrote a cpu_score measured under test load into the state
    a real launch then reads to quote model latencies on the first-run screen.

    That is the whole of what this fixes. It does NOT make a redirected profile coherent,
    because the frontend does not know about the override at all: Diagnostics.HardwareJson
    and AppSettings both build LocalApplicationData/Sunno paths by hand, so under
    Sunno_DATA_DIR the backend writes here while the diagnostics report reads the old
    location and prints "(not measured yet)". Making that whole story work is a change to
    the C# side and is not attempted here.
    """
    from .paths import data_dir

    return data_dir() / "hardware.json"


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

    Called for finalised utterances only, on the reasoning that a partial decodes greedily
    and would understate the wait at the end of a sentence.

    Be aware that this leaves the recorded figures on a different scale from the shipped
    tables, which are beam 1 without word timestamps. See the module docstring. Recording
    partials instead would match the tables but would describe a different wait, so neither
    choice is right until the two sources are deliberately reconciled.
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

    Models whose publisher has declared no licence are removed before either branch runs.
    THIRD-PARTY-NOTICES.md tells the reader none of them is a default, and that has to be
    true of the fallback as well as the scan: the fallback picks the quickest model in the
    list, and the quickest streaming model is one of the undeclared ones, so ordering alone
    would have handed it every machine too slow for anything else.
    """
    from .models import auto_selectable

    device = device or resolve_device()
    # The `or catalog_ids` is defensive rather than expected. It cannot trigger while any
    # Whisper model is catalogued, and returning nothing is not an option a caller can use.
    allowed = [m for m in catalog_ids if auto_selectable(m)] or list(catalog_ids)

    for model_id in allowed:
        if estimated_lag_ms(model_id, device) <= RESPONSIVE_LAG_MS:
            return model_id
    return min(allowed, key=lambda m: estimated_lag_ms(m, device))


# Above this, captions arrive too late to follow a live conversation. Chosen to match the
# measured gap rather than a round number that sounds nice: on a GPU every model lands
# under 0.7 s, while the slowest CPU configurations sit above 4 s, so anything in between
# separates the two cleanly.
RESPONSIVE_LAG_MS = 1000
