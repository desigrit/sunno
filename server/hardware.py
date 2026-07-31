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
# greedy decode, best of three, utterances of 2/4/8 s averaged. Rerun files/bench_latency.py
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


def has_cuda() -> bool:
    """Whether CTranslate2 can actually run on a GPU here.

    Deliberately not a check for an NVIDIA card: the card can be present while the CUDA
    payload we ship is missing or unloadable, and the only answer that matters is whether
    a model would load.
    """
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


def _score_path():
    from pathlib import Path

    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / "Sunno" / "hardware.json"


def _read_cached_score() -> float | None:
    try:
        import json

        data = json.loads(_score_path().read_text())
        score = float(data.get("cpu_score", 0))
        return score if score > 0 else None
    except Exception:
        return None


def _write_cached_score(score: float) -> None:
    try:
        import json

        path = _score_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"cpu_score": round(score, 2)}))
    except Exception:
        pass   # an un-cached score costs a little accuracy, never correctness


def estimated_lag_ms(model_id: str, device: str | None = None) -> int:
    """Roughly how long after someone stops speaking their words appear.

    An estimate for a model the user has not run yet. Once a model has actually been used
    the pipeline reports real latencies, which should be preferred over this.
    """
    device = device or resolve_device()

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
