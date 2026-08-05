"""Decode latency as a function of utterance length, which is what the picker must predict.

Conversational turns are short. Measuring only a 9.5s clip overstates the lag a user
actually waits for, and a threshold tuned to it would grey out models that are fine.

Also records a cheap CPU score (BLAS matmul) so the shipped table can be rescaled to
whatever machine the app is installed on.
"""

from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[1]

# The venv does not put the pip-installed NVIDIA DLLs on the search path; the server
# registers them on import, so borrow that rather than duplicating the logic.
import sys
sys.path.insert(0, str(ROOT))
from server import cuda_setup  # noqa: E402

try:
    cuda_setup.register_cuda_dlls(required=True)
except Exception as exc:  # noqa: BLE001
    print(f"(no CUDA: {str(exc)[:80]})", flush=True)
SRC = ROOT / "testdata" / "1-two-speakers-en.wav"
DURATIONS = [2.0, 4.0, 8.0]
MODELS = ["small", "medium", "distil-large-v3", "large-v3"]

REPO = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "medium": "Systran/faster-whisper-medium",
    "small": "Systran/faster-whisper-small",
}


def cpu_score() -> float:
    """Fixed BLAS workload; higher is faster. Cheap enough to run at startup."""
    a = np.random.rand(512, 512).astype(np.float32)
    b = np.random.rand(512, 512).astype(np.float32)
    a @ b  # warm up BLAS threads
    best = None
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(20):
            a @ b
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return round(1.0 / best, 2)


def clip(seconds: float) -> np.ndarray:
    with wave.open(str(SRC), "rb") as w:
        sr, nch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    return a[: int(seconds * sr)]


def bench(device: str, compute: str, threads: int | None) -> list[dict]:
    rows = []
    for model_id in MODELS:
        kwargs = {"device": device, "compute_type": compute, "local_files_only": True}
        if threads:
            kwargs["cpu_threads"] = threads
        try:
            model = WhisperModel(REPO[model_id], **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {model_id} {device}: {str(exc)[:80]}", flush=True)
            continue

        for seconds in DURATIONS:
            audio = clip(seconds)
            best = None
            for _ in range(3):
                t0 = time.perf_counter()
                segs, _ = model.transcribe(audio, beam_size=1, language="en")
                "".join(s.text for s in segs)
                dt = time.perf_counter() - t0
                best = dt if best is None else min(best, dt)
            rows.append({"device": device, "threads": threads, "model": model_id,
                         "audio_s": seconds, "lag_s": round(best, 2)})
            print(f"{device:<5} t={threads or '-':<3} {model_id:<18} "
                  f"audio={seconds:4.1f}s  lag={best:5.2f}s", flush=True)
        del model
    return rows


score = cpu_score()
print(f"cpu_score={score}\n", flush=True)

rows = bench("cuda", "float16", None)
rows += bench("cpu", "int8", 4)
rows += bench("cpu", "int8", 16)

out = Path(__file__).with_name("latency_table.json")
out.write_text(json.dumps({"cpu_score": score, "rows": rows}, indent=2))
print(f"\nwrote {out}", flush=True)
