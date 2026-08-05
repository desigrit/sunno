"""Measure Whisper decode speed on CPU so the model picker can be honest about it.

For live captioning the number that matters is the real-time factor (RTF):
decode_seconds / audio_seconds. RTF >= 1 means the backlog grows without bound and
captions fall further behind for as long as someone keeps talking.

Also sweeps thread counts, because this machine's i9-14900K is far above the median
Windows PC and a threshold tuned to it would be wrong for everyone else.
"""

from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[1]
CLIPS = [ROOT / "testdata" / "verify.wav", ROOT / "testdata" / "1-two-speakers-en.wav"]
MODELS = ["small", "medium", "distil-large-v3", "large-v3"]
THREADS = [int(t) for t in (sys.argv[1:] or ["4", "16"])]

REPO = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "medium": "Systran/faster-whisper-medium",
    "small": "Systran/faster-whisper-small",
}


def load(path: Path) -> tuple[np.ndarray, float]:
    with wave.open(str(path), "rb") as w:
        sr, nch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    return a, len(a) / sr


def main() -> None:
    results = []
    for threads in THREADS:
        for model_id in MODELS:
            try:
                t0 = time.perf_counter()
                model = WhisperModel(
                    REPO[model_id], device="cpu", compute_type="int8",
                    cpu_threads=threads, local_files_only=True,
                )
                load_s = time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001
                print(f"SKIP {model_id} threads={threads}: {str(exc)[:90]}", flush=True)
                continue

            for clip in CLIPS:
                if not clip.exists():
                    continue
                audio, duration = load(clip)
                # One warm-up pass, then best of two: first call pays lazy init.
                best = None
                for _ in range(3):
                    t0 = time.perf_counter()
                    segs, _info = model.transcribe(audio, beam_size=1, language="en")
                    text = "".join(s.text for s in segs)
                    dt = time.perf_counter() - t0
                    best = dt if best is None else min(best, dt)
                rtf = best / duration
                results.append({
                    "model": model_id, "threads": threads, "clip": clip.name,
                    "audio_s": round(duration, 2), "decode_s": round(best, 2),
                    "rtf": round(rtf, 3), "load_s": round(load_s, 1),
                    "text": text.strip()[:60],
                })
                verdict = "OK  " if rtf < 0.5 else ("TIGHT" if rtf < 1.0 else "TOO SLOW")
                print(f"{verdict} {model_id:<18} threads={threads:<3} "
                      f"audio={duration:5.1f}s decode={best:6.2f}s RTF={rtf:.2f} "
                      f"load={load_s:.0f}s", flush=True)
            del model

    out = Path(__file__).with_name("cpu_bench.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}", flush=True)


main()
