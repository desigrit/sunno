"""Benchmark Whisper decode settings on real speech: compute type x beam size.

Turing has INT8 tensor cores, so int8_float16 may beat float16 on this GPU. Measures
latency and prints the text so quality regressions are visible, not just speed.
"""

import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import server  # noqa: E402  (registers CUDA DLLs)
from faster_whisper import WhisperModel  # noqa: E402


def load(path):
    with wave.open(str(path), "rb") as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)
    return audio


CLIPS = [
    load(ROOT / "testdata" / "1-two-speakers-en.wav")[: 16000 * 6],
    load(ROOT / "testdata" / "3-two-speakers-en.wav")[16000 * 26 : 16000 * 31],
]


def run(model, audio, beam, **kw):
    started = time.perf_counter()
    segments, _ = model.transcribe(
        audio, language="en", beam_size=beam, temperature=0.0,
        condition_on_previous_text=False, vad_filter=False, **kw
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return (time.perf_counter() - started) * 1000, text


for compute_type in ("float16", "int8_float16", "int8"):
    print(f"\n{'=' * 78}\ncompute_type = {compute_type}\n{'=' * 78}")
    t0 = time.perf_counter()
    model = WhisperModel("large-v3", device="cuda", compute_type=compute_type)
    load_ms = (time.perf_counter() - t0) * 1000
    run(model, np.zeros(16000, dtype=np.float32), 1)  # warm up

    print(f"  model load: {load_ms:.0f} ms")
    for beam in (1, 3, 5, 8):
        times, texts = [], []
        for clip in CLIPS:
            samples = [run(model, clip, beam) for _ in range(3)]
            times.append(min(t for t, _ in samples))
            texts.append(samples[0][1])
        print(f"  beam={beam}: {times[0]:6.1f} ms | {times[1]:6.1f} ms")
        for t in texts:
            print(f"          > {t[:100]}")
    del model
