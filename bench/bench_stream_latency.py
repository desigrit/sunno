"""Decode latency for the streaming models, measured the way the shipped tables are.

``bench/bench_latency.py`` times a whole utterance decode at beam 1 and files one figure
per model. The model picker compares those figures across models, so a new engine has to
be measured the same way or its numbers are not comparable and the picker will prefer
whichever engine happened to be measured most generously.

Deliberately NOT measuring streaming emission lag here. That number is much lower and it
is a different quantity: it describes text appearing while someone is still speaking,
which the current pipeline cannot deliver because it waits for an endpoint first. Putting
it in the same table would make these models look better than they will behave.
"""

from __future__ import annotations

import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

DURATIONS = (2.0, 4.0, 8.0)   # same as bench/bench_latency.py


def build(model_dir: Path, threads: int):
    def pick(*patterns):
        for pattern in patterns:
            hits = sorted(model_dir.glob(pattern))
            if hits:
                return str(hits[0])
        return None

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(model_dir / "tokens.txt"),
        encoder=pick("encoder*int8*.onnx", "encoder*.onnx"),
        decoder=pick("decoder*int8*.onnx", "decoder*.onnx"),
        joiner=pick("joiner*int8*.onnx", "joiner*.onnx"),
        num_threads=threads, provider="cpu", decoding_method="greedy_search",
    )


def decode_once(recognizer, audio: np.ndarray, rate: int) -> float:
    """One whole-utterance decode, the way the pipeline calls an engine today."""
    start = time.perf_counter()
    stream = recognizer.create_stream()
    stream.accept_waveform(rate, audio)
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    recognizer.get_result(stream)
    return (time.perf_counter() - start) * 1000.0


def main() -> None:
    root = Path(sys.argv[1])
    wav = sys.argv[2]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    with wave.open(wav, "rb") as w:
        rate = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    print(f"{threads} threads, best of 3, durations {DURATIONS}\n")
    print(f"{'model':<24} " + " ".join(f"{d:>6.0f}s" for d in DURATIONS) + f" {'mean':>8}")
    print("-" * 60)

    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        # A directory without an encoder is not a model, it is the punctuation model or a
        # stray folder. Skipped quietly rather than reported as a failed benchmark.
        if not any(model_dir.glob("encoder*.onnx")):
            continue
        try:
            recognizer = build(model_dir, threads)
            decode_once(recognizer, audio[: rate * 2], rate)   # warm up

            per_duration = []
            for seconds in DURATIONS:
                clip = audio[: int(rate * seconds)]
                best = min(decode_once(recognizer, clip, rate) for _ in range(3))
                per_duration.append(best)

            mean = statistics.mean(per_duration)
            print(f"{model_dir.name:<24} " +
                  " ".join(f"{v:>6.0f}ms" for v in per_duration) +
                  f" {mean:>6.0f}ms")
        except Exception as e:
            print(f"{model_dir.name:<24} FAILED {type(e).__name__}: {str(e)[:40]}")


if __name__ == "__main__":
    main()
