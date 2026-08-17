"""How often a growing caption rewrites text the reader has already seen.

This exists because a number in ``asr_stream.py``'s docstring had nothing behind it. An
earlier version of that docstring said the transducer "rewrote nothing, 0 of 31", which
was true of one clean synthesized clip and not true of speech. Rather than replace one
unbacked figure with another, this reproduces it.

Churn is not the same thing as latency, and ``bench_stream_latency.py`` cannot see it: that
script times one decode of one clip. What matters to somebody reading captions is whether
words already on screen change underneath them, which only shows up when the engine is
driven the way the pipeline drives it, over the same growing buffer.

So this replays ``pipeline._maybe_partial``: the first partial once ``min_partial_ms`` of
audio exists, another every ``partial_interval_ms``, each one over the whole utterance so
far, conditioned through ``AudioConditioner`` exactly as the real path does. A refresh
counts as a rewrite when the new text is not an extension of the previous text.

Run:  .venv\\Scripts\\python.exe bench\\bench_stream_churn.py testdata
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.asr_stream import StreamingEngine  # noqa: E402
from server.config import Settings  # noqa: E402
from server.preprocess import AudioConditioner  # noqa: E402


def load(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "testdata")
    model = sys.argv[2] if len(sys.argv) > 2 else "stream-en"

    settings = Settings()
    settings.model_size = model
    engine = StreamingEngine(settings)
    engine.warmup()

    print(f"{model}, partials from {settings.min_partial_ms} ms every "
          f"{settings.partial_interval_ms} ms\n")
    print(f"{'clip':<28} {'refreshes':>10} {'rewrites':>10} {'rate':>7}")
    print("-" * 58)

    total_refreshes = total_rewrites = 0
    examples: list[tuple[str, str]] = []

    for path in sorted(folder.glob("*.wav")):
        audio, rate = load(path)
        conditioner = AudioConditioner(settings)
        start = int(rate * settings.min_partial_ms / 1000)
        step = int(rate * settings.partial_interval_ms / 1000)
        if len(audio) < start:
            continue

        refreshes = rewrites = 0
        previous: str | None = None
        for end in range(start, len(audio) + 1, step):
            text = engine.partial(conditioner(audio[:end])).text
            refreshes += 1
            if previous is not None and not text.startswith(previous):
                rewrites += 1
                if len(examples) < 3:
                    examples.append((previous[-56:], text[-56:]))
            previous = text

        total_refreshes += refreshes
        total_rewrites += rewrites
        rate_pct = rewrites / refreshes if refreshes else 0.0
        print(f"{path.name:<28} {refreshes:>10} {rewrites:>10} {rate_pct:>6.0%}")

    print("-" * 58)
    overall = total_rewrites / total_refreshes if total_refreshes else 0.0
    print(f"{'total':<28} {total_refreshes:>10} {total_rewrites:>10} {overall:>6.0%}")

    if examples:
        print("\nwhat a rewrite looks like:")
        for before, after in examples:
            print(f"  {before!r}\n  -> {after!r}")


if __name__ == "__main__":
    main()
