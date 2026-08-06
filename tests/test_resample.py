"""PyAV must preserve the streaming quality that justified removing soxr."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.resample import StreamingAudioResampler  # noqa: E402


def run_stream(resampler: StreamingAudioResampler, audio: np.ndarray) -> np.ndarray:
    output = []
    for offset in range(0, len(audio), 1024):
        output.append(resampler.resample(audio[offset:offset + 1024]))
    output.append(resampler.flush())
    return np.concatenate([chunk for chunk in output if len(chunk)])


def snr(reference: np.ndarray, actual: np.ndarray) -> float:
    length = min(len(reference), len(actual))
    signal = reference[:length]
    noise = signal - actual[:length]
    return 10 * np.log10(float(np.sum(signal ** 2)) / float(np.sum(noise ** 2)))


rate = 48_000
seconds = 4
t = np.arange(16_000 * seconds, dtype=np.float32) / 16_000
# Speech-band signal with several non-bin-centred frequencies, so chunk boundaries and filter
# state both matter. Round-trip through PyAV at a real device rate.
source = (
    0.3 * np.sin(2 * np.pi * 173.3 * t)
    + 0.2 * np.sin(2 * np.pi * 911.7 * t)
    + 0.1 * np.sin(2 * np.pi * 2711.1 * t)
).astype(np.float32)
up = run_stream(StreamingAudioResampler(16_000, rate), source)
down = run_stream(StreamingAudioResampler(rate, 16_000), up)
quality = snr(source, down)

if quality < 70:
    print(f"FAIL  streaming PyAV quality  {quality:.1f} dB")
    raise SystemExit(1)

print(f"PASS  streaming PyAV quality  {quality:.1f} dB")
