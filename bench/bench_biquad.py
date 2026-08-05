"""Is the hand-rolled biquad fast enough to sit in the caption latency path?"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.signal import butter, sosfilt  # noqa: E402

from server.config import SAMPLE_RATE  # noqa: E402
from server.preprocess import apply_biquad, butterworth_highpass_biquad  # noqa: E402

sos = butter(2, 80 / (SAMPLE_RATE / 2), btype="highpass", output="sos")
coeffs = butterworth_highpass_biquad(80.0, SAMPLE_RATE)

print(f"{'utterance':<14}{'scipy':>12}{'python loop':>14}{'slowdown':>11}")
print("-" * 51)
for secs in (2, 5, 10, 20):
    x = (np.random.randn(SAMPLE_RATE * secs) * 0.1).astype(np.float32)

    t0 = time.perf_counter()
    for _ in range(3):
        sosfilt(sos, x)
    scipy_ms = (time.perf_counter() - t0) / 3 * 1000

    t0 = time.perf_counter()
    apply_biquad(x, coeffs)
    loop_ms = (time.perf_counter() - t0) * 1000

    print(f"{str(secs) + 's':<14}{scipy_ms:>10.2f}ms{loop_ms:>12.1f}ms{loop_ms / scipy_ms:>10.0f}x")

print("\nWhisper inference for reference: ~320-400 ms per utterance")
