"""Validate the hand-rolled biquad against scipy before scipy is removed.

Includes the case the reviewer flagged: calling the SAME conditioner instance twice must
give identical output, because the old sosfilt call passed no `zi` and therefore reset
state on every utterance. A per-case fresh instance would never catch retained state.
"""

import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.signal import butter, sosfilt  # noqa: E402

from server.config import SAMPLE_RATE, Settings  # noqa: E402
from server.preprocess import (  # noqa: E402
    AudioConditioner,
    apply_biquad,
    butterworth_highpass_biquad,
)

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures += 1


# --- coefficients ----------------------------------------------------------
print("coefficients vs scipy")
for cutoff in (80.0, 40.0, 120.0, 300.0):
    sos = butter(2, cutoff / (SAMPLE_RATE / 2), btype="highpass", output="sos")[0]
    scipy_coeffs = (sos[0], sos[1], sos[2], sos[4], sos[5])
    mine = butterworth_highpass_biquad(cutoff, SAMPLE_RATE)
    diff = max(abs(a - b) for a, b in zip(scipy_coeffs, mine))
    check(f"cutoff {cutoff:>5.0f} Hz", diff < 1e-9, f"max coeff diff {diff:.2e}")

# --- filter output ---------------------------------------------------------
print("\nfilter output vs scipy.sosfilt")
rng = np.random.default_rng(1234)
sos80 = butter(2, 80.0 / (SAMPLE_RATE / 2), btype="highpass", output="sos")
coeffs80 = butterworth_highpass_biquad(80.0, SAMPLE_RATE)

cases: dict[str, np.ndarray] = {
    "impulse": np.concatenate([[1.0], np.zeros(4095)]).astype(np.float32),
    "white noise": rng.standard_normal(16000).astype(np.float32) * 0.1,
    "DC offset": np.full(8000, 0.5, dtype=np.float32),
    "low-freq sine 30Hz": (
        0.5 * np.sin(2 * np.pi * 30 * np.arange(16000) / SAMPLE_RATE)
    ).astype(np.float32),
    "speech-band sine 1kHz": (
        0.5 * np.sin(2 * np.pi * 1000 * np.arange(16000) / SAMPLE_RATE)
    ).astype(np.float32),
    "single sample": np.array([0.7], dtype=np.float32),
}

wav_path = Path(__file__).resolve().parents[1] / "testdata" / "verify.wav"
if wav_path.exists():
    with wave.open(str(wav_path), "rb") as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    cases["recorded speech WAV"] = raw.astype(np.float32) / 32768.0

# Always present, so the suite cannot silently shrink on a machine without testdata/.
# Deterministic and speech-shaped: a formant-like sum swept in amplitude, plus low-frequency
# rumble the 80 Hz high-pass is specifically there to remove.
_t = np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * _t)
_speechlike = (
    0.30 * np.sin(2 * np.pi * 210 * _t)      # pitch
    + 0.20 * np.sin(2 * np.pi * 700 * _t)    # F1
    + 0.12 * np.sin(2 * np.pi * 1220 * _t)   # F2
    + 0.06 * np.sin(2 * np.pi * 2600 * _t)   # F3
) * _envelope
_rumble = 0.25 * np.sin(2 * np.pi * 42 * _t) + 0.15 * np.sin(2 * np.pi * 18 * _t)
cases["synthetic speech + rumble"] = (
    _speechlike + _rumble + rng.standard_normal(_t.size) * 0.002
).astype(np.float32)

for name, data in cases.items():
    expected = sosfilt(sos80, data)
    actual = apply_biquad(data, coeffs80)
    diff = float(np.max(np.abs(expected - actual))) if data.size else 0.0
    check(name, diff < 1e-6, f"max diff {diff:.2e}")

# --- statefulness (the reviewer's case) ------------------------------------
print("\nstatelessness across calls (same instance)")
settings = Settings()
conditioner = AudioConditioner(settings)
sample = rng.standard_normal(8000).astype(np.float32) * 0.1
first = conditioner(sample.copy())
second = conditioner(sample.copy())
third = conditioner(sample.copy())
check(
    "same instance, 3 identical calls",
    np.array_equal(first, second) and np.array_equal(second, third),
    f"max drift {float(np.max(np.abs(first - third))):.2e}",
)

# A fresh instance must agree with a reused one.
fresh = AudioConditioner(settings)(sample.copy())
check("fresh instance == reused instance", np.array_equal(first, fresh))

# --- whole-conditioner equivalence vs the scipy version --------------------
print("\nfull AudioConditioner vs scipy implementation")


class ScipyConditioner:
    """The previous implementation, for a like-for-like comparison."""

    def __init__(self, s: Settings) -> None:
        self.settings = s
        self._sos = butter(2, s.highpass_hz / (SAMPLE_RATE / 2), btype="highpass", output="sos")

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        out = audio.astype(np.float32, copy=True)
        out -= float(out.mean())
        out = sosfilt(self._sos, out).astype(np.float32)
        target = self.settings.target_rms
        if target > 0:
            rms = float(np.sqrt(np.mean(np.square(out))))
            if rms > 1e-4:
                gain = min(target / rms, self.settings.max_gain)
                if gain > 1.0:
                    out *= gain
        np.clip(out, -1.0, 1.0, out=out)
        return out


old = ScipyConditioner(settings)
new = AudioConditioner(settings)
for name, data in cases.items():
    diff = float(np.max(np.abs(old(data.copy()) - new(data.copy())))) if data.size else 0.0
    check(f"conditioner: {name}", diff < 1e-6, f"max diff {diff:.2e}")

check("empty array", AudioConditioner(settings)(np.array([], dtype=np.float32)).size == 0)

print(f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILURE(S)'}")
print(f"({len(cases)} signals x 2 comparisons + coefficient and state checks)")
sys.exit(1 if failures else 0)
