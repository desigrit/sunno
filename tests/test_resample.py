"""The PyAV resampler has to be as good as the soxr one it stands in for.

Why this is measured rather than assumed
----------------------------------------
``soxr`` publishes no ``win_arm64`` wheel, so a native ARM64 build resamples with FFmpeg's
swresample through PyAV instead. That substitution is invisible on x64 - where soxr is still
chosen - which is exactly the shape of change that ships broken to the platform nobody tests.

It is also a repeat of a bug this project already had: the loopback path once passed soxr
``quality="QQ"``, its lowest setting, measuring 73.9 dB against HQ's 81.4 dB. Nothing caught
it until someone read the argument. So taking a new library's defaults on trust is not good
enough, and this compares the two directly on x64 where both are installable.

The measure is signal-to-noise against an analytically-known signal: resample a sine, then
compare against the same sine generated at the target rate. Any filter worth using leaves
the tone intact and the error is whatever it did to the edges and the stopband.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tests._isolate  # noqa: F401,E402

from server.resample import (  # noqa: E402
    PyAvResampler,
    SoxrResampler,
    make_resampler,
    soxr_available,
)

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    if not ok:
        failures += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  -> ' + detail) if detail else ''}")


TARGET = 16000
TONE_HZ = 440.0
SECONDS = 2.0
CHUNK = 1024


def tone(rate: int, seconds: float = SECONDS) -> np.ndarray:
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    return (0.5 * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)


def stream_through(resampler, data: np.ndarray) -> np.ndarray:
    """Feed the signal in chunks, which is how the product uses it."""
    out = []
    for i in range(0, data.size, CHUNK):
        block = resampler.resample_chunk(data[i : i + CHUNK])
        if block.size:
            out.append(block)
    flush = getattr(resampler, "flush", None)
    if flush is not None:
        tail = flush()
        if tail.size:
            out.append(tail)
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def snr_db_against(got: np.ndarray, reference: np.ndarray) -> float:
    """SNR of ``got`` against an arbitrary reference signal at the target rate."""
    n = min(got.size, reference.size)
    margin = TARGET // 10
    if n <= 2 * margin:
        return float("-inf")
    a = got[margin : n - margin].astype(np.float64)
    b = reference[margin : n - margin].astype(np.float64)
    best = -np.inf
    for lag in range(-80, 81):
        err = np.roll(a, lag) - b
        power = float(np.sum(b * b))
        noise = float(np.sum(err * err))
        if noise <= 0:
            return float("inf")
        best = max(best, 10.0 * np.log10(power / noise))
    return best


def snr_db(got: np.ndarray) -> float:
    """SNR against an ideal tone at the target rate, ignoring filter edges."""
    return snr_db_against(got, tone(TARGET, SECONDS))


print(f"\nsoxr installed: {soxr_available()}\n")

print("-- both resamplers reconstruct a synthetic tone --")

# Reported, not gated. A pure tone measures passband fidelity, and soxr's steeper filter is
# also more conservative about where the passband ends, so the two are not comparable this
# way - the gap here says nothing about which is better on speech. The gate is the speech
# comparison below. Neither figure measures alias rejection, where soxr is genuinely the
# better filter; see server/resample.py.
results: dict[str, float] = {}
for rate in (44100, 48000):
    source = tone(rate)

    pyav = snr_db(stream_through(PyAvResampler(rate, TARGET), source))
    results[f"pyav-{rate}"] = pyav
    # 85 dB is already below one least-significant bit of a 16-bit sample, which is the
    # format the microphone path ultimately feeds.
    check(f"PyAV {rate} -> {TARGET} reconstructs the tone", pyav > 85.0, f"{pyav:.1f} dB")

    if soxr_available():
        sox = snr_db(stream_through(SoxrResampler(rate, TARGET), source))
        results[f"soxr-{rate}"] = sox
        check(f"soxr {rate} -> {TARGET} reconstructs the tone", sox > 85.0, f"{sox:.1f} dB")

print("\n-- and agree on real speech, which is the comparison that decides it --")

speech_path = Path(__file__).resolve().parents[1] / "testdata" / "3-two-speakers-en.wav"
if not speech_path.is_file() or not soxr_available():
    print(f"  --    skipped (speech corpus or soxr unavailable): {speech_path.name}")
else:
    import wave

    with wave.open(str(speech_path), "rb") as wf:
        native_rate = wf.getframerate()
        channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    speech = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        speech = speech.reshape(-1, channels).mean(axis=1)

    for device_rate in (44100, 48000):
        # Stand in for what a device delivers: take the corpus up to the capture rate with
        # one shared reference filter, then bring it back down with each candidate. Both
        # candidates inherit the same upsampling error, so the difference between them is
        # only what each did on the way down - which is the step the product performs.
        up = SoxrResampler(native_rate, device_rate)
        device_audio = stream_through(up, speech)

        ref = snr_db_against(stream_through(SoxrResampler(device_rate, TARGET), device_audio),
                             speech)
        got = snr_db_against(stream_through(PyAvResampler(device_rate, TARGET), device_audio),
                             speech)
        results[f"speech-soxr-{device_rate}"] = ref
        results[f"speech-pyav-{device_rate}"] = got

        check(
            f"PyAV holds up on speech from {device_rate}",
            got > 30.0,
            f"{got:.1f} dB",
        )
        # This is the "QQ" guard. On real speech the two filters were measured 0.2 dB apart;
        # 3 dB leaves room for run-to-run variation while still catching a resampler that is
        # meaningfully worse than the one it replaced.
        check(
            f"PyAV is within 3 dB of soxr on speech from {device_rate}",
            got > ref - 3.0,
            f"PyAV {got:.1f} dB vs soxr {ref:.1f} dB",
        )

print("\n-- streaming behaviour --")

# State across chunks is the whole reason these are stream objects. A resampler that
# restarted per block would put a discontinuity at every seam; feeding one long block and
# many short ones must agree.
rate = 48000
source = tone(rate)
chunked = stream_through(PyAvResampler(rate, TARGET), source)
whole = PyAvResampler(rate, TARGET)
single = whole.resample_chunk(source)
single = np.concatenate([single, whole.flush()]) if single.size else whole.flush()
n = min(chunked.size, single.size)
check(
    "PyAV carries filter state across chunk boundaries",
    n > 0 and float(np.max(np.abs(chunked[:n] - single[:n]))) < 1e-4,
    f"max diff {float(np.max(np.abs(chunked[:n] - single[:n]))) if n else float('nan'):.2e}",
)

check(
    "an empty block resamples to an empty block, not an error",
    PyAvResampler(48000, TARGET).resample_chunk(np.empty(0, dtype=np.float32)).size == 0,
)

out = PyAvResampler(48000, TARGET).resample_chunk(tone(48000, 0.1))
check("output is float32", out.dtype == np.float32, str(out.dtype))

print("\n-- selection --")

check(
    "both resamplers expose the same interface",
    hasattr(SoxrResampler(48000, TARGET), "flush")
    and hasattr(PyAvResampler(48000, TARGET), "flush"),
    "an interface that differs by architecture works on ARM and raises on x64",
)

chosen = make_resampler(48000, TARGET)
expected = SoxrResampler if soxr_available() else PyAvResampler
check(
    f"make_resampler picks {expected.__name__} on this machine",
    isinstance(chosen, expected),
    type(chosen).__name__,
)
check(
    "selection tests for the library, not the architecture",
    "soxr_available" in open(
        Path(__file__).resolve().parents[1] / "server" / "resample.py", encoding="utf-8"
    ).read(),
    "so a future soxr ARM64 wheel is used with no code change",
)

print("\nmeasured: " + ", ".join(f"{k} {v:.1f} dB" for k, v in sorted(results.items())))
print(f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILURE(S)'}")
sys.exit(1 if failures else 0)
