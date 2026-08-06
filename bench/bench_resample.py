"""Does replacing soxr cost accuracy? And is loopback's QQ setting hurting us?

Two questions, one harness:

1. soxr is LGPL and statically linked, which is the one licensing snag before a Store
   submission. scipy's resample_poly is BSD. Is it as good?
2. The microphone path resamples at soxr "HQ" but the loopback path at "QQ" — the lowest
   setting. If that costs accuracy, system-audio captions have been quietly worse than
   microphone ones.

Method: take real 16 kHz speech, upsample to the rates devices actually run at (44.1/48 kHz)
with the highest-quality setting available, then bring each back down to 16 kHz by every
candidate. The upsample is common to all, so it can't favour one. Score by SNR against the
original, and — the number that actually matters — by whether Whisper produces the same
words.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import soxr
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.resample import StreamingAudioResampler  # noqa: E402
TARGET = 16_000
DEFAULT_CLIP = ROOT / "testdata" / "1-two-speakers-en.wav"


def load16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr, nch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    assert sr == TARGET, sr
    return a


class ScipyStream:
    """Streaming polyphase resampler, the shape the server needs.

    resample_poly is a batch call: applied per chunk it restarts the filter each time and
    leaves a discontinuity at every boundary. Carrying the filter tail across chunks is what
    makes it equivalent to a continuous stream, and is the part a naive swap would get wrong.
    """

    def __init__(self, src: int, dst: int, taps_per_phase: int = 32):
        g = np.gcd(src, dst)
        self.up, self.down = dst // g, src // g
        # Anti-alias at the lower of the two Nyquist limits, expressed against the upsampled
        # rate; the 1.1 factor keeps a little margin below the true cutoff.
        self.ntaps = taps_per_phase * max(self.up, self.down) | 1
        self.h = signal.firwin(
            self.ntaps, 1.0 / max(self.up, self.down), window=("kaiser", 5.0)
        ) * self.up
        self.tail = np.zeros(self.ntaps - 1, dtype=np.float32)

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        x = np.concatenate((self.tail, chunk))
        self.tail = x[-(self.ntaps - 1):].copy()
        y = signal.upfirdn(self.h, x, self.up, self.down)
        # Drop the samples that depend on the padding carried in from last time.
        skip = ((self.ntaps - 1) * self.up) // (2 * self.down)
        take = int(len(chunk) * self.up / self.down)
        return y[skip:skip + take].astype(np.float32)


def soxr_stream(src: int, quality: str):
    st = soxr.ResampleStream(src, TARGET, 1, dtype="float32", quality=quality)
    return lambda c: st.resample_chunk(c)


def snr(ref: np.ndarray, test: np.ndarray) -> float:
    n = min(len(ref), len(test))
    r, t = ref[:n], test[:n]
    noise = r - t
    p = float(np.sum(r ** 2))
    q = float(np.sum(noise ** 2))
    return 10 * np.log10(p / q) if q > 0 else float("inf")


def run_stream(fn, audio: np.ndarray, chunk: int = 1024) -> np.ndarray:
    out = []
    for i in range(0, len(audio), chunk):
        out.append(fn(audio[i:i + chunk]))
    flush = getattr(fn, "flush", None)
    if flush is not None:
        out.append(flush())
    return np.concatenate([o for o in out if len(o)])


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
parser.add_argument(
    "--quality-only",
    action="store_true",
    help="measure signal quality without loading Whisper or requiring CUDA",
)
args = parser.parse_args()

original = load16k(args.clip)
print(f"source: {args.clip.name}  {len(original) / TARGET:.1f}s\n")

model = None
if not args.quality_only:
    from server import cuda_setup  # noqa: E402

    cuda_setup.register_cuda_dlls(required=True)
    from faster_whisper import WhisperModel  # noqa: E402

    print("loading Whisper large-v3 ...", flush=True)
    model = WhisperModel(
        "Systran/faster-whisper-large-v3",
        device="cuda",
        compute_type="float16",
        local_files_only=True,
    )

    def transcribe(a: np.ndarray) -> str:
        segs, _ = model.transcribe(a, beam_size=5, language="en")
        return " ".join(s.text.strip() for s in segs)

    reference_text = transcribe(original)
    print(f"reference transcript ({len(reference_text)} chars)\n")

results = []
for device_rate in (44_100, 48_000):
    # Common source material for every candidate, at the highest quality soxr offers.
    upsampled = soxr.resample(original, TARGET, device_rate, quality="VHQ").astype(np.float32)

    candidates = {
        "soxr HQ  (mic path today)": soxr_stream(device_rate, "HQ"),
        "soxr QQ  (loopback today)": soxr_stream(device_rate, "QQ"),
        "soxr VHQ": soxr_stream(device_rate, "VHQ"),
        "PyAV AudioResampler": StreamingAudioResampler(device_rate, TARGET),
        "scipy polyphase (BSD)": ScipyStream(device_rate, TARGET),
    }

    print(f"=== {device_rate} Hz -> {TARGET} Hz ===")
    for name, fn in candidates.items():
        out = run_stream(fn, upsampled)
        s = snr(original, out)
        if model is None:
            same = "not measured"
        else:
            text = transcribe(out)
            same = "IDENTICAL" if text.strip() == reference_text.strip() else "differs"
        results.append((device_rate, name, s, same))
        print(f"  {name:<28} SNR {s:6.1f} dB   transcript {same}")
    print()

print("=== summary ===")
for rate, name, s, same in results:
    print(f"  {rate}  {name:<28} {s:6.1f} dB  {same}")
