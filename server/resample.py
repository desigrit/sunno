"""Streaming resampling for live capture, without depending on soxr.

Why this exists
---------------
Microphones rarely run at 16 kHz. The device is opened at a rate it supports and the frames
are resampled on the way to the engine, continuously, so the resampler has to carry state
across chunk boundaries - restarting the filter every block would put a discontinuity at
each seam and the artefacts land squarely in the speech band.

``soxr`` does that well and is what the x64 build uses. It publishes no ``win_arm64`` wheel,
so on a native ARM64 build it is simply absent, and without a replacement a Snapdragon could
only caption from a microphone that already happened to run at 16 kHz.

PyAV wraps FFmpeg's swresample, which is also stateful across calls and which the app
already ships for recording, so ARM gains no new native dependency by using it.

Quality is not assumed, and the default is not good enough. swresample's default polyphase
filter is 32 taps; measured on the project's two-speaker corpus, downsampling 48 kHz to
16 kHz, that reaches 58.0 dB against soxr HQ's 74.6 dB. Pinned at 256 taps it reaches
78.8 dB, and the two filters then agree to 0.00 dB across 100-7500 Hz.

What that harness does **not** measure is alias rejection, and the difference there is real:
the corpus is 16 kHz, so upsampling it to make test input leaves nothing above 8 kHz for a
stopband to reject. On a genuine out-of-band tone soxr is the better filter - a 9 kHz tone,
which aliases to 7 kHz, comes back at -146.5 dB through soxr and -110.0 dB through this. So
this is not "better than soxr"; it is close enough in the passband and far enough below a
16-bit least-significant bit out of it to be inaudible either way.

``tests/test_resample.py`` measures both on x64, where both libraries are installable, so
the substitution is checked rather than trusted: the loopback path once shipped soxr at
``"QQ"``, its lowest setting, and nothing caught it until someone read the argument.

Selection is by availability, not by architecture. An x64 machine keeps soxr; ARM falls
through to PyAV. Testing the import rather than the platform means a future soxr ARM64 wheel
starts being used with no code change, and means the fallback is reachable - and therefore
testable - on the developer's own machine.
"""

from __future__ import annotations

import numpy as np


class SoxrResampler:
    """soxr's stateful stream resampler, the x64 path."""

    def __init__(self, source_rate: int, target_rate: int) -> None:
        import soxr

        self._stream = soxr.ResampleStream(
            source_rate, target_rate, 1, dtype="float32", quality="HQ"
        )

    def resample_chunk(self, block: np.ndarray) -> np.ndarray:
        return self._stream.resample_chunk(block)

    def flush(self) -> np.ndarray:
        """Whatever the filter is still holding, at end of stream.

        Present so both resamplers expose the same interface. They are chosen by which
        library is installed, so an interface that differed by architecture would mean a
        caller could work on ARM and raise AttributeError on x64 - the one class of bug this
        module's "select by availability, not by platform" rule exists to keep testable here.
        """
        return self._stream.resample_chunk(np.empty(0, dtype=np.float32), last=True)


class PyAvResampler:
    """FFmpeg's swresample through PyAV, for machines with no soxr wheel.

    Built on the ``aresample`` filter rather than ``av.AudioResampler`` because the latter
    exposes no quality settings and swresample's default ``filter_size`` is 32 taps. That
    default is not adequate here: measured on the project's two-speaker corpus, downsampling
    48 kHz to 16 kHz, it reaches 58.0 dB against soxr HQ's 74.6 dB. A 16 dB deficit on the
    signal feeding a speech recogniser is not a detail, and it is invisible on x64 because
    x64 keeps soxr.

    At ``filter_size=256`` the same measurement gives **78.8 dB**, and the two filters agree
    to 0.00 dB from 100-7500 Hz. That is not the same as being better than soxr: the harness
    upsamples a 16 kHz corpus, so it has no out-of-band energy and cannot measure alias
    rejection, where soxr genuinely wins (a 9 kHz tone aliasing to 7 kHz returns at
    -146.5 dB through soxr against -110.0 dB here). Both are far below a 16-bit
    least-significant bit, so the substitution is inaudible; the claim is "close enough",
    not "better". ``tests/test_resample.py`` re-measures and fails if this regresses - the
    loopback path once shipped soxr at its lowest setting for months, and the only reason
    anyone noticed was reading the argument.

    The filter graph is also stateful across pushes, which is the property the streaming
    caller depends on: restarting per chunk would put a discontinuity at every seam.
    """

    # Tap count for the polyphase filter. See the class docstring for the measurements; this
    # is the single number that separates a usable ARM resampler from a poor one.
    FILTER_SIZE = 256

    def __init__(self, source_rate: int, target_rate: int) -> None:
        import av
        import av.filter

        self._av = av
        self.source_rate = source_rate
        graph = av.filter.Graph()
        self._source = graph.add_abuffer(
            sample_rate=source_rate, format="fltp", layout="mono"
        )
        self._filter = graph.add(
            "aresample", f"osr={target_rate}:filter_size={self.FILTER_SIZE}"
        )
        self._sink = graph.add("abuffersink")
        self._source.link_to(self._filter)
        self._filter.link_to(self._sink)
        graph.configure()
        self._graph = graph

    def _drain(self) -> np.ndarray:
        out = []
        while True:
            try:
                out.append(self._sink.pull().to_ndarray().reshape(-1))
            except Exception:
                # Raised for both "nothing buffered yet" and end of stream. Neither is an
                # error: the filter emits when it has enough input, so an empty return early
                # in a stream is normal.
                break
        if not out:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(out).astype(np.float32, copy=False)

    def resample_chunk(self, block: np.ndarray) -> np.ndarray:
        block = np.ascontiguousarray(block, dtype=np.float32)
        if block.size == 0:
            return np.empty(0, dtype=np.float32)
        frame = self._av.AudioFrame.from_ndarray(
            block.reshape(1, -1), format="fltp", layout="mono"
        )
        frame.sample_rate = self.source_rate
        frame.pts = None
        self._source.push(frame)
        return self._drain()

    def flush(self) -> np.ndarray:
        """Whatever the filter is still holding. Safe to call once, at end of stream."""
        try:
            self._source.push(None)
        except Exception:
            return np.empty(0, dtype=np.float32)
        return self._drain()


def soxr_available() -> bool:
    """Whether the preferred resampler is installed, without importing it."""
    import importlib.util

    return importlib.util.find_spec("soxr") is not None


def make_resampler(source_rate: int, target_rate: int):
    """The best stateful resampler this machine has.

    Raises nothing architecture-specific: if neither library is present that is a broken
    install, and the ImportError naming the missing package is more useful than anything
    this could invent.
    """
    if soxr_available():
        return SoxrResampler(source_rate, target_rate)
    return PyAvResampler(source_rate, target_rate)
