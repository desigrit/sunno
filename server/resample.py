"""Stateful streaming resampling for live audio."""

from __future__ import annotations

import av
import numpy as np


class StreamingAudioResampler:
    """Convert mono float32 chunks without restarting the filter at chunk boundaries."""

    def __init__(self, source_rate: int, target_rate: int) -> None:
        self.source_rate = source_rate
        self._resampler = av.AudioResampler(
            format="fltp",
            layout="mono",
            rate=target_rate,
        )

    def resample(self, chunk: np.ndarray) -> np.ndarray:
        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(chunk, dtype=np.float32).reshape(1, -1),
            format="fltp",
            layout="mono",
        )
        frame.sample_rate = self.source_rate
        return self._join(self._resampler.resample(frame))

    __call__ = resample

    def flush(self) -> np.ndarray:
        return self._join(self._resampler.resample(None))

    @staticmethod
    def _join(frames) -> np.ndarray:
        if not frames:
            return np.empty(0, dtype=np.float32)
        return np.concatenate([frame.to_ndarray().reshape(-1) for frame in frames])
