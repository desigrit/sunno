"""Streaming wrapper around the Silero VAD v6 ONNX model bundled with faster-whisper.

faster_whisper.vad.SileroVADModel re-zeroes its LSTM state on every call, which makes it
a batch API. For live captioning we need frame-by-frame probabilities with state carried
across calls, so we drive the same ONNX graph directly.
"""

from __future__ import annotations

import os

import numpy as np
import onnxruntime

CONTEXT_SAMPLES = 64
_STATE_DIM = 128


class StreamingSileroVAD:
    """Emits a speech probability per fixed-size frame, preserving state between frames."""

    def __init__(self, frame_samples: int = 512) -> None:
        from faster_whisper.utils import get_assets_path

        model_path = os.path.join(get_assets_path(), "silero_vad_v6.onnx")

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4

        self._session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.frame_samples = frame_samples
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, _STATE_DIM), dtype=np.float32)
        self._c = np.zeros((1, 1, _STATE_DIM), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Return P(speech) for one frame of exactly ``frame_samples`` float32 samples."""
        if frame.shape[0] != self.frame_samples:
            raise ValueError(
                f"expected {self.frame_samples} samples, got {frame.shape[0]}"
            )

        batch = np.concatenate(
            [self._context, frame.reshape(1, -1).astype(np.float32)], axis=1
        )
        out, self._h, self._c = self._session.run(
            None, {"input": batch, "h": self._h, "c": self._c}
        )
        self._context = batch[:, -CONTEXT_SAMPLES:]
        # 'speech_probs' is 1-D: one probability per sequence element (here, one frame).
        return float(np.ravel(out)[0])
