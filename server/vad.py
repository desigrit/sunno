"""Streaming wrapper around the Silero VAD v6 ONNX model.

faster_whisper.vad.SileroVADModel re-zeroes its LSTM state on every call, which makes it
a batch API. For live captioning we need frame-by-frame probabilities with state carried
across calls, so we drive the same ONNX graph directly.

The graph itself runs on plain onnxruntime and has no CTranslate2 in it, but it used to be
located by importing faster_whisper - and importing anything under that package executes its
__init__, which imports ctranslate2. So a module that needs none of CTranslate2 could not start
without it. That matters on Windows on ARM, where no ctranslate2 wheel exists: the file is
sitting right there on disk and the only thing standing between us and it was an import.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnxruntime

CONTEXT_SAMPLES = 64
_STATE_DIM = 128

_ASSET_NAME = "silero_vad_v6.onnx"


def _asset_path() -> str:
    """Find the Silero graph without importing faster_whisper.

    A copy vendored beside this package is the real answer, and the others are fallbacks for a
    development checkout. That ordering matters more than it looks: faster-whisper depends on
    ctranslate2, which has no wheel on Windows on ARM and will not build, so `pip install
    faster-whisper` cannot succeed there at all and no faster_whisper/assets tree exists to find.
    Removing the import was only half the job - the file itself has to travel with us.

    Order-preserving rather than a set, because set iteration order varies per process (string
    hashing is randomised), and a search that silently picks a different file between runs is a
    bad way to load a model.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "assets", _ASSET_NAME),
        os.path.join(os.path.dirname(here), "assets", _ASSET_NAME),
    ]
    for prefix in dict.fromkeys([sys.prefix, getattr(sys, "base_prefix", sys.prefix)]):
        candidates.append(
            os.path.join(prefix, "Lib", "site-packages", "faster_whisper", "assets", _ASSET_NAME)
        )
    for path in candidates:
        if os.path.isfile(path):
            return path

    try:
        from faster_whisper.utils import get_assets_path

        path = os.path.join(get_assets_path(), _ASSET_NAME)
        if os.path.isfile(path):
            return path
    except Exception:
        pass

    raise FileNotFoundError(
        f"{_ASSET_NAME} not found. Looked beside the server package and in site-packages. "
        "Voice detection cannot start without it."
    )


class StreamingSileroVAD:
    """Emits a speech probability per fixed-size frame, preserving state between frames."""

    def __init__(self, frame_samples: int = 512) -> None:
        model_path = _asset_path()

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
