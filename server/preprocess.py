"""Conservative audio conditioning applied to each utterance before ASR.

Deliberately NOT noise suppression. Neural denoisers measurably lower transcription
accuracy and disproportionately clip accented speakers and the first words of short
utterances, which is precisely the workload here. These two steps are safe:

  * a high-pass filter, which removes HVAC rumble, footfall and handling noise that sit
    below the speech band and otherwise eat headroom;
  * gentle level normalisation, which helps quiet far-field audio without touching the
    spectral content the model relies on.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from .config import SAMPLE_RATE, Settings


class AudioConditioner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._sos = None
        if settings.highpass_hz > 0:
            nyquist = SAMPLE_RATE / 2
            self._sos = butter(
                2, settings.highpass_hz / nyquist, btype="highpass", output="sos"
            )

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio

        out = audio.astype(np.float32, copy=True)
        out -= float(out.mean())  # remove DC offset

        if self._sos is not None:
            out = sosfilt(self._sos, out).astype(np.float32)

        target = self.settings.target_rms
        if target > 0:
            rms = float(np.sqrt(np.mean(np.square(out))))
            # Skip near-silence: amplifying it just raises the noise floor.
            if rms > 1e-4:
                gain = min(target / rms, self.settings.max_gain)
                if gain > 1.0:
                    out *= gain

        np.clip(out, -1.0, 1.0, out=out)
        return out
