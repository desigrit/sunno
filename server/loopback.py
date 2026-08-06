"""WASAPI loopback capture: transcribe whatever is being played, not just the microphone.

SoundCard reaches WASAPI through CFFI rather than a CPython audio extension, so the same code
runs natively on x64 and ARM64. sounddevice/PortAudio has no loopback flag, and "Stereo Mix" is
not a substitute: it is driver-specific and often captures silence from USB or Bluetooth output.
"""

from __future__ import annotations

import warnings
from typing import Callable, Iterator

import numpy as np

from .config import FRAME_SAMPLES, SAMPLE_RATE

def _soundcard():
    # SoundCard's WASAPI calls are COM-based, and COM apartments are per thread. Device
    # enumeration runs on an HTTP worker while capture runs on the pump thread, so both must
    # initialize independently before importing or calling SoundCard.
    from .audio import _ensure_com_initialized

    _ensure_com_initialized()
    import soundcard

    return soundcard


def _loopback_devices():
    return [device for device in _soundcard().all_microphones(include_loopback=True)
            if device.isloopback]


def list_loopback_devices() -> list[dict]:
    """Output endpoints that can be captured, newest-style WASAPI only."""
    soundcard = _soundcard()
    default_id = soundcard.default_speaker().id
    return [
        {
            # SoundCard identifies devices by WASAPI id. The UI protocol currently carries an
            # integer, so this is the enumeration slot; the UI persists and validates the name
            # because audio-library indices have never been stable across device changes.
            "index": index,
            "id": device.id,
            "name": device.name,
            "channels": device.channels,
            "default_samplerate": SAMPLE_RATE,
            "hostapi": "Windows WASAPI",
            "loopback": True,
            "is_default_output": device.id == default_id,
        }
        for index, device in enumerate(_loopback_devices())
    ]


def loopback_available() -> bool:
    try:
        _soundcard()
        return True
    except Exception:
        return False


class LoopbackStream:
    """Yields mono float32 frames of exactly FRAME_SAMPLES at SAMPLE_RATE.

    Mirrors MicrophoneStream's contract so the pipeline is indifferent to which one it is
    given. Loopback endpoints run at the device's mix rate (44.1 or 48 kHz) and are always
    multi-channel, so audio is downmixed and resampled the same way microphone input is.
    """

    def __init__(self, device_index: int | str, max_queued_blocks: int = 128):
        self.device_index = device_index
        self._recorder_context = None
        self._recorder = None
        self._device_id = None
        self._name = f"device {device_index}"
        self.dropped_blocks = 0
        self.capture_rate: int = SAMPLE_RATE
        self.capture_channels: int = 1

    def __enter__(self) -> "LoopbackStream":
        devices = _loopback_devices()
        if isinstance(self.device_index, str) and not self.device_index.isdigit():
            device = next(
                (candidate for candidate in devices if candidate.id == self.device_index),
                None,
            )
        else:
            index = int(self.device_index)
            device = devices[index] if 0 <= index < len(devices) else None
        if device is None:
            raise IndexError("the selected system-audio device is no longer available")
        self._name = device.name
        self._device_id = device.id
        self.capture_channels = device.channels
        self._recorder_context = device.recorder(
            samplerate=SAMPLE_RATE,
            channels=self.capture_channels,
            blocksize=FRAME_SAMPLES,
        )
        self._recorder = self._recorder_context.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        context = self._recorder_context
        self._recorder = None
        self._recorder_context = None
        if context is not None:
            context.__exit__(*exc)

    @property
    def device_name(self) -> str:
        return f"{self._name} (system audio)"

    @property
    def is_alive(self) -> bool:
        """Whether the endpoint is still there.

        An output device can disappear underneath us — Bluetooth headphones leaving range,
        a monitor being switched off — and PortAudio reports that by the stream ceasing to
        be active rather than by raising. Distinguishing that from an idle-but-healthy
        endpoint is the whole reason the caller can trust a silent loopback.
        """
        if self._recorder is None or self._device_id is None:
            return False
        try:
            return any(device.id == self._device_id for device in _loopback_devices())
        except Exception:
            return False

    def frames(self, should_continue: Callable[[], bool] | None = None) -> Iterator[np.ndarray]:
        keep_going = should_continue or (lambda: True)
        while keep_going():
            if self._recorder is None:
                return
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                block = self._recorder.record(numframes=FRAME_SAMPLES)
            self.dropped_blocks += sum(
                issubclass(warning.category, RuntimeWarning) for warning in caught
            )
            yield block.mean(axis=1).astype(np.float32, copy=False)
