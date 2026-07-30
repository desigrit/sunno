"""WASAPI loopback capture: transcribe whatever is being played, not just the microphone.

This is a separate module because it needs a different audio library. sounddevice/PortAudio
has no loopback flag (WasapiSettings exposes only exclusive, auto_convert and
explicit_sample_format), and "Stereo Mix" is not a substitute — it is a driver-specific input
that taps one particular device, so it captures silence whenever the default output is
anything else, which is exactly the case for a USB or Bluetooth headset.

pyaudiowpatch is a PyAudio fork that exposes WASAPI loopback endpoints as ordinary input
devices. It is ~0.09 MB and used only on this path; the microphone path is untouched.

The device that matters here is the one the user actually listens through: capturing its
loopback yields precisely the audio reaching their ears — a call, a video, a film — which is
the material a hard-of-hearing user most needs captioned.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Iterator

import numpy as np
import soxr

from .config import FRAME_SAMPLES, SAMPLE_RATE

_LOOPBACK_SUFFIX = " [Loopback]"


def _pyaudio():
    import pyaudiowpatch as pa

    return pa


def list_loopback_devices() -> list[dict]:
    """Output endpoints that can be captured, newest-style WASAPI only."""
    pa = _pyaudio()
    audio = pa.PyAudio()
    try:
        default_name = ""
        try:
            wasapi = audio.get_host_api_info_by_type(pa.paWASAPI)
            default_name = audio.get_device_info_by_index(
                wasapi["defaultOutputDevice"]
            )["name"]
        except Exception:
            pass

        devices = []
        for dev in audio.get_loopback_device_info_generator():
            # The suffix is an implementation detail of the loopback enumeration; the user
            # recognises the device by its own name.
            name = dev["name"]
            if name.endswith(_LOOPBACK_SUFFIX):
                name = name[: -len(_LOOPBACK_SUFFIX)]
            devices.append(
                {
                    "index": int(dev["index"]),
                    "name": name,
                    "channels": int(dev["maxInputChannels"]),
                    "default_samplerate": float(dev["defaultSampleRate"]),
                    "hostapi": "Windows WASAPI",
                    "loopback": True,
                    "is_default_output": name in default_name or default_name in name,
                }
            )
        return devices
    finally:
        audio.terminate()


def loopback_available() -> bool:
    try:
        _pyaudio()
        return True
    except Exception:
        return False


class LoopbackStream:
    """Yields mono float32 frames of exactly FRAME_SAMPLES at SAMPLE_RATE.

    Mirrors MicrophoneStream's contract so the pipeline is indifferent to which one it is
    given. Loopback endpoints run at the device's mix rate (44.1 or 48 kHz) and are always
    multi-channel, so audio is downmixed and resampled the same way microphone input is.
    """

    def __init__(self, device_index: int, max_queued_blocks: int = 128):
        self.device_index = device_index
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max_queued_blocks)
        self._audio = None
        self._stream = None
        self._name = f"device {device_index}"
        self.dropped_blocks = 0
        self.capture_rate: int = SAMPLE_RATE
        self.capture_channels: int = 1
        self._lock = threading.Lock()

    def __enter__(self) -> "LoopbackStream":
        pa = _pyaudio()
        self._audio = pa.PyAudio()
        info = self._audio.get_device_info_by_index(self.device_index)

        name = info["name"]
        if name.endswith(_LOOPBACK_SUFFIX):
            name = name[: -len(_LOOPBACK_SUFFIX)]
        self._name = name
        self.capture_rate = int(info["defaultSampleRate"])
        self.capture_channels = int(info["maxInputChannels"])

        def callback(in_data, frame_count, time_info, status):  # noqa: ANN001
            block = np.frombuffer(in_data, dtype=np.float32)
            try:
                self._queue.put_nowait(block)
            except queue.Full:
                # Dropping is correct under back-pressure: stale audio is worse than a gap.
                self.dropped_blocks += 1
            return (None, pa.paContinue)

        self._stream = self._audio.open(
            format=pa.paFloat32,
            channels=self.capture_channels,
            rate=self.capture_rate,
            input=True,
            input_device_index=self.device_index,
            # 0 lets WASAPI choose its own period, which is what it wants in shared mode.
            frames_per_buffer=0,
            stream_callback=callback,
        )
        self._stream.start_stream()
        return self

    def __exit__(self, *exc) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            if self._audio is not None:
                try:
                    self._audio.terminate()
                except Exception:
                    pass
                self._audio = None
        self._queue.put(None)

    @property
    def device_name(self) -> str:
        return f"{self._name} (system audio)"

    def frames(self, should_continue: Callable[[], bool] | None = None) -> Iterator[np.ndarray]:
        keep_going = should_continue or (lambda: True)
        pending = np.empty(0, dtype=np.float32)
        resampler = None
        if self.capture_rate != SAMPLE_RATE:
            resampler = soxr.ResampleStream(
                self.capture_rate, SAMPLE_RATE, 1, dtype="float32", quality="QQ"
            )

        while keep_going():
            try:
                block = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if block is None:
                break

            if self.capture_channels > 1:
                block = block.reshape(-1, self.capture_channels).mean(axis=1)
            if resampler is not None:
                block = resampler.resample_chunk(block)
            if block.size == 0:
                continue

            pending = np.concatenate((pending, block))
            while pending.size >= FRAME_SAMPLES:
                yield pending[:FRAME_SAMPLES]
                pending = pending[FRAME_SAMPLES:]
