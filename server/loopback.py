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
import time
from typing import Callable, Iterator

import numpy as np
import soxr

from .config import FRAME_SAMPLES, SAMPLE_RATE

# How long to wait on the queue before deciding the endpoint is merely idle. Short enough
# that the silence generator tracks wall clock closely, long enough not to spin a core.
_IDLE_POLL_S = 0.1
_FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
# One scheduling hiccup should not dump a burst of silence into the VAD in a single go.
_MAX_CATCHUP_FRAMES = 16
# Shared read-only buffer; the pipeline never mutates the frames it is handed.
_SILENCE = np.zeros(FRAME_SAMPLES, dtype=np.float32)

_LOOPBACK_SUFFIX = " [Loopback]"


def _strip_loopback_suffix(name: str) -> str:
    """The endpoint's own name, without the tag the loopback enumeration appends.

    Shared rather than inlined because the default-output comparison depends on both sides
    having been through exactly the same treatment. The render endpoint that Windows names
    as the default carries no suffix, and the capture-side twin of the same device does, so
    comparing them raw never matches.
    """
    if name.endswith(_LOOPBACK_SUFFIX):
        return name[: -len(_LOOPBACK_SUFFIX)]
    return name


def _pyaudio():
    import pyaudiowpatch as pa

    return pa


def list_loopback_devices() -> list[dict]:
    """Output endpoints that can be captured, newest-style WASAPI only."""
    pa = _pyaudio()
    audio = pa.PyAudio()
    try:
        # None, not "", so "we could not find out" stays distinguishable from "it is called
        # nothing". An empty string here used to be compared with `in`, and every name
        # contains the empty string, so a machine that failed this lookup marked its entire
        # output list as the default.
        default_name: str | None = None
        try:
            wasapi = audio.get_host_api_info_by_type(pa.paWASAPI)
            default_name = audio.get_device_info_by_index(
                wasapi["defaultOutputDevice"]
            )["name"]
        except Exception:
            pass
        if default_name is not None:
            default_name = _strip_loopback_suffix(default_name).strip()

        devices = []
        for dev in audio.get_loopback_device_info_generator():
            # The suffix is an implementation detail of the loopback enumeration; the user
            # recognises the device by its own name.
            name = _strip_loopback_suffix(dev["name"])
            devices.append(
                {
                    "index": int(dev["index"]),
                    "name": name,
                    "channels": int(dev["maxInputChannels"]),
                    "default_samplerate": float(dev["defaultSampleRate"]),
                    "hostapi": "Windows WASAPI",
                    "loopback": True,
                    # Exact match, on names both put through the same stripping.
                    #
                    # This was a two-way substring test, which is wrong in both directions:
                    # every string contains "", so a failed default lookup marked the whole
                    # list as default, and any pair sharing a fragment matched each other —
                    # "Realtek" against "Speakers (Realtek Audio)" is a true that means
                    # nothing. Marking the wrong endpoint as the default is not cosmetic
                    # once the UI starts selecting by it: it captures a device the user is
                    # not listening to, and silently captioning the wrong thing is the
                    # failure this app can least afford.
                    "is_default_output": default_name is not None
                    and name.strip() == default_name,
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

    @property
    def is_alive(self) -> bool:
        """Whether the endpoint is still there.

        An output device can disappear underneath us — Bluetooth headphones leaving range,
        a monitor being switched off — and PortAudio reports that by the stream ceasing to
        be active rather than by raising. Distinguishing that from an idle-but-healthy
        endpoint is the whole reason the caller can trust a silent loopback.
        """
        with self._lock:
            stream = self._stream
        if stream is None:
            return False
        try:
            return bool(stream.is_active())
        except Exception:
            # A dead stream typically throws rather than returning False.
            return False

    def frames(self, should_continue: Callable[[], bool] | None = None) -> Iterator[np.ndarray]:
        keep_going = should_continue or (lambda: True)
        pending = np.empty(0, dtype=np.float32)
        last_yield = time.monotonic()
        resampler = None
        if self.capture_rate != SAMPLE_RATE:
            # HQ, matching the microphone path. This was "QQ" — soxr's lowest setting — which
            # measured 73.9 dB against HQ's 81.4 dB resampling 44.1 kHz to 16 kHz. The
            # transcript happened to come out identical on the clip tested, but there was no
            # reason for system audio to be fed a worse signal than the microphone, and the
            # cost of the better filter is not measurable next to a Whisper decode.
            resampler = soxr.ResampleStream(
                self.capture_rate, SAMPLE_RATE, 1, dtype="float32", quality="HQ"
            )

        while keep_going():
            try:
                block = self._queue.get(timeout=_IDLE_POLL_S)
            except queue.Empty:
                # WASAPI delivers no callbacks at all from an output endpoint while nothing is
                # playing, so a quiet desktop produces an empty queue indefinitely. Yield real
                # silence instead of spinning: silence is the truthful description of an idle
                # output, and it keeps the pipeline's level reporting alive so the UI can tell
                # "nothing is playing" apart from "capture has died".
                #
                # This is what makes the stall warning trustworthy on loopback. If the endpoint
                # actually disappears — a Bluetooth headset walking out of range — the stream
                # stops being active, this loop exits, levels stop, and the UI surfaces it.
                # Without the distinction a vanished device looked exactly like a quiet one,
                # and the app sat showing a running clock and no captions.
                if not self.is_alive:
                    break

                # Enough frames to cover the wall-clock gap, not one per poll. The pipeline
                # measures silence by counting frames, so under-producing would stretch
                # end-of-utterance detection by the same factor — a 520 ms hangover would take
                # eight seconds, and the last thing said before a pause would hang unfinalised.
                now = time.monotonic()
                owed = int((now - last_yield) / _FRAME_SECONDS)
                if owed <= 0:
                    continue
                # Cap the catch-up so a scheduling hiccup can't dump a burst of silence into
                # the VAD in one go.
                for _ in range(min(owed, _MAX_CATCHUP_FRAMES)):
                    yield _SILENCE
                last_yield = now
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
                # Real audio resets the clock too, so the silence generator only ever fills
                # gaps rather than double-counting time already covered by captured frames.
                last_yield = time.monotonic()
