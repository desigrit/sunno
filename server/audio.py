"""Microphone capture via sounddevice/PortAudio (WASAPI on Windows)."""

from __future__ import annotations

import queue
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import sounddevice as sd

from .config import FRAME_SAMPLES, SAMPLE_RATE

_COM_STATE = threading.local()


def list_input_devices() -> list[dict]:
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append(
                {
                    "index": idx,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "default_samplerate": dev["default_samplerate"],
                    "hostapi": sd.query_hostapis(dev["hostapi"])["name"],
                }
            )
    return devices


def print_input_devices() -> None:
    default_in = sd.default.device[0]
    print("Available input devices:\n")
    for dev in list_input_devices():
        marker = "*" if dev["index"] == default_in else " "
        print(
            f" {marker} [{dev['index']:>2}] {dev['name']}  "
            f"({dev['hostapi']}, {dev['channels']}ch)"
        )
    print("\n  * = system default. Pass --device <index> to override.")


def _ensure_com_initialized() -> None:
    """Initialise COM on the calling thread (Windows only).

    PortAudio's WASAPI backend is COM-based. Pa_Initialize() sets up COM on whichever
    thread first touches sounddevice, but opening a stream from a *different* thread —
    which is what happens here, since capture runs in a worker thread — fails with
    AUDCLNT_E_UNSUPPORTED_FORMAT unless that thread has its own COM apartment.
    """
    if sys.platform != "win32":
        return
    if getattr(_COM_STATE, "initialized", False):
        return
    import ctypes

    COINIT_APARTMENTTHREADED = 0x2
    RPC_E_CHANGED_MODE = -2147417850  # 0x80010106: already in a different apartment
    try:
        hr = ctypes.windll.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    except Exception:
        return
    if hr < 0 and hr != RPC_E_CHANGED_MODE:
        print(f"[audio] CoInitializeEx returned 0x{hr & 0xFFFFFFFF:08X}", file=sys.stderr)
        return
    _COM_STATE.initialized = True


# Windows returns E_ACCESSDENIED (0x80070005) from IAudioClient::Initialize when the
# microphone privacy setting blocks the caller. PortAudio surfaces it inside a generic
# "Unanticipated host error" string, so match on the code and on the text WASAPI uses.
_ACCESS_DENIED_MARKERS = (
    "-2147024891",          # 0x80070005 as a signed 32-bit int
    "0x80070005",
    "access is denied",
    "accessdenied",
)


class MicrophoneOpenError(RuntimeError):
    """Raised when no candidate format could open the device.

    Distinguishes "Windows is blocking microphone access" from "the device is broken or
    busy", because the two need completely different advice and this app is useless to its
    user until the microphone works.
    """

    def __init__(self, device, failures: list[str]) -> None:
        self.device = device
        self.failures = failures
        blob = " ".join(failures).lower()
        self.access_denied = any(m in blob for m in _ACCESS_DENIED_MARKERS)
        # Deliberately NOT treated as a privacy denial: the device being held by another app
        # (Teams, Zoom, a game) is a routine daily occurrence, and sending the user to a
        # privacy toggle that is already switched on strands them.
        self.device_busy = "device_in_use" in blob

        if self.access_denied:
            message = (
                "Microphone access is blocked for this app. Open Settings > Privacy & "
                "security > Microphone and allow access."
            )
        elif self.device_busy:
            message = (
                f"The microphone is in use by another app. Close whatever is using it, "
                f"or choose a different microphone."
            )
        else:
            # Deliberately does not name the device.
            #
            # This message is printed to stdout, which the app captures into backend.log, and
            # the crash banner points users at that file — so anything here is something they
            # will be asked to send to a stranger. A capture device name like
            # "Headset (R-Phonak hearing aid)" says the person wears a hearing aid, which is
            # health information arriving through a field nobody thinks of as sensitive. The
            # UI already shows which device was chosen; the log does not need to.
            message = (
                "Could not open the selected input device. "
                "It may be unplugged or in use by another app."
            )
        super().__init__(message)

    def detail(self) -> str:
        return "\n  ".join(self.failures)


class MicrophoneStream:
    """Yields mono float32 frames of exactly FRAME_SAMPLES at SAMPLE_RATE.

    Many microphones (measurement mics, most USB interfaces) will not open at 16 kHz, so
    the device is opened at a rate it supports and resampled to 16 kHz with PyAV. The audio
    callback stays minimal; resampling happens on the consumer side.
    """

    def __init__(self, device: int | str | None = None, max_queued_blocks: int = 128):
        self.device = device
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max_queued_blocks)
        self._stream: sd.InputStream | None = None
        self.dropped_blocks = 0
        self.capture_rate: int = SAMPLE_RATE
        self.capture_channels: int = 1

    def _device_info(self) -> dict:
        try:
            return sd.query_devices(
                self.device if self.device is not None else sd.default.device[0]
            )
        except Exception:
            return {}

    def _candidate_formats(self) -> list[tuple[int, int]]:
        """(sample_rate, channels) pairs to try, best first.

        WASAPI shared mode only accepts the device's native mix format, so the device's
        advertised default rate and full channel count must be among the candidates -
        requesting mono from a stereo device raises AUDCLNT_E_UNSUPPORTED_FORMAT.
        """
        info = self._device_info()
        native_rate = int(round(float(info.get("default_samplerate") or SAMPLE_RATE)))
        native_ch = int(info.get("max_input_channels") or 1)

        pairs: list[tuple[int, int]] = [
            (SAMPLE_RATE, 1),           # ideal: no resampling needed
            (native_rate, native_ch),   # native mix format: most likely to be accepted
            (native_rate, 1),
        ]
        for rate in (48_000, 44_100, 32_000, SAMPLE_RATE):
            for ch in (native_ch, 1, 2):
                pairs.append((rate, ch))

        ordered: list[tuple[int, int]] = []
        for pair in pairs:
            if pair[1] >= 1 and pair not in ordered:
                ordered.append(pair)
        return ordered

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        # Copy: PortAudio reuses the buffer after the callback returns.
        if indata.ndim > 1 and indata.shape[1] > 1:
            mono = indata.mean(axis=1).astype(np.float32)
        else:
            mono = indata.reshape(-1).astype(np.float32).copy()
        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            # Never block the audio thread; dropping is preferable to glitching.
            self.dropped_blocks += 1

    def __enter__(self) -> "MicrophoneStream":
        # sd.check_input_settings() reports formats the driver will not actually start, so
        # probe by really opening and starting the stream. blocksize=0 lets PortAudio pick
        # the driver's native period (WASAPI rejects arbitrary block sizes); frames() below
        # reassembles whatever size arrives into fixed FRAME_SAMPLES frames.
        failures: list[str] = []
        _ensure_com_initialized()
        for rate, channels in self._candidate_formats():
            stream = None
            try:
                stream = sd.InputStream(
                    samplerate=rate,
                    blocksize=0,
                    device=self.device,
                    channels=channels,
                    dtype="float32",
                    callback=self._callback,
                )
                stream.start()
            except Exception as exc:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                failures.append(f"{rate} Hz / {channels}ch: {exc}")
                continue
            self.capture_rate = rate
            self.capture_channels = channels
            self._stream = stream
            return self

        raise MicrophoneOpenError(self.device, failures)

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._queue.put(None)

    def frames(self, should_continue: Callable[[], bool] | None = None) -> Iterator[np.ndarray]:
        resampler = None
        if self.capture_rate != SAMPLE_RATE:
            from .resample import StreamingAudioResampler

            resampler = StreamingAudioResampler(self.capture_rate, SAMPLE_RATE)

        pending = np.zeros(0, dtype=np.float32)
        while True:
            try:
                # Time out rather than block forever, so a stalled or unplugged device
                # can't wedge a pause request.
                block = self._queue.get(timeout=0.2)
            except queue.Empty:
                if should_continue is not None and not should_continue():
                    return
                continue
            if block is None:
                return
            if should_continue is not None and not should_continue():
                return

            if resampler is not None:
                block = resampler.resample(block)
                if block.size == 0:
                    continue
                block = block.reshape(-1)

            pending = np.concatenate([pending, block])
            n_full = len(pending) // FRAME_SAMPLES
            for i in range(n_full):
                yield pending[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            pending = pending[n_full * FRAME_SAMPLES :]

    @property
    def device_name(self) -> str:
        idx = self.device if self.device is not None else sd.default.device[0]
        try:
            name = sd.query_devices(idx)["name"].strip()
        except Exception:
            name = str(idx)
        detail = f"{self.capture_rate} Hz"
        if self.capture_channels > 1:
            detail += f", {self.capture_channels}ch->mono"
        if self.capture_rate != SAMPLE_RATE:
            detail += f" -> {SAMPLE_RATE} Hz"
        return f"{name} ({detail})"


class WavFileStream:
    """Replays a WAV file as if it were the microphone.

    Used to validate the pipeline deterministically and to benchmark recorded room audio
    without re-recording it. ``realtime=True`` paces frames at wall-clock speed so VAD
    endpointing behaves exactly as it does live.
    """

    def __init__(self, path: str | Path, realtime: bool = True):
        self.path = Path(path)
        self.realtime = realtime

    def __enter__(self) -> "WavFileStream":
        return self

    def __exit__(self, *exc) -> None:
        return None

    @property
    def device_name(self) -> str:
        return f"file: {self.path.name}"

    def _read_mono_16k(self) -> np.ndarray:
        with wave.open(str(self.path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            raw = wav.readframes(wav.getnframes())

        if width != 2:
            raise ValueError(f"{self.path.name}: expected 16-bit PCM, got {width * 8}-bit")

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        if rate != SAMPLE_RATE:
            # Linear resample is adequate here; real capture already runs at 16 kHz.
            target_len = int(round(len(audio) * SAMPLE_RATE / rate))
            audio = np.interp(
                np.linspace(0, len(audio), target_len, endpoint=False),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        return audio

    def frames(self, should_continue: Callable[[], bool] | None = None) -> Iterator[np.ndarray]:
        audio = self._read_mono_16k()
        pad = (-len(audio)) % FRAME_SAMPLES
        if pad:
            audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])

        started = time.monotonic()
        for i in range(0, len(audio), FRAME_SAMPLES):
            if should_continue is not None and not should_continue():
                return
            if self.realtime:
                due = started + (i / SAMPLE_RATE)
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            yield audio[i : i + FRAME_SAMPLES]

        # Trailing silence so the final utterance reaches its end-of-speech timeout.
        silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        for _ in range(40):
            if should_continue is not None and not should_continue():
                return
            if self.realtime:
                time.sleep(FRAME_SAMPLES / SAMPLE_RATE)
            yield silence
