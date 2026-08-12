"""Microphone capture via sounddevice/PortAudio (WASAPI on Windows)."""

from __future__ import annotations

import queue
import sys
import time
import wave
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import sounddevice as sd

from .config import FRAME_SAMPLES, SAMPLE_RATE


# The one host API whose device list tracks whether the hardware is actually there.
#
# PortAudio's WASAPI backend enumerates with
#     IMMDeviceEnumerator::EnumAudioEndpoints(eAll, DEVICE_STATE_ACTIVE, ...)
# so its list is, by construction, the set Windows marks ACTIVE. The legacy backends
# (MME, DirectSound, WDM-KS) report every endpoint a driver declares, connected or not.
#
# Measured on one desktop: 22 input entries across the four APIs, of which WASAPI saw 4.
# The other 18 included two unplugged Realtek jacks, four cameras last connected months
# ago, and several virtual mixers. Cross-checked against DeviceState in
# HKLM\...\MMDevices\Audio\Capture, the WASAPI 4 were exactly the 4 marked ACTIVE. Hours
# later, with one device unplugged, both lists moved to the same 3 together.
_LIVE_HOST_API = "Windows WASAPI"


def list_input_devices() -> list[dict]:
    """Input devices worth showing, which is not the same as every device PortAudio sees.

    Narrows to the WASAPI enumeration when it returns anything, because that is the only
    host API that distinguishes a microphone which is plugged in from one the driver
    merely remembers. Falls back to the unfiltered list when WASAPI reports nothing, which
    covers non-Windows machines and any box where that backend failed to start: a picker
    holding stale entries is a poor experience, but an empty one is a broken app.

    Indices are PortAudio's own and are deliberately not renumbered. The app persists the
    chosen index and the backend opens the stream by it, so renumbering here would move
    every saved microphone by a silent, variable offset.
    """
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

    live = [d for d in devices if d["hostapi"] == _LIVE_HOST_API]
    if not live:
        _mark_default(devices)
        return devices

    hidden = len(devices) - len(live)
    if hidden:
        # Counts only, never names. This reaches backend.log, which users are asked to
        # send when something breaks, and a capture device name like
        # "Headset (R-Phonak hearing aid)" is health information. The numbers are here
        # because a device visible only through a legacy API would vanish from the picker
        # with no other trace, and that is the one way this filter can hurt someone.
        print(
            f"[audio] input devices: {len(live)} connected, "
            f"{hidden} hidden as not currently connected",
            flush=True,
        )
    _mark_default(live)
    return live


def _mark_default(devices: list[dict]) -> None:
    """Flag the one device that is in use when nobody has chosen.

    Computed here rather than in the /devices.json handler so that the refresh path gets it
    too — refresh runs `list_input_devices` in a child process, and a flag added downstream
    of that would be present at startup and missing after a refresh.

    Exactly one entry is marked, or none. The picker uses this to say which microphone is
    being captured on a first run, and two candidates would be worse than none.
    """
    default = _default_input_index(devices)
    for d in devices:
        d["is_default_input"] = d["index"] == default


def _default_input_index(devices: list[dict]) -> int:
    """Which of `devices` is the system default, or -1 when none of them is.

    The return is always either an index present in `devices` or -1. That matters more than
    it looks: the obvious implementation returns sd.default.device[0], which on Windows is
    an MME index — literally 1 on the machine this was written on — and so is never in a
    WASAPI-filtered list. A caller that pre-selects by this value would silently select
    nothing, and a listing would print with no default marked at all. Each host API also
    publishes its own default, so prefer that when the list has been narrowed to one API.

    Windows can legitimately have no default capture device. -1 says so rather than
    promoting an arbitrary device to "default", which would be a lie in a picker.
    """
    present = {d["index"] for d in devices}

    # A list that still holds legacy entries was never filtered, so PortAudio's own global
    # default is the right answer and is one of them.
    if any(d["hostapi"] != _LIVE_HOST_API for d in devices):
        idx = sd.default.device[0]
        return idx if idx in present else -1

    for api in sd.query_hostapis():
        if api["name"] == _LIVE_HOST_API:
            idx = api.get("default_input_device", -1)
            return idx if idx is not None and idx in present else -1
    return -1


def print_input_devices() -> None:
    devices = list_input_devices()
    default_in = _default_input_index(devices)
    print("Available input devices:\n")
    for dev in devices:
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
    import ctypes

    COINIT_APARTMENTTHREADED = 0x2
    RPC_E_CHANGED_MODE = -2147417850  # 0x80010106: already in a different apartment
    try:
        hr = ctypes.windll.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    except Exception:
        return
    if hr < 0 and hr != RPC_E_CHANGED_MODE:
        print(f"[audio] CoInitializeEx returned 0x{hr & 0xFFFFFFFF:08X}", file=sys.stderr)


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


def _default_device() -> int | None:
    """Which microphone to open when nobody has chosen one.

    Resolved once, in the constructor, rather than at each place that wants it. There are
    three: the stream open itself, the format probe, and the name used for logging. Letting
    them each ask separately is how you end up describing one device while capturing
    another — and the answer here differs from PortAudio's own, so that divergence would be
    real rather than theoretical.

    PortAudio's global default is an MME index on Windows. It works, and it is roughly as
    fast, but the picker only offers WASAPI devices, so a user who never chooses gets a
    device the list cannot show them, cannot explain, and cannot return them to. That is the
    reason to prefer the WASAPI default here, not speed.

    None when nothing resolves, which hands the decision back to PortAudio exactly as
    before. A machine with no capture hardware at all should fail where it always failed.
    """
    try:
        devices = list_input_devices()
    except Exception:
        return None
    idx = _default_input_index(devices)
    return idx if idx >= 0 else None


class MicrophoneStream:
    """Yields mono float32 frames of exactly FRAME_SAMPLES at SAMPLE_RATE.

    Many microphones (measurement mics, most USB interfaces) will not open at 16 kHz, so
    the device is opened at a rate it supports and resampled to 16 kHz with soxr. The audio
    callback stays minimal; resampling happens on the consumer side.
    """

    def __init__(self, device: int | str | None = None, max_queued_blocks: int = 128):
        self.device = device if device is not None else _default_device()
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
            import soxr

            resampler = soxr.ResampleStream(
                self.capture_rate, SAMPLE_RATE, 1, dtype="float32", quality="HQ"
            )

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
                block = resampler.resample_chunk(block)
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
