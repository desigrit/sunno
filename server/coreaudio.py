"""Windows endpoint identity and shared-mode capture, owned by the capture child.

Endpoint IDs survive enumeration changes. Both microphones and loopback use the
endpoint's mix format, so switching never depends on a cached PortAudio index.
No third-party COM runtime is required.
"""

from __future__ import annotations

import ctypes as c
import functools
import sys
import time
import uuid
from contextlib import contextmanager

import numpy as np

from .config import FRAME_SAMPLES, SAMPLE_RATE


class CaptureError(RuntimeError):
    def __init__(self, code: str, message: str, retryable: bool = True):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class GUID(c.Structure):
    _fields_ = [("data1", c.c_uint32), ("data2", c.c_uint16),
                ("data3", c.c_uint16), ("data4", c.c_ubyte * 8)]

    @classmethod
    def parse(cls, value):
        return cls.from_buffer_copy(uuid.UUID(value).bytes_le)


class PropertyKey(c.Structure):
    _fields_ = [("fmtid", GUID), ("pid", c.c_uint32)]


class VariantValue(c.Union):
    _fields_ = [("text", c.c_void_p), ("storage", c.c_byte * 16)]


class PropVariant(c.Structure):
    _fields_ = [("vt", c.c_uint16), ("reserved", c.c_uint16 * 3),
                ("value", VariantValue)]


class WaveFormat(c.Structure):
    _pack_ = 1
    _fields_ = [("tag", c.c_uint16), ("channels", c.c_uint16),
                ("rate", c.c_uint32), ("bytes_per_second", c.c_uint32),
                ("block_align", c.c_uint16), ("bits", c.c_uint16),
                ("extra_size", c.c_uint16)]


_ENUM_CLASS = GUID.parse("BCDE0395-E52F-467C-8E3D-C4579291692E")
_ENUM_IID = GUID.parse("A95664D2-9614-4F35-A746-DE8DB63617E6")
_CLIENT_IID = GUID.parse("1CB9AD4C-DBFA-4C32-B178-C2F568A703B2")
_CAPTURE_IID = GUID.parse("C8ADBD64-E71E-48A0-A4DE-185C395CD317")
_NAME_KEY = PropertyKey(GUID.parse("A45C254E-DF1C-4EFD-8020-67D146A850E0"), 14)
_CALL = getattr(c, "WINFUNCTYPE", c.CFUNCTYPE)


@functools.lru_cache(maxsize=1)
def _ole():
    if sys.platform != "win32":
        raise CaptureError("unsupported", "Windows audio capture is unavailable.", False)
    lib = c.WinDLL("ole32")
    lib.CoInitializeEx.argtypes = [c.c_void_p, c.c_uint32]
    lib.CoInitializeEx.restype = c.c_int32
    lib.CoCreateInstance.argtypes = [c.POINTER(GUID), c.c_void_p, c.c_uint32,
                                    c.POINTER(GUID), c.POINTER(c.c_void_p)]
    lib.CoCreateInstance.restype = c.c_int32
    lib.CoTaskMemFree.argtypes = [c.c_void_p]
    lib.CoTaskMemFree.restype = None
    lib.CoUninitialize.restype = None
    lib.PropVariantClear.argtypes = [c.POINTER(PropVariant)]
    lib.PropVariantClear.restype = c.c_int32
    return lib


def _check(hr):
    if hr < 0:
        code = hr & 0xFFFFFFFF
        if code == 0x80070005:
            raise CaptureError("capture_denied", "Windows is blocking access to this input.", False)
        raise CaptureError("device_unavailable", f"Windows audio is unavailable (0x{code:08X}).")


def _call(pointer, slot, types=(), *args):
    table = c.cast(pointer, c.POINTER(c.POINTER(c.c_void_p))).contents
    hr = _CALL(c.c_int32, c.c_void_p, *types)(table[slot])(pointer, *args)
    _check(hr)


def _release(pointer):
    if pointer:
        table = c.cast(pointer, c.POINTER(c.POINTER(c.c_void_p))).contents
        _CALL(c.c_uint32, c.c_void_p)(table[2])(pointer)


@contextmanager
def apartment():
    hr = _ole().CoInitializeEx(None, 2)
    if hr != -2147417850:  # An existing MTA apartment is also usable.
        _check(hr)
    try:
        yield
    finally:
        if hr >= 0:
            _ole().CoUninitialize()


def _enumerator():
    pointer = c.c_void_p()
    _check(_ole().CoCreateInstance(c.byref(_ENUM_CLASS), None, 1,
                                 c.byref(_ENUM_IID), c.byref(pointer)))
    return pointer


def _device_id(device):
    value = c.c_void_p()
    _call(device, 5, (c.POINTER(c.c_void_p),), c.byref(value))
    try:
        if not value:
            raise CaptureError("device_unavailable", "The audio endpoint is no longer available.")
        return c.wstring_at(value)
    finally:
        _ole().CoTaskMemFree(value)


def _device_name(device):
    store, value = c.c_void_p(), PropVariant()
    _call(device, 4, (c.c_uint32, c.POINTER(c.c_void_p)), 0, c.byref(store))
    try:
        _call(store, 5, (c.POINTER(PropertyKey), c.POINTER(PropVariant)),
              c.byref(_NAME_KEY), c.byref(value))
        return c.wstring_at(value.value.text) if value.vt == 31 and value.value.text else "Audio device"
    finally:
        _ole().PropVariantClear(c.byref(value))
        _release(store)


def _default_id(enumerator, kind):
    device = c.c_void_p()
    try:
        _call(enumerator, 4, (c.c_int, c.c_int, c.POINTER(c.c_void_p)),
              0 if kind == "loopback" else 1, 0, c.byref(device))
        return _device_id(device)
    except CaptureError:
        return None
    finally:
        _release(device)


def default_endpoint_id(kind):
    with apartment():
        enumerator = _enumerator()
        try:
            return _default_id(enumerator, kind)
        finally:
            _release(enumerator)


def list_endpoints(kind):
    with apartment():
        enumerator, collection = _enumerator(), c.c_void_p()
        try:
            default = _default_id(enumerator, kind)
            _call(enumerator, 3, (c.c_int, c.c_uint32, c.POINTER(c.c_void_p)),
                  0 if kind == "loopback" else 1, 1, c.byref(collection))
            count = c.c_uint32()
            _call(collection, 3, (c.POINTER(c.c_uint32),), c.byref(count))
            devices = []
            for index in range(count.value):
                device = c.c_void_p()
                try:
                    _call(collection, 4, (c.c_uint32, c.POINTER(c.c_void_p)), index, c.byref(device))
                    identity = _device_id(device)
                    try:
                        name = _device_name(device)
                    except CaptureError:
                        name = "Audio device"
                    devices.append({"index": index, "name": name,
                                    "endpoint_id": identity, "hostapi": "Windows WASAPI",
                                    "loopback": kind == "loopback",
                                    "is_default_output" if kind == "loopback" else
                                    "is_default_input": identity == default})
                except CaptureError:
                    continue  # An endpoint can disappear while the collection is being read.
                finally:
                    _release(device)
            return devices
        finally:
            _release(collection)
            _release(enumerator)


def decode_packet(raw, tag, bits, channels):
    """Decode the native mix format before releasing the WASAPI packet."""
    if tag == 3 and bits == 32:
        samples = np.frombuffer(raw, dtype="<f4")
    elif tag == 1 and bits in (16, 32):
        samples = np.frombuffer(raw, dtype=f"<i{bits // 8}").astype(np.float32) / (2 ** (bits - 1))
    elif tag == 1 and bits == 24:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        samples = ((values ^ 0x800000) - 0x800000).astype(np.float32) / 8388608
    else:
        raise CaptureError("unsupported_format", "This input uses an unsupported audio format.", False)
    return np.nan_to_num(samples.reshape(-1, channels).mean(axis=1),
                         nan=0, posinf=0, neginf=0).astype(np.float32)


class EndpointStream:
    """One native endpoint, including healthy silence from an idle loopback."""
    def __init__(self, endpoint_id, kind):
        self.endpoint_id, self.kind = endpoint_id, kind
        self._client = self._capture = self._device = None
        self._apartment = None

    def __enter__(self):
        self._apartment = apartment()
        self._apartment.__enter__()
        enumerator, mix = None, c.c_void_p()
        try:
            enumerator = _enumerator()
            self._device = c.c_void_p()
            _call(enumerator, 5, (c.c_wchar_p, c.POINTER(c.c_void_p)),
                  self.endpoint_id, c.byref(self._device))
            self._client = c.c_void_p()
            _call(self._device, 3, (c.POINTER(GUID), c.c_uint32, c.c_void_p,
                                   c.POINTER(c.c_void_p)), c.byref(_CLIENT_IID), 23, None,
                  c.byref(self._client))
            _call(self._client, 8, (c.POINTER(c.c_void_p),), c.byref(mix))
            fmt = c.cast(mix, c.POINTER(WaveFormat)).contents
            self.capture_rate, self.capture_channels = fmt.rate, fmt.channels
            self._bits, self._align, self._tag = fmt.bits, fmt.block_align, fmt.tag
            if fmt.tag == 0xFFFE:
                if fmt.extra_size < 22:
                    raise CaptureError("unsupported_format", "This input uses an unsupported audio format.", False)
                self._tag = c.c_uint32.from_address(mix.value + 24).value
            if not (1 <= self.capture_channels <= 32 and 8000 <= self.capture_rate <= 384000
                    and self._align == self.capture_channels * (self._bits // 8)):
                raise CaptureError("unsupported_format", "This input uses an unsupported audio format.", False)
            # Reject unsupported formats before starting the device.
            decode_packet(bytes(self._align), self._tag, self._bits, self.capture_channels)
            _call(self._client, 3, (c.c_int, c.c_uint32, c.c_int64, c.c_int64,
                                   c.c_void_p, c.c_void_p), 0,
                  0x20000 if self.kind == "loopback" else 0, 2_000_000, 0, mix, None)
            self._capture = c.c_void_p()
            _call(self._client, 14, (c.POINTER(GUID), c.POINTER(c.c_void_p)),
                  c.byref(_CAPTURE_IID), c.byref(self._capture))
            _call(self._client, 10)
            return self
        except BaseException:
            if mix:
                _ole().CoTaskMemFree(mix)
                mix = c.c_void_p()
            _release(enumerator)
            enumerator = None
            self.__exit__()
            raise
        finally:
            if mix:
                _ole().CoTaskMemFree(mix)
            _release(enumerator)

    def __exit__(self, *exc):
        if self._client:
            try:
                _call(self._client, 11)
            except CaptureError:
                pass
        for field in ("_capture", "_client", "_device"):
            _release(getattr(self, field))
            setattr(self, field, None)
        if self._apartment is not None:
            self._apartment.__exit__(None, None, None)
            self._apartment = None

    def frames(self, should_continue):
        import soxr
        resampler = (soxr.ResampleStream(self.capture_rate, SAMPLE_RATE, 1,
                                       dtype="float32", quality="HQ")
                     if self.capture_rate != SAMPLE_RATE else None)
        pending = np.empty(0, dtype=np.float32)
        last_audio = last_yield = time.monotonic()
        while should_continue():
            available = c.c_uint32()
            _call(self._capture, 5, (c.POINTER(c.c_uint32),), c.byref(available))
            if not available.value:
                # This call detects endpoint invalidation even when nothing is playing.
                padding = c.c_uint32()
                _call(self._client, 6, (c.POINTER(c.c_uint32),), c.byref(padding))
                now = time.monotonic()
                if self.kind == "loopback":
                    owed = min(16, int((now - last_yield) * SAMPLE_RATE / FRAME_SAMPLES))
                    for _ in range(owed):
                        yield np.zeros(FRAME_SAMPLES, dtype=np.float32)
                        last_yield += FRAME_SAMPLES / SAMPLE_RATE
                elif now - last_audio > 2:
                    raise CaptureError("capture_stalled", "The microphone stopped sending audio.")
                time.sleep(0.01)
                continue
            pointer, count, flags = c.c_void_p(), c.c_uint32(), c.c_uint32()
            _call(self._capture, 3, (c.POINTER(c.c_void_p), c.POINTER(c.c_uint32),
                                    c.POINTER(c.c_uint32), c.c_void_p, c.c_void_p),
                  c.byref(pointer), c.byref(count), c.byref(flags), None, None)
            try:
                block = (np.zeros(count.value, dtype=np.float32) if flags.value & 2 else
                         decode_packet(c.string_at(pointer, count.value * self._align),
                                       self._tag, self._bits, self.capture_channels))
            finally:
                _call(self._capture, 4, (c.c_uint32,), count)
            last_audio = time.monotonic()
            if resampler is not None:
                block = resampler.resample_chunk(block)
            pending = np.concatenate((pending, block))
            while pending.size >= FRAME_SAMPLES:
                if not should_continue():
                    return
                yield pending[:FRAME_SAMPLES]
                pending = pending[FRAME_SAMPLES:]
                last_yield = time.monotonic()
