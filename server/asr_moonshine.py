"""Moonshine Voice 0.1.5, with one loaded model and bounded utterance streams."""

from __future__ import annotations

import ctypes
import importlib.util
import time
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE, Settings
from .engine import Transcript
from .paths import data_dir

MODEL_ARCHES = {
    "moonshine-tiny": "TINY_STREAMING",
    "moonshine-small": "SMALL_STREAMING",
    "moonshine-medium": "MEDIUM_STREAMING",
}


def is_available() -> bool:
    return importlib.util.find_spec("moonshine_voice") is not None


def _spec(model_id: str):
    from moonshine_voice.download import _stt_dependency_manifest, find_model_info
    from moonshine_voice.moonshine_api import ModelArch

    arch = getattr(ModelArch, MODEL_ARCHES[model_id])
    info = find_model_info("en", arch)
    manifest = _stt_dependency_manifest("en", arch)
    return arch, info, manifest["groups"][0]


def model_path(model_id: str) -> Path:
    """Resolve locally without invoking any downloader or network operation."""
    _, _, group = _spec(model_id)
    root = data_dir() / "moonshine" / group["base_url"].removeprefix("https://")
    for item in group["files"]:
        path = root / item["name"]
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise FileNotFoundError(f"{model_id} needs to be downloaded before it can start")
    return root


def total_bytes(model_id: str) -> int:
    return sum(item["size"] for item in _spec(model_id)[2]["files"])


def download(model_id: str, on_progress=None) -> str:
    from moonshine_voice.download import download_model_from_info

    _, info, _ = _spec(model_id)
    total = total_bytes(model_id)
    report = (lambda fraction, _: on_progress(round(fraction * total), total)) if on_progress else None
    path, _ = download_model_from_info(
        info, cache_root=data_dir() / "moonshine", on_progress=report
    )
    model_path(model_id)  # reject incomplete cache entries before claiming success
    return path


class MoonshineEngine:
    def __init__(self, settings: Settings) -> None:
        from moonshine_voice.transcriber import Transcriber

        self.settings = settings
        arch, _, _ = _spec(settings.model_size)
        self._model = Transcriber(
            model_path(settings.model_size), arch,
            # Sunno schedules updates; disable the wrapper's second scheduler.
            update_interval=3600.0,
            options={"identify_speakers": "false", "ort_providers": "CPU"},
        )
        if settings.vocabulary:
            self._model.set_keyterms(list(settings.vocabulary))
        self._stream = None
        self._samples = 0
        self._lines: dict[int, object] = {}

    def start_stream(self) -> None:
        self.reset_stream()
        self._stream = self._model.create_stream(update_interval=3600.0)
        self._stream.start()

    def reset_stream(self) -> None:
        if self._stream is not None:
            from moonshine_voice.errors import check_error

            stream, self._stream = self._stream, None
            # Release each utterance's native allocation explicitly. The pinned
            # wrapper's close() ignores the C return code; check it here.
            check_error(self._model._lib.moonshine_free_stream(
                self._model._handle, stream._handle
            ))
            stream._handle = None
        self._samples = 0
        self._lines = {}

    def stream(self, audio: np.ndarray, *, final: bool = False) -> Transcript:
        from moonshine_voice.errors import check_error

        if self._stream is None:
            raise RuntimeError("No active Moonshine stream")
        started = time.perf_counter()
        samples = np.ascontiguousarray(audio, dtype=np.float32)
        if samples.size:
            # Pass the array directly. The Python convenience API expands every
            # sample into Python objects and can schedule an extra decode.
            check_error(self._model._lib.moonshine_transcribe_add_audio_to_stream(
                self._model._handle, self._stream._handle,
                samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                samples.size, SAMPLE_RATE, 0,
            ))
        self._samples += samples.size
        if final:
            check_error(self._model._lib.moonshine_stop_stream(
                self._model._handle, self._stream._handle
            ))
        decoded = self._stream.update_transcription()
        # Retain completed lines if an update only includes changed lines.
        for line in decoded.lines:
            self._lines[line.line_id] = line
        text = " ".join(
            line.text.strip() for line in sorted(
                self._lines.values(), key=lambda line: (line.start_time, line.line_id)
            ) if line.text.strip()
        )
        result = Transcript(text, self._samples / SAMPLE_RATE,
                            (time.perf_counter() - started) * 1000, final)
        if final:
            self.reset_stream()
        return result

    def partial(self, audio: np.ndarray) -> Transcript:
        self.start_stream()
        try:
            return self.stream(audio)
        finally:
            self.reset_stream()

    def final(self, audio: np.ndarray) -> Transcript:
        self.start_stream()
        try:
            return self.stream(audio, final=True)
        finally:
            self.reset_stream()

    def warmup(self) -> float:
        return self.final(np.zeros(SAMPLE_RATE, np.float32)).latency_ms

    def close(self) -> None:
        self.reset_stream()
        self._model.close()
