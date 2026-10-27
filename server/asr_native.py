"""Bounded local IPC for Apple SpeechAnalyzer and Windows AI speech.

Helpers receive audio from Sunno, never open a microphone independently, and
never use a cloud recognizer. Models are provisioned only by the download action.
"""

from __future__ import annotations

import base64
import functools
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE, Settings
from .engine import Transcript

NATIVE_MODELS = {"apple-speech", "windows-speech"}


def service_path(model_id: str) -> Path | None:
    root = Path(__file__).resolve().parent.parent
    if model_id == "apple-speech" and sys.platform == "darwin":
        candidates = [root / "speech-service/.build/release/speech-service"]
    elif model_id == "windows-speech" and sys.platform == "win32":
        from .hardware import native_machine

        architecture = "arm64" if native_machine().lower() == "arm64" else "x64"
        candidates = [root / f"native-speech/{architecture}/Sunno.SpeechService.exe",
                      root / f"windows-speech-service/bin/Release/net8.0-windows10.0.26100.0/win-{architecture}/publish/Sunno.SpeechService.exe"]
    else:
        return None
    return next((path for path in candidates if path.is_file()), None)


class NativeService:
    def __init__(self, model_id: str) -> None:
        executable = service_path(model_id)
        if executable is None:
            raise RuntimeError("The native speech engine is not included for this platform")
        self._process = subprocess.Popen(
            [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._responses: queue.Queue = queue.Queue(maxsize=4)
        self._last_error = ""
        self._closed = False
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_errors, daemon=True).start()

    def _read(self) -> None:
        try:
            for line in self._process.stdout:
                # Protocol messages contain at most one short utterance of text.
                if len(line) > 1024 * 1024:
                    break
                self._responses.put(json.loads(line), timeout=1)
        except (ValueError, queue.Full):
            pass
        finally:
            try:
                self._responses.put_nowait({"ok": False, "error": "Native speech helper exited"})
            except queue.Full:
                pass

    def _read_errors(self) -> None:
        for line in self._process.stderr:
            self._last_error = line.strip()[-500:]

    def request(self, command: str, *, timeout: float = 30.0, **values) -> dict:
        if self._closed:
            raise RuntimeError("Native speech helper is closed")
        try:
            self._process.stdin.write(json.dumps({"command": command, **values}) + "\n")
            self._process.stdin.flush()
            response = self._responses.get(timeout=timeout)
        except (BrokenPipeError, OSError, queue.Empty) as exc:
            self.close()
            raise RuntimeError(f"Native speech engine did not respond. {self._last_error}") from exc
        if not response.get("ok", False):
            raise RuntimeError(response.get("error") or "Native speech engine failed")
        return response

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            stream.close()


def availability(model_id: str) -> dict:
    # The OS can install, remove, or re-enable the model while Sunno is open.
    # Share short-lived probes without freezing the first result for a session.
    return _availability(model_id, int(time.monotonic() // 15))


def clear_availability() -> None:
    _availability.cache_clear()


@functools.lru_cache(maxsize=4)
def _availability(model_id: str, freshness: int) -> dict:
    if service_path(model_id) is None:
        return {"supported": False, "available": False,
                "reason": "Not included for this platform"}
    service = None
    try:
        service = NativeService(model_id)
        return service.request("probe", timeout=15)
    except Exception as exc:
        return {"supported": False, "available": False, "reason": str(exc)}
    finally:
        if service:
            service.close()


def download(model_id: str, on_progress=None) -> str:
    service = NativeService(model_id)
    try:
        if on_progress:
            on_progress(0, 0)  # system downloads do not expose a byte total
        service.request("prepare", timeout=1800)
        if on_progress:
            on_progress(1, 1)
        return model_id
    finally:
        # Even a setup error can mean the OS continued downloading in the
        # background. Do not keep reporting the pre-download readiness forever.
        clear_availability()
        service.close()


class NativeSpeechEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._service = NativeService(settings.model_size)
        self._active = False
        self._samples = 0
        try:
            self._service.request("load", timeout=120)
        except Exception:
            self._service.close()
            raise

    def start_stream(self) -> None:
        self.reset_stream()
        if self._service._closed or self._service._process.poll() is not None:
            self._service.close()
            self._service = NativeService(self.settings.model_size)
            try:
                self._service.request("load", timeout=120)
            except Exception:
                self._service.close()
                raise
        self._service.request("start")
        self._active = True

    def reset_stream(self) -> None:
        if self._active:
            self._active = False
            try:
                self._service.request("reset")
            except Exception:
                self._service.close()
        self._samples = 0

    def stream(self, audio: np.ndarray, *, final: bool = False) -> Transcript:
        if not self._active:
            raise RuntimeError("No active native speech stream")
        started = time.perf_counter()
        self._samples += len(audio)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
        try:
            response = self._service.request(
                "audio", samples=base64.b64encode(pcm.tobytes()).decode("ascii"), final=final,
            )
        except Exception:
            self._service.close()
            raise
        result = Transcript(str(response.get("text", "")), self._samples / SAMPLE_RATE,
                            (time.perf_counter() - started) * 1000, final)
        if final:
            self._active = False
            self._samples = 0
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
        # Loading is sufficient for OS engines. Do not manufacture a speech
        # request or acquire microphone access as a warmup side effect.
        return 0.0

    def close(self) -> None:
        self._active = False
        self._service.close()
