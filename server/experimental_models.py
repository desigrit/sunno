"""Opt-in engines. No latency or accuracy ranking is invented before testing."""

from __future__ import annotations

import sys

from .distribution import model_allowed

CATALOG = [
    {"id": "moonshine-small", "name": "Moonshine Small Streaming",
     "detail": "English streaming. Experimental; speed varies by PC.",
     "approx_mb": 136, "languages": "English"},
    {"id": "moonshine-medium", "name": "Moonshine Medium Streaming",
     "detail": "Larger English streaming model. Experimental; speed varies by PC.",
     "approx_mb": 257, "languages": "English"},
    {"id": "moonshine-tiny", "name": "Moonshine Tiny Streaming",
     "detail": "Smallest Moonshine model. Experimental; check accuracy on your speech.",
     "approx_mb": 43, "languages": "English"},
    {"id": "windows-speech", "name": "Windows AI Speech",
     "detail": "Experimental on-device engine. Windows manages the model download.",
     "approx_mb": 0, "languages": "English"},
    {"id": "apple-speech", "name": "Apple SpeechAnalyzer",
     "detail": "On-device streaming for macOS 26 or later. Apple manages the model download.",
     "approx_mb": 0, "languages": "English"},
]
MODEL_IDS = frozenset(entry["id"] for entry in CATALOG)


def status(model_id: str) -> dict:
    if not model_allowed(model_id):
        return {"supported": False, "available": False,
                "reason": "Windows AI Speech is not included in the Microsoft Store version."}
    from . import asr_moonshine, asr_native

    if model_id in asr_moonshine.MODEL_ARCHES:
        supported = asr_moonshine.is_available()
        available = False
        reason = ""
        if supported:
            try:
                asr_moonshine.model_path(model_id)
                available = True
            except FileNotFoundError:
                pass
            except Exception as exc:
                supported, reason = False, f"Moonshine could not load: {exc}"
        else:
            reason = "Moonshine is not included for this processor architecture"
        return {"supported": supported, "available": available, "reason": reason}
    return asr_native.availability(model_id)


def catalog_entries() -> list[dict]:
    entries = []
    for entry in CATALOG:
        model_id = entry["id"]
        if not model_allowed(model_id):
            continue
        if model_id == "apple-speech" and sys.platform != "darwin":
            continue
        if model_id == "windows-speech" and sys.platform != "win32":
            continue
        capability = status(model_id)
        supported = bool(capability.get("supported"))
        entries.append(dict(
            entry, available=bool(capability.get("available")), supported=supported,
            detail=entry["detail"] if supported else str(capability.get("reason", "Unavailable")),
            lag_ms=0, lag_text="No device-specific delay estimate", responsive=False,
            auto_select=False, uses_gpu=False, experimental=True,
        ))
    return entries


def download(model_id: str, on_progress=None) -> str:
    from . import asr_moonshine, asr_native

    if model_id in asr_moonshine.MODEL_ARCHES:
        return asr_moonshine.download(model_id, on_progress)
    return asr_native.download(model_id, on_progress)
