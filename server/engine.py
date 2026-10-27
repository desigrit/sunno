"""What the pipeline needs from a speech engine, and which one to build.

Whisper engines accept complete utterance snapshots. Streaming transducers, Moonshine,
and platform-native helpers additionally accept each new sample once through the optional
incremental interface. Model loading, prompts, compute providers, and resource cleanup
stay inside each implementation. New engines remain opt-in until evaluated in Sunno.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .config import Settings


@dataclass
class Word:
    text: str
    probability: float
    # Seconds from the start of the utterance this word was decoded from, not from the start
    # of the session or the recording. The caller knows when the utterance began and can add
    # the offset; this object cannot, and inventing a session-relative number here would be
    # wrong the moment an utterance is re-decoded.
    #
    # None on engines that do not produce timings. The streaming recognisers emit no words at
    # all, so their transcripts have nothing to sync against and a saved recording made with
    # one of them can only be navigated line by line.
    start_s: float | None = None
    end_s: float | None = None


@dataclass
class Transcript:
    text: str
    duration_s: float
    latency_ms: float
    is_final: bool
    # 0-100, how confidently the model decoded this. None when the engine cannot say: the
    # ONNX path exposes no per-segment log-probabilities, and the UI already treats the
    # figure as optional rather than defaulting it to something invented.
    clarity: int | None = None
    words: list[Word] = field(default_factory=list)


@runtime_checkable
class SpeechEngine(Protocol):
    """The whole of what the pipeline requires.

    ``settings`` is read for the model name and device when reporting what is running.
    ``partial`` is the greedy pass whose words appear first; ``final`` is the careful pass that
    replaces them. ``warmup`` runs one real decode so the first sentence someone speaks does not
    pay the load cost.
    """

    settings: "Settings"

    def partial(self, audio: np.ndarray) -> Transcript: ...

    def final(self, audio: np.ndarray) -> Transcript: ...

    def warmup(self) -> float: ...


@runtime_checkable
class IncrementalSpeechEngine(Protocol):
    """Optional streaming interface, used only on the serial ASR worker thread.

    A stream owns one utterance. Every sample is delivered exactly once, even
    when queued caption refreshes are coalesced. Reset abandons decoder state.
    Batch engines keep the existing partial/final interface.
    """

    def start_stream(self) -> None: ...

    def stream(self, audio: np.ndarray, *, final: bool = False) -> Transcript: ...

    def reset_stream(self) -> None: ...


def available_engines() -> dict[str, bool]:
    """Which engines could actually run here.

    Import-only checks, deliberately: both are heavy to construct, and the answer is needed
    before a model is chosen so the picker can offer models the machine can decode.
    """
    import importlib.util

    return {
        "ct2": importlib.util.find_spec("ctranslate2") is not None,
        "onnx": importlib.util.find_spec("onnxruntime_genai") is not None,
        "stream": importlib.util.find_spec("sherpa_onnx") is not None,
        "moonshine": importlib.util.find_spec("moonshine_voice") is not None,
    }


def resolve_engine(preference: str = "auto", model_id: str | None = None) -> str:
    """Turn a preference into an engine that will really load.

    The model decides first. A streaming transducer and a Whisper checkpoint are different
    artifacts, not two settings of one thing, so asking for one of these models is asking
    for this engine and there is nothing to weigh.

    Otherwise CTranslate2 wins when it is present, because at the same model size it is
    faster and more accurate than the ONNX path, which exists for Windows on ARM where
    CTranslate2 has no wheel at all.

    Note what is deliberately absent: no rule here prefers the streaming engine on a
    machine with no graphics card. Which model suits a machine is already decided, better,
    by hardware.default_model measuring it. Duplicating that judgement here would override
    people who chose a model on purpose.
    """
    have = available_engines()

    if model_id is not None:
        from .models import is_stream_model
        from .asr_moonshine import MODEL_ARCHES
        from .asr_native import NATIVE_MODELS

        if model_id in MODEL_ARCHES:
            if not have.get("moonshine"):
                raise RuntimeError("Moonshine is not included for this processor architecture")
            return "moonshine"
        if model_id in NATIVE_MODELS:
            return "native"

        if is_stream_model(model_id):
            if not have["stream"]:
                raise RuntimeError(
                    f"'{model_id}' needs the streaming engine, but sherpa-onnx could not "
                    "be imported."
                )
            return "stream"

    if preference in ("ct2", "onnx", "stream"):
        return preference
    if have["ct2"]:
        return "ct2"
    if have["onnx"]:
        return "onnx"
    # Deliberately no bare fallback to "stream" here. sherpa-onnx is a hard requirement of
    # the app, so it is always importable, and falling through to it would make the error
    # below unreachable: a machine whose CTranslate2 install is broken would be handed the
    # streaming engine for a Whisper id, which then raises a KeyError naming the model.
    # That is exactly the "reads as a missing model rather than a missing engine" failure
    # this check exists to prevent. A streaming model already returned above.
    raise RuntimeError(
        "No speech engine is available for this model. Expected ctranslate2 (Intel and "
        "AMD builds) or onnxruntime-genai (ARM builds); neither could be imported."
    )


def create_engine(settings: "Settings", preference: str = "auto") -> SpeechEngine:
    """Build the engine this machine can run."""
    kind = resolve_engine(preference, getattr(settings, "model_size", None))
    if kind == "moonshine":
        from .asr_moonshine import MoonshineEngine

        return MoonshineEngine(settings)
    if kind == "native":
        from .asr_native import NativeSpeechEngine

        return NativeSpeechEngine(settings)
    if kind == "stream":
        from .asr_stream import StreamingEngine

        return StreamingEngine(settings)
    if kind == "onnx":
        from .asr_onnx import OnnxEngine

        return OnnxEngine(settings)

    from .asr import CTranslate2Engine

    return CTranslate2Engine(settings)
