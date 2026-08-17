"""What the pipeline needs from a speech engine, and which one to build.

Two engines exist because CTranslate2 - fast, accurate, and what the app was built on - has
no win_arm64 wheel and will not build there. A Snapdragon laptop cannot run it at all, so ARM
needs a second implementation over ONNX Runtime rather than a slower configuration of the first.

The seam is deliberately narrow. The pipeline only ever asks an engine to decode audio twice
per utterance - once fast, once well - and to warm itself up. Everything else about how a model
is loaded, prompted, or quantised stays inside the implementation, which is why a second one is
a new file rather than a set of branches through the existing code.
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
    if kind == "stream":
        from .asr_stream import StreamingEngine

        return StreamingEngine(settings)
    if kind == "onnx":
        from .asr_onnx import OnnxEngine

        return OnnxEngine(settings)

    from .asr import CTranslate2Engine

    return CTranslate2Engine(settings)
