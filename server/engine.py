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
    }


def resolve_engine(preference: str = "auto") -> str:
    """Turn a preference into an engine that will really load.

    CTranslate2 wins when both are present. It is faster and more accurate at the same model
    size, and on a machine where it works there is no reason to take the ONNX path.
    """
    have = available_engines()
    if preference in ("ct2", "onnx"):
        return preference
    if have["ct2"]:
        return "ct2"
    if have["onnx"]:
        return "onnx"
    # Neither. Say so here rather than letting an ImportError surface from three frames deep
    # in a decode call, where it reads as a missing model rather than a missing engine.
    raise RuntimeError(
        "No speech engine is available. Expected either ctranslate2 (Intel and AMD builds) "
        "or onnxruntime-genai (ARM builds); neither could be imported."
    )


def create_engine(settings: "Settings", preference: str = "auto") -> SpeechEngine:
    """Build the engine this machine can run."""
    kind = resolve_engine(preference)
    if kind == "onnx":
        from .asr_onnx import OnnxEngine

        return OnnxEngine(settings)

    from .asr import CTranslate2Engine

    return CTranslate2Engine(settings)
