"""faster-whisper engine wrapper providing fast provisional and accurate final passes."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import cuda_setup  # noqa: F401  (must precede ctranslate2 import)
from .config import SAMPLE_RATE, Settings


@dataclass
class Transcript:
    text: str
    duration_s: float
    latency_ms: float
    is_final: bool
    clarity: int | None = None  # 0-100, how confidently the model decoded this


def _clarity_from_logprob(avg_logprob: float) -> int:
    """Map Whisper's average token log-probability onto a 0-100 clarity score.

    Not a calibrated probability - it's a monotonic proxy for how confidently the model
    decoded the audio. Useful as relative feedback ("that came through more clearly than
    the last attempt"), not as an absolute measure. In practice avg_logprob runs from
    about -1.0 on badly-degraded speech to about -0.1 on clean, confident decodes.
    """
    scaled = (avg_logprob + 1.0) / 0.9
    return int(round(max(0.0, min(1.0, scaled)) * 100))


class WhisperEngine:
    """Wraps a single loaded Whisper model used for both provisional and final decoding.

    Only one model is held in VRAM; the two passes differ purely by decoding parameters
    (greedy for speed, beam search for accuracy).
    """

    def __init__(self, settings: Settings) -> None:
        from faster_whisper import WhisperModel

        self.settings = settings
        self._model = WhisperModel(
            settings.model_size,
            device=settings.device,
            compute_type=settings.compute_type,
        )
        self._context: deque[str] = deque(maxlen=6)
        self._context_updated = 0.0

    # --- context -------------------------------------------------------
    def _build_prompt(self) -> str | None:
        """Vocabulary plus recent conversation, as Whisper's initial_prompt.

        Giving Whisper prior context improves proper nouns and continuity. It is capped and
        expired because unbounded feedback of the model's own output can seed hallucination
        loops - the same reason condition_on_previous_text stays off.
        """
        if not self.settings.use_context_prompt:
            return None

        parts: list[str] = []
        if self.settings.vocabulary:
            parts.append(", ".join(self.settings.vocabulary) + ".")

        if self._context and (
            time.monotonic() - self._context_updated <= self.settings.context_expiry_s
        ):
            recent = " ".join(self._context)
            parts.append(recent[-self.settings.context_chars :])

        prompt = " ".join(parts).strip()
        return prompt or None

    def add_context(self, text: str) -> None:
        if text:
            self._context.append(text)
            self._context_updated = time.monotonic()

    def clear_context(self) -> None:
        self._context.clear()

    # --- decoding ------------------------------------------------------
    def _run(self, audio: np.ndarray, beam_size: int, is_final: bool) -> Transcript:
        started = time.perf_counter()
        cfg = self.settings
        segments, _info = self._model.transcribe(
            audio,
            language=cfg.language,
            beam_size=beam_size,
            # Temperature fallback only on the final pass: provisional text is replaced
            # moments later anyway, and retries would blow the partial latency budget.
            temperature=cfg.temperature_fallback if is_final else 0.0,
            initial_prompt=self._build_prompt(),
            condition_on_previous_text=False,  # avoids run-away hallucination loops
            vad_filter=False,  # our own VAD already gated this audio
            no_speech_threshold=cfg.no_speech_threshold,
            log_prob_threshold=cfg.log_prob_threshold,
            compression_ratio_threshold=cfg.compression_ratio_threshold,
            word_timestamps=False,
        )
        collected = list(segments)
        text = " ".join(seg.text.strip() for seg in collected).strip()

        clarity = None
        if collected:
            # Duration-weighted, so a long clear sentence isn't dragged down by a short
            # trailing fragment.
            weights = [max(0.1, seg.end - seg.start) for seg in collected]
            weighted = sum(s.avg_logprob * w for s, w in zip(collected, weights))
            clarity = _clarity_from_logprob(weighted / sum(weights))

        return Transcript(
            text=self._clean(text),
            duration_s=len(audio) / SAMPLE_RATE,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            is_final=is_final,
            clarity=clarity,
        )

    def _clean(self, text: str) -> str:
        """Drop known Whisper hallucinations that appear over near-silence."""
        if text.lower().strip() in self.settings.hallucinations:
            return ""
        return text

    def partial(self, audio: np.ndarray) -> Transcript:
        return self._run(audio, self.settings.partial_beam_size, is_final=False)

    def final(self, audio: np.ndarray) -> Transcript:
        result = self._run(audio, self.settings.final_beam_size, is_final=True)
        self.add_context(result.text)
        return result

    def warmup(self) -> float:
        """Run one throwaway inference so the first real utterance isn't slow."""
        started = time.perf_counter()
        self._run(np.zeros(SAMPLE_RATE, dtype=np.float32), beam_size=1, is_final=False)
        self.clear_context()
        return (time.perf_counter() - started) * 1000.0
