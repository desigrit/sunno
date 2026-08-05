"""CTranslate2 engine, providing fast provisional and accurate final passes.

The engine the app was built on and the one it uses everywhere CTranslate2 will load, which is
every Intel and AMD machine. See server/engine.py for the seam and server/asr_onnx.py for the
ARM implementation.
"""

from __future__ import annotations

import re
import time
from collections import deque

import numpy as np

from . import cuda_setup  # noqa: F401  (must precede ctranslate2 import)
from .config import SAMPLE_RATE, Settings
from .engine import Transcript, Word


# Whisper was trained on subtitled video, so over near-silence it reproduces the caption
# credits that pad such files. Three rules, because the risk is asymmetric: putting words
# nobody said into the transcript is bad, but silently deleting a sentence someone did say is
# worse for a user who is relying on this to follow a conversation.
#
# 1. Tokens that are never speech in this setting — an org, a URL, or a subscribe-to-channel
#    plea. Safe to match anywhere, at any length.
_HALLUCINATION_TOKENS = re.compile(
    r"""
    amara\.org
  | castingwords
  | zeoranger
  | nanostudio
  | www\.\w+\.\w+
  | \bsubs\s+hamburg\b
  | \bsubscribe\s+to\s+(?:my|our|the|this)\s+channel\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# 2. Credit phrasings, which also occur in ordinary speech ("the book was translated by
#    Tolkien", "subtitles by default are off", "transcription by hand takes forever"). Three
#    conditions must hold together before this counts as boilerplate:
#      * the sentence OPENS with the phrase — "the subtitles by that studio" is speech;
#      * it is short — a credit is terse, a sentence about one usually isn't;
#      * the attribution is a proper noun — credits name a studio or handle, whereas speech
#        says "by my sister", "by hand", "by default", "by the court reporter".
#    The last condition is what rescues the common collocations, and it is why the original
#    text is inspected rather than a lower-cased copy.
_CREDIT_OPENER = re.compile(
    r"""
    ^(?:please\s+)?
    (?:subtitl\w*|subs|transcription|transcript\w*|translated)
    \s+by\s+
    (?P<who>\S+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Credits are terse. A longer sentence that merely opens with "Translated by ..." is much
# more likely to be speech, so length is part of the test.
_MAX_CREDIT_WORDS = 8

# Leading decoration seen around credits: quotes, dashes, brackets and the music notes that
# subtitle files use to mark theme music.
_CREDIT_TRIM = "\"'“‘‚„([{-–—*_ \t♪♫†‡"

_SENTENCE_SPLIT = re.compile(r"[.!?…]+")


def _sentence_is_credit(sentence: str) -> bool:
    stripped = sentence.strip().lstrip(_CREDIT_TRIM).strip()
    if not stripped or len(stripped.split()) > _MAX_CREDIT_WORDS:
        return False

    match = _CREDIT_OPENER.match(stripped)
    if match is None:
        return False

    # A proper noun (or an ALLCAPS handle) marks an attribution; anything lower-case is
    # almost certainly ordinary speech.
    who = match.group("who").lstrip(_CREDIT_TRIM)
    return bool(who) and who[:1].isupper()


def _looks_like_caption_credit(text: str) -> bool:
    """Whether a segment is boilerplate rather than something a person said."""
    if not text or not text.strip():
        return False
    if _HALLUCINATION_TOKENS.search(text):
        return True
    # Checked per sentence so a credit tacked onto the end ("Thanks for watching! Subtitles
    # by NanoStudio") is still recognised, without letting a mid-sentence mention count.
    return any(_sentence_is_credit(part) for part in _SENTENCE_SPLIT.split(text))


def _clarity_from_logprob(avg_logprob: float) -> int:
    """Map Whisper's average token log-probability onto a 0-100 clarity score.

    Not a calibrated probability - it's a monotonic proxy for how confidently the model
    decoded the audio. Useful as relative feedback ("that came through more clearly than
    the last attempt"), not as an absolute measure. In practice avg_logprob runs from
    about -1.0 on badly-degraded speech to about -0.1 on clean, confident decodes.
    """
    scaled = (avg_logprob + 1.0) / 0.9
    return int(round(max(0.0, min(1.0, scaled)) * 100))


class CTranslate2Engine:
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
            # Measured at ~2.7% on a 9.5 s clip (941 ms vs 916 ms best-of-3), which is inside
            # run-to-run noise, and it is the only source of per-word confidence.
            word_timestamps=is_final,
        )
        # Drop segments the model itself thinks are silence. faster-whisper's own
        # no_speech_threshold only suppresses when the log-prob check also fails, which lets
        # confidently-decoded boilerplate over silence straight through.
        collected = [
            seg for seg in segments
            if seg.no_speech_prob < cfg.drop_no_speech_above
        ]
        text = " ".join(seg.text.strip() for seg in collected).strip()

        clarity = None
        if collected:
            # Duration-weighted, so a long clear sentence isn't dragged down by a short
            # trailing fragment.
            weights = [max(0.1, seg.end - seg.start) for seg in collected]
            weighted = sum(s.avg_logprob * w for s, w in zip(collected, weights))
            clarity = _clarity_from_logprob(weighted / sum(weights))

        cleaned = self._clean(text)
        words: list[Word] = []
        if cleaned and is_final:
            words = [
                Word(w.word, float(w.probability))
                for seg in collected
                for w in (seg.words or [])
            ]

        return Transcript(
            text=cleaned,
            duration_s=len(audio) / SAMPLE_RATE,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            is_final=is_final,
            clarity=clarity,
            words=words,
        )

    def _clean(self, text: str) -> str:
        """Drop known Whisper hallucinations that appear over near-silence."""
        stripped = text.lower().strip()
        if stripped in self.settings.hallucinations:
            return ""
        if _looks_like_caption_credit(text):
            return ""
        return text

    def partial(self, audio: np.ndarray) -> Transcript:
        return self._run(audio, self.settings.partial_beam_size, is_final=False)

    def final(self, audio: np.ndarray) -> Transcript:
        result = self._run(audio, self.settings.final_beam_size, is_final=True)
        self.add_context(result.text)
        return result

    def warmup(self) -> float:
        """Exercise the real decode path once, so first-use cost lands at startup.

        Deliberately uses non-silent audio through ``final()``: that is the beam-search plus
        temperature-fallback path the user's first sentence will take. Warming only the
        greedy path on zeros would leave any missing-kernel or lazy-load cost to surface
        mid-conversation, behind no loading indicator.
        """
        started = time.perf_counter()
        # Low-amplitude broadband noise: enough to drive the encoder and decoder without
        # producing text worth keeping.
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(SAMPLE_RATE * 2) * 0.01).astype(np.float32)
        self._run(audio, self.settings.final_beam_size, is_final=True)
        self.clear_context()
        return (time.perf_counter() - started) * 1000.0
