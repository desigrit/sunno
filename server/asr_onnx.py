"""ONNX Runtime engine, for machines where CTranslate2 cannot run.

That means Windows on ARM: CTranslate2 publishes no win_arm64 wheel and will not build one, so
a Snapdragon laptop needs a different engine rather than a slower setting.

Whisper runs here through onnxruntime-genai, which owns the mel front end, the decode loop and
the tokenizer, so this file is mostly translation between the pipeline's numpy frames and what
genai expects. Measured on a Snapdragon X Elite: base decodes in about 300 ms and tiny in about
130 ms, against a 1000 ms budget - the same figures an i9-14900K posts for the same models.

Two things the CTranslate2 path provides and this one does not:

* Clarity, the per-utterance confidence badge. It comes from segment log-probabilities, which
  genai does not expose. Transcript.clarity is already optional and the UI has a toggle for it.
* Per-word uncertainty, which greys words the model was unsure of. That needs per-token
  probabilities, also unavailable here.

Both are reported as absent rather than invented. A confidence figure that is really a guess is
worse than no figure at all for someone relying on this to follow a conversation.
"""

from __future__ import annotations

import io
import time
import wave
from collections import deque

import numpy as np

from .config import SAMPLE_RATE, Settings
from .engine import Transcript

# Whisper decodes from a forced prefix declaring language and task. Timestamps are switched off
# because the pipeline does its own segmentation and asking for them only makes the decoder emit
# tokens nobody reads.
_PROMPT_TEMPLATE = "<|startoftranscript|><|{lang}|><|transcribe|><|notimestamps|>"


class OnnxEngine:
    """A Whisper model held once and used for both the provisional and final pass."""

    def __init__(self, settings: Settings) -> None:
        import onnxruntime_genai as og

        from .models import onnx_model_path

        self.settings = settings
        self._og = og

        model_dir = onnx_model_path(settings.model_size)
        self._model = og.Model(str(model_dir))
        self._processor = self._model.create_multimodal_processor()
        self._prompt = _PROMPT_TEMPLATE.format(lang=settings.language or "en")

        self._context: deque[str] = deque(maxlen=6)
        self._context_updated = 0.0

    # --- context -------------------------------------------------------
    def add_context(self, text: str) -> None:
        if text:
            self._context.append(text)
            self._context_updated = time.monotonic()

    def clear_context(self) -> None:
        self._context.clear()

    # --- decoding ------------------------------------------------------
    def _to_wav(self, audio: np.ndarray) -> bytes:
        """genai takes a file or bytes, never an array, so live audio is encoded in memory.

        Measured at well under a millisecond for utterance-length audio - about 0.15 ms for
        eight seconds - which is noise beside a decode, but it is a real step and it is here
        rather than hidden in the timing.
        """
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(pcm.tobytes())
        return buffer.getvalue()

    def _run(self, audio: np.ndarray, is_final: bool) -> Transcript:
        started = time.perf_counter()
        og = self._og

        audios = og.Audios.open_bytes(self._to_wav(audio))
        inputs = self._processor(prompt=self._prompt, audios=audios)

        params = og.GeneratorParams(self._model)
        generator = og.Generator(self._model, params)
        generator.set_inputs(inputs)
        while not generator.is_done():
            generator.generate_next_token()

        text = self._processor.decode(generator.get_sequence(0))

        return Transcript(
            text=self._clean(text.strip()),
            duration_s=len(audio) / SAMPLE_RATE,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            is_final=is_final,
            # Neither is available from this engine. See the module docstring.
            clarity=None,
            words=[],
        )

    def _clean(self, text: str) -> str:
        """Drop known Whisper hallucinations that appear over near-silence.

        Shares the CTranslate2 path's rules rather than repeating them: they describe how
        Whisper behaves over silence, which is a property of the model and not of the runtime
        decoding it.
        """
        from .asr import _looks_like_caption_credit

        stripped = text.lower().strip()
        if stripped in self.settings.hallucinations:
            return ""
        if _looks_like_caption_credit(text):
            return ""
        return text

    def partial(self, audio: np.ndarray) -> Transcript:
        return self._run(audio, is_final=False)

    def final(self, audio: np.ndarray) -> Transcript:
        result = self._run(audio, is_final=True)
        self.add_context(result.text)
        return result

    def warmup(self) -> float:
        """Exercise the real decode path once, so first-use cost lands at startup.

        Non-silent audio through final(), matching the CTranslate2 engine: warming only on
        zeros would leave any lazy graph setup to surface mid-conversation, behind no loading
        indicator.
        """
        started = time.perf_counter()
        tone = (
            0.05 * np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE)
        ).astype(np.float32)
        self.final(tone)
        self.clear_context()
        return (time.perf_counter() - started) * 1000.0
