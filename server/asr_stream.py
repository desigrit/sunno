"""Streaming transducer engine, for machines with no usable graphics card.

Most PCs have no NVIDIA card, and on those Whisper is slow because of what it is rather
than how it is run: an encoder-decoder that pads every window to thirty seconds, so a
two second sentence costs about what a twenty second one does. A transducer does not
pad. Measured here on a processor at four threads, averaged over 2, 4 and 8 second
utterances, a streaming Zipformer decodes in about 135 ms and Kroko in about 120, where
Whisper base takes 730 and large-v3 takes 4540. That is the reason this file exists.

It runs on sherpa-onnx, which the app already ships and uses for speaker embeddings, so
this adds a model and no new native dependency. sherpa-onnx also publishes native
win_arm64 wheels, which CTranslate2 does not, so this is the first engine that can run
everywhere without emulation.

**Used statelessly, on purpose.** sherpa-onnx can hold state across chunks and emit words
while someone is still speaking, which is a better design for captions than waiting for a
sentence to end. This engine deliberately does not do that yet. The pipeline hands an
engine the whole utterance so far on every partial, and ``AudioConditioner`` renormalises
gain over everything it is given, so the prefix of a partial is not the same audio it was
last time: measured, one loud word changed already-decoded samples by 67 percent. Feeding
tails into a recogniser holding state would splice two different gains together, quietly,
with nothing to catch it. So each call builds a fresh stream over the whole utterance,
exactly as the CTranslate2 and ONNX engines already do. That is correct, and it is still
several times faster than what it replaces. True streaming needs the pipeline to hand out
deltas and to say which utterance they belong to, which is a change to a seam every engine
shares and is not this file's to make.

Two things this cannot report, both accepted deliberately: ``clarity``, the per-utterance
confidence badge, and per-word uncertainty. A transducer exposes no comparable score, and
inventing one would be worse than leaving it out for someone relying on this to follow a
conversation. The UI already treats both as optional.
"""

from __future__ import annotations

import time

import numpy as np

from .config import SAMPLE_RATE, Settings
from .engine import Transcript


class StreamingEngine:
    """A streaming transducer, used one utterance at a time."""

    def __init__(self, settings: Settings) -> None:
        import sherpa_onnx

        from .models import stream_model_is_cased, stream_model_paths

        self.settings = settings
        self._sherpa = sherpa_onnx
        self._cased = stream_model_is_cased(settings.model_size)

        paths = stream_model_paths(settings.model_size)
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(paths["tokens"]),
            encoder=str(paths["encoder"]),
            decoder=str(paths["decoder"]),
            joiner=str(paths["joiner"]),
            num_threads=_threads(),
            provider="cpu",
            decoding_method="greedy_search",
        )

    # --- context -------------------------------------------------------
    # A transducer takes no prompt, so previous text cannot bias it and a custom vocabulary
    # has nowhere to go. Both are no-ops here, and both are called by this engine's own
    # decode paths in the other implementations rather than by the pipeline, so they exist
    # to keep the three engines interchangeable.
    def add_context(self, text: str) -> None:
        return None

    def clear_context(self) -> None:
        return None

    # --- decoding ------------------------------------------------------
    def _run(self, audio: np.ndarray, is_final: bool) -> Transcript:
        started = time.perf_counter()

        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        stream.input_finished()
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
        text = self._recognizer.get_result(stream).strip()

        return Transcript(
            text=self._readable(text),
            duration_s=len(audio) / SAMPLE_RATE,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            is_final=is_final,
            # See the module docstring. Reported as absent rather than invented.
            clarity=None,
            words=[],
        )

    def _readable(self, text: str) -> str:
        """Lower-case, for the models that emit unbroken upper case.

        A whole conversation in capitals is legible but tiring, and this app exists to be
        read for the length of a conversation. Models that write their own capitals and
        punctuation are left alone: lower-casing Kroko would discard the thing that makes
        it worth choosing.

        Restoring sentence case on a model that has none needs a second model, and the one
        that fits has no clear licence and no clean download path, so it is left out. See
        the note in models.py. Doing it here without a model is not an option worth taking:
        capitalising after every full stop requires knowing where the full stops are, which
        is the thing that is missing.

        Applied identically to provisional and final text, on purpose, in both directions.
        Anything that depends on later words rewrites text already on screen: an earlier
        version ran a punctuation model over each growing partial and the opening words
        changed on 13 of 31 refreshes, cycling between "lazy dog. We", "lazy dog. we" and
        "lazy dog we". Neither lower-casing nor passing text through unchanged can do that,
        because neither looks beyond the character in front of it.

        That is a claim about this function, not about the captions. The transducer itself
        revises as more audio arrives, and on real two-speaker audio it does so often: over
        testdata/ at the pipeline's own 700 ms and 450 ms cadence, stream-en changed a
        prefix already displayed on 147 of 251 refreshes and Kroko on 44 of the same 251,
        measured by bench/bench_stream_churn.py so the figures can be re-run rather than
        believed. An earlier version of this note cited "0 of 31" as though that were a
        property of the engine; it was a property of one clean synthesized clip and it does
        not survive real speech. The cause is structural rather than fixable here: the
        pipeline re-decodes the whole utterance on every partial, so the model is free to
        reach a different answer each time. Handing out deltas with a committed prefix is
        what would stop it, which is the same pipeline change described above.
        """
        return text if self._cased else text.lower()

    def partial(self, audio: np.ndarray) -> Transcript:
        return self._run(audio, is_final=False)

    def final(self, audio: np.ndarray) -> Transcript:
        return self._run(audio, is_final=True)

    def warmup(self) -> float:
        """One real decode, so the first thing anyone says does not pay the load cost."""
        started = time.perf_counter()
        self._run(np.zeros(SAMPLE_RATE, dtype=np.float32), is_final=True)
        return (time.perf_counter() - started) * 1000.0


def _threads() -> int:
    """Threads for the recogniser.

    Capped low on purpose, and the cap is not a guess. These models are measurably worse
    with more threads, because the work per chunk is too small to spread and the
    coordination costs more than it returns. So this is not "four is enough", it is "four is
    better", on top of not taking cores the rest of the machine is using.

    hardware.py depends on this cap: it files one figure for these models in both the four
    and sixteen thread tables, on the grounds that the sixteen thread column is unreachable.
    Raising this number without remeasuring those rows makes the picker quote a latency the
    app does not deliver, so tests/test_stream_engine.py asserts the two stay in step.
    """
    import os

    return max(1, min(4, os.cpu_count() or 4))
