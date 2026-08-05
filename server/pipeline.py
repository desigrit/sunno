"""Live captioning pipeline: VAD endpointing plus a two-pass ASR worker.

Audio flows: mic frames -> VAD state machine -> utterance buffer -> ASR worker.

The worker keeps a single-slot queue for provisional jobs so that a newer snapshot of the
current utterance always supersedes a stale one; final jobs are queued and take priority.
This keeps displayed text close to live even when a provisional decode is slower than the
partial interval.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .engine import SpeechEngine
from .config import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, Settings

Emit = Callable[[dict], None]


class SessionController:
    """Run/pause state shared between the WebSocket handler and the capture thread.

    Pausing releases the microphone entirely rather than discarding frames, so Windows'
    microphone-in-use indicator switches off and people in the room can see that capture has
    actually stopped. The Whisper model stays loaded, so resuming is immediate.
    """

    def __init__(self, running: bool = True) -> None:
        self._running = threading.Event()
        self._shutdown = threading.Event()
        if running:
            self._running.set()

    @property
    def is_running(self) -> bool:
        return self._running.is_set() and not self._shutdown.is_set()

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown.is_set()

    def start(self) -> None:
        self._running.set()

    def pause(self) -> None:
        self._running.clear()

    def toggle(self) -> bool:
        if self.is_running:
            self.pause()
        else:
            self.start()
        return self.is_running

    def shutdown(self) -> None:
        self._shutdown.set()
        self._running.set()  # unblock anyone waiting to start

    def wait_for_start(self, timeout: float | None = None) -> bool:
        return self._running.wait(timeout)


@dataclass
class _Job:
    utterance_id: int
    audio: np.ndarray
    is_final: bool
    speaker_id: int | None = None
    speaker_label: str | None = None
    started_at: float = 0.0


class AsrWorker:
    """Serialises GPU work, preferring finals and collapsing superseded partials."""

    def __init__(self, engine: SpeechEngine, emit: Emit) -> None:
        self._engine = engine
        self._emit = emit
        self._lock = threading.Condition()
        self._pending_partial: _Job | None = None
        self._finals: deque[_Job] = deque()
        self._busy = False
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="asr", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop = True
            self._lock.notify_all()
        self._thread.join(timeout=5)

    def submit_partial(self, job: _Job) -> None:
        with self._lock:
            self._pending_partial = job  # latest wins
            self._lock.notify()

    def submit_final(self, job: _Job) -> None:
        with self._lock:
            self._finals.append(job)
            # A final supersedes any provisional decode for the same utterance.
            if self._pending_partial and self._pending_partial.utterance_id == job.utterance_id:
                self._pending_partial = None
            self._lock.notify()

    def flush(self) -> None:
        """Drop queued work so stale text can't surface after a pause/resume."""
        with self._lock:
            self._pending_partial = None
            self._finals.clear()

    def drain(self, timeout: float = 30.0) -> bool:
        """Block until queued work is done. Needed when a finite source (a WAV file)
        ends, so the process doesn't exit with transcriptions still in flight."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                idle = not self._finals and self._pending_partial is None and not self._busy
            if idle:
                return True
            time.sleep(0.05)
        return False

    def _next_job(self) -> _Job | None:
        with self._lock:
            self._busy = False
            while not self._stop and not self._finals and self._pending_partial is None:
                self._lock.wait(timeout=0.25)
            if self._stop:
                return None
            self._busy = True
            if self._finals:
                return self._finals.popleft()
            job, self._pending_partial = self._pending_partial, None
            return job

    def _loop(self) -> None:
        while True:
            job = self._next_job()
            if job is None:
                return
            try:
                result = (
                    self._engine.final(job.audio)
                    if job.is_final
                    else self._engine.partial(job.audio)
                )
            except Exception as exc:  # keep the pipeline alive on a bad decode
                self._emit({"type": "error", "message": f"ASR failure: {exc}"})
                continue

            if not result.text and not job.is_final:
                continue

            # Learn what this machine actually does, so the model picker can quote measured
            # figures instead of the ones recorded on the developer's hardware. Finals only:
            # partials decode greedily and would understate the wait for a finished sentence.
            if job.is_final:
                from . import hardware

                hardware.record_latency(
                    self._engine.settings.model_size,
                    self._engine.settings.device,
                    result.latency_ms,
                )

            self._emit(
                {
                    "type": "final" if job.is_final else "partial",
                    "id": job.utterance_id,
                    "text": result.text,
                    "speaker_id": job.speaker_id,
                    "speaker": job.speaker_label,
                    "clarity": result.clarity,
                    "latency_ms": round(result.latency_ms, 1),
                    "duration_s": round(result.duration_s, 2),
                    # When the utterance was spoken, not when decoding finished, so a
                    # timestamp reflects the conversation rather than our queue depth.
                    "started_at": job.started_at,
                    "words": [
                        {"t": w.text, "p": round(w.probability, 3)} for w in result.words
                    ],
                }
            )


class CaptionPipeline:
    """VAD-driven segmentation feeding the two-pass ASR worker.

    ``run()`` may be called repeatedly — once per capture session — as the user stops and
    starts transcription. The ASR worker thread outlives those sessions so the model stays
    resident; only ``close()`` tears it down.
    """

    def __init__(
        self,
        settings: Settings,
        engine: SpeechEngine,
        emit: Emit,
        should_run: Callable[[], bool] | None = None,
        speaker: "SpeakerIdentifier | None" = None,
    ) -> None:
        from .preprocess import AudioConditioner
        from .vad import StreamingSileroVAD

        self.settings = settings
        self._emit = emit
        self._should_run = should_run or (lambda: True)
        self._vad = StreamingSileroVAD(FRAME_SAMPLES)
        self._speaker = speaker
        self._condition = AudioConditioner(settings)
        self._worker = AsrWorker(engine, emit)
        self._worker_started = False

        pre_roll_frames = max(1, int(settings.pre_roll_ms / FRAME_MS))
        self._pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_frames)
        self._end_silence_frames = max(1, int(settings.end_silence_ms / FRAME_MS))
        self._max_utterance_frames = int(settings.max_utterance_s * 1000 / FRAME_MS)
        self._min_utterance_frames = max(1, int(settings.min_utterance_ms / FRAME_MS))
        self._min_partial_frames = max(1, int(settings.min_partial_ms / FRAME_MS))

        self._utterance_id = 0
        self._last_level_at = 0.0
        self._roster_signature: tuple = ()
        self._stop = threading.Event()
        self._reset_segmentation()

    def _reset_segmentation(self) -> None:
        self._speaking = False
        self._speech_run = 0
        self._silence_run = 0
        self._buffer = []
        self._last_partial_at = 0.0
        self._current_speaker = None
        self._current_label = None
        self._utterance_started_at = 0.0
        self._pre_roll.clear()
        self._vad.reset()

    def stop(self) -> None:
        self._stop.set()

    def drain(self, timeout: float = 30.0) -> bool:
        """Wait for queued transcriptions to complete (used when a WAV source ends)."""
        return self._worker.drain(timeout) if self._worker_started else True

    def close(self) -> None:
        """Shut down the ASR worker. Call once, at process exit."""
        if self._worker_started:
            self._worker.stop()
            self._worker_started = False

    def run(self, frames: Iterable[np.ndarray]) -> None:
        """Consume frames until the source ends, stop() is called, or should_run() goes false."""
        if not self._worker_started:
            self._worker.start()
            self._worker_started = True

        self._reset_segmentation()
        try:
            for frame in frames:
                if self._stop.is_set() or not self._should_run():
                    break
                self._process(frame)
        finally:
            if self._speaking:
                # Commit whatever was mid-utterance rather than losing it, unless the user
                # explicitly stopped - in that case drop it, since they asked us to stop.
                if self._should_run() and not self._stop.is_set():
                    self._finalise()
                else:
                    self._emit({"type": "discard", "id": self._utterance_id})
                    self._worker.flush()
            self._reset_segmentation()

    def _process(self, frame: np.ndarray) -> None:
        prob = self._vad(frame)
        now = time.monotonic()
        self._publish_level(frame, prob, now)

        if not self._speaking:
            self._pre_roll.append(frame)
            if prob >= self.settings.vad_start_threshold:
                self._speech_run += 1
                if self._speech_run >= self.settings.start_frames:
                    self._begin_utterance()
            else:
                self._speech_run = 0
            return

        self._buffer.append(frame)
        if prob < self.settings.vad_end_threshold:
            self._silence_run += 1
        else:
            self._silence_run = 0

        if self._silence_run >= self._end_silence_frames:
            self._finalise()
            return

        if len(self._buffer) >= self._max_utterance_frames:
            # Bound provisional latency on long monologues by committing early.
            self._finalise(continued=True)
            return

        self._maybe_partial(now)

    def _begin_utterance(self) -> None:
        self._speaking = True
        self._speech_run = 0
        self._silence_run = 0
        self._utterance_id += 1
        self._utterance_started_at = time.time()
        self._buffer = list(self._pre_roll)  # keep word onsets
        self._pre_roll.clear()
        self._last_partial_at = 0.0
        self._emit({"type": "speech_start", "id": self._utterance_id})

    def _maybe_partial(self, now: float) -> None:
        if len(self._buffer) < self._min_partial_frames:
            return
        if (now - self._last_partial_at) * 1000.0 < self.settings.partial_interval_ms:
            return
        self._last_partial_at = now
        audio = self._condition(np.concatenate(self._buffer))
        self._worker.submit_partial(
            _Job(self._utterance_id, audio, is_final=False,
                 speaker_id=self._current_speaker, speaker_label=self._current_label,
                 started_at=self._utterance_started_at)
        )

    def _finalise(self, continued: bool = False) -> None:
        buffer, self._buffer = self._buffer, []
        self._speaking = False
        self._silence_run = 0
        self._speech_run = 0
        self._pre_roll.clear()

        if len(buffer) >= self._min_utterance_frames:
            audio = self._condition(np.concatenate(buffer))
            speaker_id, label = self._resolve_speaker(audio)
            self._worker.submit_final(
                _Job(self._utterance_id, audio, is_final=True,
                     speaker_id=speaker_id, speaker_label=label,
                     started_at=self._utterance_started_at)
            )
        else:
            self._emit({"type": "discard", "id": self._utterance_id})

        self._current_speaker = None
        self._current_label = None

        if continued:
            # Long monologue: immediately reopen so speech isn't dropped mid-sentence.
            self._vad.reset()
            self._begin_utterance()

    def _resolve_speaker(self, audio: np.ndarray) -> tuple[int | None, str | None]:
        """Identify the speaker for a completed utterance.

        Runs before the Whisper call and costs ~10-35 ms against Whisper's ~350 ms, so it
        adds no meaningful latency and the label is available the moment the text is.
        """
        if self._speaker is None:
            return None, None
        try:
            speaker_id, _score = self._speaker.identify(audio)
        except Exception as exc:
            self._emit({"type": "error", "message": f"speaker id failed: {exc}"})
            return None, None

        # identify() can enrol someone new, and nothing else announces that. Without this the
        # transcript shows "Speaker 2" while the Speakers pane still reads as empty.
        self._publish_roster_if_changed()

        if speaker_id is None:
            return None, None
        self._current_speaker = speaker_id
        self._current_label = self._speaker.label(speaker_id)
        return speaker_id, self._current_label

    def _publish_roster_if_changed(self) -> None:
        if self._speaker is None:
            return
        roster = self._speaker.roster()
        signature = tuple((r["id"], r["label"], r["is_self"]) for r in roster)
        if signature == self._roster_signature:
            return
        self._roster_signature = signature
        self._emit({"type": "roster", "speakers": roster})

    def _publish_level(self, frame: np.ndarray, prob: float, now: float) -> None:
        if now - self._last_level_at < 0.1:  # ~10 Hz is plenty for a meter
            return
        self._last_level_at = now
        rms = float(np.sqrt(np.mean(np.square(frame))))
        self._emit(
            {
                "type": "level",
                "rms": round(rms, 5),
                "db": round(20 * np.log10(rms + 1e-9), 1),
                "speech_prob": round(prob, 3),
                "speaking": self._speaking,
            }
        )
