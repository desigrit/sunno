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

from .engine import IncrementalSpeechEngine, SpeechEngine
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
    audio: np.ndarray | None
    is_final: bool
    speaker_id: int | None = None
    speaker_label: str | None = None
    started_at: float = 0.0
    chunks: tuple[np.ndarray, ...] = ()
    generation: int = 0
    queued_at: float = 0.0
    reset_only: bool = False

    def samples(self) -> np.ndarray:
        if self.audio is not None:
            return self.audio
        return np.concatenate(self.chunks) if self.chunks else np.empty(0, np.float32)


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
        self._generation = 0
        self._reset_pending = False
        self._incremental = isinstance(engine, IncrementalSpeechEngine)
        self._stream_id: int | None = None
        self._stream_chunks = 0
        self._stream_samples = 0
        self._last_decode_ms = 0.0
        self._max_finals = 8
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
            if self._stop or self._finals:
                return
            job.generation = self._generation
            job.queued_at = time.monotonic()
            self._pending_partial = job  # latest wins
            self._lock.notify()

    def submit_final(self, job: _Job) -> None:
        with self._lock:
            generation = self._generation
            # Backpressure, never a deque(maxlen) that silently loses final captions.
            if len(self._finals) >= self._max_finals:
                self._emit({"type": "error", "code": "asr_backpressure",
                            "message": "Recognition is falling behind. Audio may be skipped; "
                                       "choose a lighter speech model."})
            while len(self._finals) >= self._max_finals and not self._stop:
                self._lock.wait(timeout=0.25)
                if generation != self._generation:
                    return
            if self._stop:
                return
            job.generation = generation
            job.queued_at = time.monotonic()
            self._finals.append(job)
            # A final supersedes any provisional decode for the same utterance.
            if self._pending_partial and self._pending_partial.utterance_id == job.utterance_id:
                self._pending_partial = None
            self._lock.notify()

    def flush(self) -> None:
        """Drop queued work so stale text can't surface after a pause/resume."""
        with self._lock:
            self._generation += 1
            self._pending_partial = None
            self._finals.clear()
            self._reset_pending = True
            self._lock.notify_all()

    def partial_interval_ms(self, configured: int) -> float:
        """Avoid scheduling refreshes faster than this engine can finish them."""
        with self._lock:
            return max(configured, min(2000.0, self._last_decode_ms * 1.25))

    def drain(self, timeout: float = 30.0) -> bool:
        """Block until queued work is done. Needed when a finite source (a WAV file)
        ends, so the process doesn't exit with transcriptions still in flight."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                idle = (not self._finals and self._pending_partial is None
                        and not self._busy and not self._reset_pending)
            if idle:
                return True
            time.sleep(0.05)
        return False

    def _next_job(self) -> _Job | None:
        with self._lock:
            self._busy = False
            while (not self._stop and not self._finals and self._pending_partial is None
                   and not self._reset_pending):
                self._lock.wait(timeout=0.25)
            if self._stop:
                return None
            self._busy = True
            if self._reset_pending:
                self._reset_pending = False
                return _Job(-1, None, False, generation=self._generation, reset_only=True)
            if self._finals:
                job = self._finals.popleft()
                self._lock.notify_all()
                return job
            job, self._pending_partial = self._pending_partial, None
            return job

    def _reset_stream(self) -> None:
        try:
            if self._incremental:
                self._engine.reset_stream()
        finally:
            self._stream_id = None
            self._stream_chunks = self._stream_samples = 0

    def _recover_stream(self) -> None:
        try:
            self._reset_stream()
        except Exception:
            # Cleanup can fail after a helper exits. Keep the worker alive so the
            # next job can recreate the engine's stream and replay its snapshot.
            pass

    def _decode(self, job: _Job):
        if not self._incremental:
            audio = job.samples()
            return self._engine.final(audio) if job.is_final else self._engine.partial(audio)
        if self._stream_id != job.utterance_id:
            self._reset_stream()
            self._engine.start_stream()
            self._stream_id = job.utterance_id
        if job.chunks:
            tail = job.chunks[self._stream_chunks:]
            audio = np.concatenate(tail) if tail else np.empty(0, np.float32)
            self._stream_chunks = len(job.chunks)
        else:
            # Compatibility for callers submitting a complete immutable snapshot.
            audio = job.samples()[self._stream_samples:]
        self._stream_samples += len(audio)
        result = self._engine.stream(audio, final=job.is_final)
        if job.is_final:
            self._reset_stream()
        return result

    def _loop(self) -> None:
        try:
            self._run_jobs()
        finally:
            self._recover_stream()
            close = getattr(self._engine, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass

    def _run_jobs(self) -> None:
        while True:
            job = self._next_job()
            if job is None:
                return
            if job.reset_only:
                self._recover_stream()
                continue
            started = time.monotonic()
            try:
                result = self._decode(job)
            except Exception as exc:  # keep the pipeline alive on a bad decode
                self._recover_stream()
                with self._lock:
                    if job.generation == self._generation:
                        self._emit({"type": "error", "message": f"ASR failure: {exc}"})
                continue

            with self._lock:
                self._last_decode_ms = (time.monotonic() - started) * 1000.0
                if job.generation != self._generation or self._stop:
                    continue

            if not result.text and not job.is_final:
                continue

            # Learn what this machine actually does, so the model picker can quote measured
            # figures instead of the ones recorded on the developer's hardware. Finals only:
            # partials decode greedily and would understate the wait for a finished sentence.
            if job.is_final and not self._incremental:
                from . import hardware

                hardware.record_latency(
                    self._engine.settings.model_size,
                    self._engine.settings.device,
                    result.latency_ms,
                )

            event = {
                    "type": "final" if job.is_final else "partial",
                    "id": job.utterance_id,
                    "text": result.text,
                    "speaker_id": job.speaker_id,
                    "speaker": job.speaker_label,
                    "clarity": result.clarity,
                    "latency_ms": round(result.latency_ms, 1),
                    "duration_s": round(result.duration_s, 2),
                    "queue_ms": round(max(0.0, started - job.queued_at) * 1000.0, 1),
                    # When the utterance was spoken, not when decoding finished, so a
                    # timestamp reflects the conversation rather than our queue depth.
                    "started_at": job.started_at,
                    "words": [
                        # Timings are omitted rather than sent as null when the engine has
                        # none, so the frame does not grow by two dead keys per word on the
                        # streaming path, where every word would carry them.
                        ({"t": w.text, "p": round(w.probability, 3)}
                         if w.start_s is None else
                         {"t": w.text, "p": round(w.probability, 3),
                          "s": round(w.start_s, 2), "e": round(w.end_s or w.start_s, 2)})
                        for w in result.words
                    ],
                }
            with self._lock:
                if job.generation == self._generation and not self._stop:
                    self._emit(event)


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
        on_audio: Callable[[np.ndarray], None] | None = None,
        speaker_labels=None,
    ) -> None:
        from .preprocess import StreamingAudioConditioner
        from .vad import StreamingSileroVAD

        self.settings = settings
        self._emit = emit
        self._should_run = should_run or (lambda: True)
        # Set once and read on every frame. A recording outlives any one capture session --
        # pausing, switching microphone and a transient stall all end run() and start it
        # again -- so the destination lives with the caller and this is only the tap.
        self._on_audio = on_audio
        self._vad = StreamingSileroVAD(FRAME_SAMPLES)
        self._speaker = speaker
        from .speaker_labels import LiveSpeakerLabels
        self._labels = speaker_labels or LiveSpeakerLabels(
            speaker, emit, settings.enable_speakers and settings.speaker_labels)
        self._owns_labels = speaker_labels is None
        self._condition = StreamingAudioConditioner(settings)
        self._worker = AsrWorker(engine, self._labels.emit_caption)
        self._worker_started = False

        pre_roll_frames = max(1, int(settings.pre_roll_ms / FRAME_MS))
        self._pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_frames)
        self._speaker_pre_roll: deque[np.ndarray | None] = deque(maxlen=pre_roll_frames)
        self._end_silence_frames = max(1, int(settings.end_silence_ms / FRAME_MS))
        self._max_utterance_frames = int(settings.max_utterance_s * 1000 / FRAME_MS)
        self._min_utterance_frames = max(1, int(settings.min_utterance_ms / FRAME_MS))
        self._min_partial_frames = max(1, int(settings.min_partial_ms / FRAME_MS))

        self._utterance_id = 0
        self._last_level_at = 0.0
        self._stop = threading.Event()
        self._reset_segmentation()

    def _reset_segmentation(self) -> None:
        self._speaking = False
        self._speech_run = 0
        self._silence_run = 0
        self._buffer = []
        self._last_partial_at = 0.0
        self._utterance_started_at = 0.0
        self._pre_roll.clear()
        self._speaker_pre_roll.clear()
        self._speaker_buffer: deque[np.ndarray] = deque(maxlen=int(8 * SAMPLE_RATE / FRAME_SAMPLES))
        self._speaker_frames = 0
        self._speaker_observed = 0
        self._speaker_generation = self._labels.generation
        self._vad.reset()
        self._condition.reset()

    def stop(self) -> None:
        self._stop.set()

    def drain(self, timeout: float = 30.0) -> bool:
        """Wait for queued transcriptions to complete (used when a WAV source ends)."""
        labels_done = self._labels.drain(timeout)
        return (self._worker.drain(timeout) if self._worker_started else True) and labels_done

    def close(self) -> None:
        """Shut down the ASR worker. Call once, at process exit."""
        if self._worker_started:
            self._worker.stop()
            self._worker_started = False
        if self._owns_labels:
            self._labels.close()

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
                    self._labels.discard(self._utterance_id)
                    self._emit({"type": "discard", "id": self._utterance_id})
                    self._worker.flush()
            if not self._should_run() or self._stop.is_set():
                self._worker.flush()
            self._reset_segmentation()

    def _process(self, frame: np.ndarray) -> None:
        # Ahead of everything else, so a recording gets a continuous timeline: every frame,
        # speech or silence, and the raw samples rather than the conditioned ones _finalise
        # hands the recogniser. What was recorded should be what the microphone heard.
        if self._on_audio is not None:
            try:
                self._on_audio(frame)
            except Exception:
                # A failing recorder must never take captions down with it.
                pass

        prob = self._vad(frame)
        now = time.monotonic()
        self._publish_level(frame, prob, now)
        conditioned = self._condition(frame)
        speaker_generation = self._labels.generation
        if speaker_generation != self._speaker_generation:
            self._speaker_generation = speaker_generation
            self._speaker_pre_roll.clear()
            self._speaker_buffer.clear()
            self._speaker_frames = self._speaker_observed = 0

        if not self._speaking:
            self._pre_roll.append(conditioned)
            self._speaker_pre_roll.append(conditioned if self._labels.enabled and
                                          prob >= self.settings.vad_end_threshold else None)
            if prob >= self.settings.vad_start_threshold:
                self._speech_run += 1
                if self._speech_run >= self.settings.start_frames:
                    self._begin_utterance()
            else:
                self._speech_run = 0
            return

        self._buffer.append(conditioned)
        if self._labels.enabled and prob >= self.settings.vad_end_threshold:
            self._speaker_buffer.append(conditioned)
            self._speaker_frames += 1
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
        self._speaker_buffer = deque((f for f in self._speaker_pre_roll if f is not None),
                                     maxlen=int(8 * SAMPLE_RATE / FRAME_SAMPLES))
        self._speaker_frames = len(self._speaker_buffer)
        self._speaker_observed = 0
        self._pre_roll.clear()
        self._speaker_pre_roll.clear()
        self._last_partial_at = 0.0
        self._labels.begin(self._utterance_id, self._utterance_started_at)
        self._emit({"type": "speech_start", "id": self._utterance_id})

    def _maybe_partial(self, now: float) -> None:
        if len(self._buffer) < self._min_partial_frames:
            return
        if ((now - self._last_partial_at) * 1000.0
                < self._worker.partial_interval_ms(self.settings.partial_interval_ms)):
            return
        self._last_partial_at = now
        self._labels.begin(self._utterance_id, self._utterance_started_at)
        if self._speaker_frames - self._speaker_observed >= 40:
            self._speaker_observed = self._speaker_frames
            self._labels.observe(self._utterance_id, tuple(self._speaker_buffer), final=False,
                                 expected_generation=self._speaker_generation)
        self._worker.submit_partial(
            _Job(self._utterance_id, None, is_final=False, chunks=tuple(self._buffer),
                 started_at=self._utterance_started_at)
        )

    def _finalise(self, continued: bool = False) -> None:
        buffer, self._buffer = self._buffer, []
        self._speaking = False
        self._silence_run = 0
        self._speech_run = 0
        self._pre_roll.clear()

        if len(buffer) >= self._min_utterance_frames:
            self._labels.begin(self._utterance_id, self._utterance_started_at)
            self._labels.observe(self._utterance_id, tuple(self._speaker_buffer), final=True,
                                 expected_generation=self._speaker_generation)
            self._worker.submit_final(
                _Job(self._utterance_id, None, is_final=True, chunks=tuple(buffer),
                     started_at=self._utterance_started_at)
            )
        else:
            self._labels.discard(self._utterance_id)
            self._emit({"type": "discard", "id": self._utterance_id})

        if continued:
            # Long monologue: immediately reopen so speech isn't dropped mid-sentence.
            self._vad.reset()
            self._begin_utterance()

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
