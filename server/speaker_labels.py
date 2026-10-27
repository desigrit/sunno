"""Best-effort, revisable speaker labels, independent of the caption decoder.

Only voice vectors and a bounded set of recent assignments survive processing. Audio
snapshots are bounded, stay in memory and are dropped after inference or when disabled.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import SAMPLE_RATE


@dataclass
class Assignment:
    speaker_id: int | None
    started_at: float
    embedding: np.ndarray | None = None
    final: bool = False


class LiveSpeakerLabels:
    def __init__(self, speaker, emit: Callable[[dict], None], enabled: bool = True):
        self.speaker = speaker
        self.session = uuid.uuid4().hex
        self._emit = emit
        self._enabled = enabled and speaker is not None
        self._condition = threading.Condition(threading.RLock())
        self._generation = 0
        self._closed = False
        self._busy = False
        self._pending: OrderedDict[int, tuple] = OrderedDict()
        self._assignments: OrderedDict[int, Assignment] = OrderedDict()
        self._last_speaker: int | None = None
        self._roster_signature = None
        self._thread = threading.Thread(target=self._run, name="speaker-labels", daemon=True)

    @property
    def enabled(self) -> bool:
        with self._condition:
            return self._enabled

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    def state(self) -> dict:
        return {"type": "speaker_labels", "mode": "automatic" if self._enabled else "off",
                "available": self.speaker is not None, "session": self.session}

    def set_enabled(self, enabled: bool) -> None:
        with self._condition:
            enabled = enabled and self.speaker is not None
            if enabled != self._enabled:
                self._enabled = enabled
                self._generation += 1
                self._pending.clear()
                # Retain displayed assignments, but never retrospectively identify audio
                # heard while Off. A snapshot from the old mode cannot update a new one.
                for assignment in self._assignments.values():
                    assignment.embedding = None
                self._condition.notify_all()
            self._emit(self.state())
            self._publish_roster()

    def begin(self, utterance_id: int, started_at: float) -> None:
        with self._condition:
            if not self._enabled or utterance_id in self._assignments:
                return
            reserve = getattr(self.speaker, "provisional", None)
            sid = reserve(self._last_speaker) if reserve else self._last_speaker
            self._assignments[utterance_id] = Assignment(sid, started_at)
            while len(self._assignments) > 256:
                self._assignments.popitem(last=False)
            self._last_speaker = sid
            self._publish_roster()

    def observe(self, utterance_id: int, chunks: tuple[np.ndarray, ...], final: bool,
                expected_generation: int | None = None) -> None:
        with self._condition:
            if not self._enabled or self._closed or utterance_id not in self._assignments:
                return
            if expected_generation is not None and expected_generation != self._generation:
                return
            # Latest snapshot wins. Eight bounded utterances cannot build an unbounded
            # inference backlog on a slow CPU, and speaker work never stalls captions.
            self._pending[utterance_id] = (chunks, final, self._generation)
            while len(self._pending) > 8:
                self._pending.popitem(last=False)
            if self._thread.ident is None:
                self._thread.start()
            self._condition.notify_all()

    def discard(self, utterance_id: int) -> None:
        with self._condition:
            self._pending.pop(utterance_id, None)
            self._assignments.pop(utterance_id, None)

    def emit_caption(self, event: dict) -> None:
        with self._condition:
            if event.get("type") in ("partial", "final"):
                assignment = self._assignments.get(event["id"])
                sid = assignment.speaker_id if self._enabled and assignment else None
                event = {**event, "speaker_id": sid,
                         "speaker": self.speaker.label(sid) if sid is not None else None,
                         "speaker_session": self.session}
            # Ordering is shared with correction events: a stale ASR job can never put
            # its original guess back after a correction has already been published.
            self._emit(event)

    def _publish_roster(self) -> None:
        roster = self.speaker.roster() if self._enabled else []
        signature = tuple((r["id"], r["label"], r["is_self"]) for r in roster)
        if signature != self._roster_signature:
            self._roster_signature = signature
            self._emit({"type": "roster", "speakers": roster})

    def _correct(self, utterance_id: int, assignment: Assignment, sid: int | None) -> None:
        if assignment.speaker_id == sid:
            return
        assignment.speaker_id = sid
        self._emit({"type": "speaker_label", "id": utterance_id,
                    "started_at": assignment.started_at, "speaker_id": sid,
                    "speaker": self.speaker.label(sid)})

    def merge(self, source: int, target: int, automatic: bool = False) -> bool:
        with self._condition:
            if self.speaker is None or not self.speaker.merge(source, target):
                return False
            if not automatic:
                self._generation += 1
                self._pending.clear()
            for assignment in self._assignments.values():
                if assignment.speaker_id == source:
                    assignment.speaker_id = target
            if self._last_speaker == source:
                self._last_speaker = target
            self._emit({"type": "speaker_merged", "from": source, "into": target})
            self._publish_roster()
            return True

    def edit(self, action: str, speaker_id: int = -1, value=None) -> None:
        with self._condition:
            if self.speaker is None:
                return
            self._generation += 1
            self._pending.clear()
            if action == "rename":
                self.speaker.rename(speaker_id, value)
            elif action == "self":
                self.speaker.set_self(speaker_id, value)
            elif action == "delete":
                label = self.speaker.delete(speaker_id)
                if label is not None:
                    self._emit({"type": "speaker_deleted", "id": speaker_id, "label": label})
            elif action == "reset":
                self.speaker.reset()
            self._publish_roster()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                utterance_id, (chunks, final, generation) = self._pending.popitem(last=False)
                self._busy = True
            try:
                audio = np.concatenate(chunks) if chunks else np.empty(0, np.float32)
                # Extraction happens outside the state lock, so Off applies immediately
                # even if a native model call is still completing in this thread.
                embedding = self.speaker.embed(audio) if hasattr(self.speaker, "assign") else None
                with self._condition:
                    if generation != self._generation or not self._enabled or self._closed:
                        continue
                    assignment = self._assignments.get(utterance_id)
                    if assignment is None:
                        continue
                    if hasattr(self.speaker, "assign"):
                        if embedding is None:
                            continue
                        sid, _score = self.speaker.assign(embedding, len(audio) / SAMPLE_RATE,
                                                         assignment.speaker_id, learn=final)
                    else:
                        # Small injected identifiers are also useful to pipeline callers.
                        sid, _score = self.speaker.identify(audio)
                    if (sid is not None and sid != assignment.speaker_id and
                            hasattr(self.speaker, "is_placeholder") and
                            self.speaker.is_placeholder(assignment.speaker_id)):
                        self.merge(assignment.speaker_id, sid, automatic=True)
                    self._publish_roster()
                    self._correct(utterance_id, assignment, sid)
                    assignment.final = final
                    assignment.embedding = embedding if final else None
                    if utterance_id == next(reversed(self._assignments)):
                        self._last_speaker = sid
                    if final and embedding is not None:
                        # Review only recent completed utterances. No audio is retained
                        # and the cost is bounded independently of session length.
                        recent = list(self._assignments.items())[-128:]
                        for old_id, old in recent:
                            if old.final and old.embedding is not None and old.speaker_id is not None:
                                revised = self.speaker.reconsider(old.embedding, old.speaker_id)
                                self._correct(old_id, old, revised)
                        duplicate = self.speaker.duplicate()
                        if duplicate is not None:
                            self.merge(*duplicate, automatic=True)
                        self._last_speaker = next(reversed(self._assignments.values())).speaker_id
                        self._publish_roster()
            except Exception:
                # Speaker labels are optional. One failed embedding cannot turn a
                # functioning speech engine into an error screen or halt transcription.
                pass
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()

    def drain(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._pending or self._busy:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._condition.wait(left)
            return True

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._condition.notify_all()
        if self._thread.ident is not None:
            self._thread.join(timeout=5)
