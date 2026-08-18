"""Online speaker labelling.

Each finalised utterance is turned into a speaker embedding (WeSpeaker CAM++, 512-dim,
~10-35 ms on CPU) and matched against the speakers seen so far by cosine similarity. No
advance knowledge of how many people are present is needed.

Measured behaviour of these embeddings on real audio:

    identical audio                     1.00
    overlapping windows, same speaker   0.82
    two halves of one recording         0.57
    different speakers (M vs F)         0.34
    short (<2 s) segments, same speaker 0.33   <-- unreliable

The last line matters most: speaker embeddings need roughly 2-3 s of speech to be
dependable, and conversational turns are often shorter than that. Two safeguards follow:

  * `min_identify_s`    - below this, return None rather than guess.
  * `min_new_speaker_s` - creating a *new* speaker requires more evidence than matching an
    existing one, so brief utterances cannot fragment the roster.

Automatic labelling is therefore best-effort. Naming a speaker once (see `rename`) pins
their profile and makes subsequent matching markedly more reliable, because comparison then
happens against a deliberately-chosen reference instead of a noisy first guess.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE

DEFAULT_MODEL = "speaker-embedding-campplus-en.onnx"


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class _Profile:
    """Running centroid for one speaker."""

    __slots__ = ("id", "centroid", "count", "name", "pinned", "is_self")

    def __init__(self, profile_id: int, embedding: np.ndarray, name: str | None = None,
                 pinned: bool = False, is_self: bool = False) -> None:
        self.id = profile_id
        self.centroid = embedding
        self.count = 1
        self.name = name
        self.pinned = pinned
        self.is_self = is_self

    def update(self, embedding: np.ndarray, max_weight: int = 20) -> None:
        # Pinned profiles are deliberately chosen references; don't let noisy far-field
        # utterances drift them.
        if self.pinned:
            return
        # Cap the weight so the centroid keeps adapting to the speaker's current acoustics
        # (distance from mic, volume) rather than freezing on the first sample.
        weight = min(self.count, max_weight)
        self.centroid = _l2_normalise((self.centroid * weight + embedding) / (weight + 1))
        self.count += 1


class SpeakerIdentifier:
    """Assigns stable speaker ids to utterances, discovering speakers as they appear.

    Ids are handed out from a counter and never reused within a session. They are
    deliberately NOT positions in a list: merging one speaker into another used to delete
    from the middle of a list and renumber everybody above, while the UI relabels existing
    captions by matching on id. The visible effect was that folding two speakers together
    silently re-attributed other people's already-scrolled lines to the wrong person - in a
    transcript that is, for a deaf user, the only record of who said what.

    Profiles are held in a dict keyed by id rather than a list, so indexing by position is
    not merely discouraged but unavailable.

    Ids are not persisted. Nothing refers to one across a restart: the transcript is
    session-only, and saved speakers are re-matched by centroid, not by id. Loading assigns
    fresh compact ids in file order, which keeps the numbers small and needs no migration
    for profiles written before ids existed.
    """

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.50,
        max_speakers: int = 8,
        min_identify_s: float = 1.0,
        min_new_speaker_s: float = 2.0,
        num_threads: int = 4,
        profile_path: str | Path | None = None,
    ) -> None:
        import sherpa_onnx

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"speaker embedding model not found: {model_path}")

        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(model_path), num_threads=num_threads
            )
        )
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.min_identify_samples = int(min_identify_s * SAMPLE_RATE)
        self.min_new_speaker_samples = int(min_new_speaker_s * SAMPLE_RATE)
        self.profile_path = Path(profile_path) if profile_path else None
        self._profiles: dict[int, _Profile] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        if self.profile_path:
            self.load()

    def _mint(self, embedding: np.ndarray, name: str | None = None,
              pinned: bool = False, is_self: bool = False) -> _Profile:
        """Create a profile under a fresh id. Caller holds the lock."""
        profile = _Profile(self._next_id, embedding, name=name, pinned=pinned, is_self=is_self)
        self._profiles[profile.id] = profile
        self._next_id += 1
        return profile

    @property
    def num_speakers(self) -> int:
        return len(self._profiles)

    def label(self, speaker_id: int | None) -> str | None:
        profile = self._profiles.get(speaker_id) if speaker_id is not None else None
        if profile is None:
            return None
        return profile.name or f"Speaker {profile.id + 1}"

    def roster(self) -> list[dict]:
        # Sorted by id so the UI, which inserts new rows at the roster's own index, still
        # sees people in the order they were first heard even once ids have gaps in them.
        with self._lock:
            return [
                {"id": p.id, "label": p.name or f"Speaker {p.id + 1}",
                 "named": p.name is not None, "is_self": p.is_self}
                for p in sorted(self._profiles.values(), key=lambda p: p.id)
            ]

    def reset(self) -> None:
        """Forget discovered speakers, keeping any the user has named."""
        with self._lock:
            self._profiles = {i: p for i, p in self._profiles.items() if p.pinned}

    def rename(self, speaker_id: int, name: str) -> bool:
        """Name a speaker and pin their profile so it stops drifting."""
        with self._lock:
            profile = self._profiles.get(speaker_id)
            if profile is None:
                return False
            cleaned = name.strip()
            profile.name = cleaned or None
            profile.pinned = bool(cleaned)
        self.save()
        return True

    def set_self(self, speaker_id: int, is_self: bool = True) -> bool:
        """Mark a speaker as the user themselves.

        Their speech is still transcribed - the user reads it back as clarity feedback for
        speech practice - but it is rendered distinctly, so their own lines never get
        confused with what other people said. On a Whisper model it also carries a clarity
        score; the streaming engines produce none, which is why the setting and both speaker
        dialogs name Whisper.
        """
        with self._lock:
            if speaker_id not in self._profiles:
                return False
            for profile in self._profiles.values():
                # Only one speaker can be "you".
                profile.is_self = is_self and profile.id == speaker_id
            if is_self:
                # Pin it: this profile must stay stable or the marking will drift off.
                self._profiles[speaker_id].pinned = True
        self.save()
        return True

    def is_self_speaker(self, speaker_id: int | None) -> bool:
        if speaker_id is None:
            return False
        with self._lock:
            profile = self._profiles.get(speaker_id)
            return profile is not None and profile.is_self

    def delete(self, speaker_id: int) -> str | None:
        """Forget one speaker entirely. Returns the label their old captions should fall
        back to, or None if there was no such speaker.

        The returned label is the generic one derived from the retired id, never a name.
        That is safe precisely because ids are never reused: no live speaker can ever be
        shown under this label, so captions from the deleted person cannot be confused with
        anybody else's. Their lines keep saying "someone said this" without claiming who.
        """
        with self._lock:
            profile = self._profiles.pop(speaker_id, None)
            if profile is None:
                return None
            fallback = f"Speaker {profile.id + 1}"
        self.save()
        return fallback

    def merge(self, source_id: int, target_id: int) -> bool:
        """Fold one speaker into another, for when discovery split one person in two."""
        with self._lock:
            target = self._profiles.get(target_id)
            source = self._profiles.get(source_id)
            if target is None or source is None or source_id == target_id:
                return False
            total = target.count + source.count
            target.centroid = _l2_normalise(
                (target.centroid * target.count + source.centroid * source.count) / total
            )
            target.count = total
            # Everything the user asserted about either profile has to survive, because the
            # premise of a merge is that these were always one person.
            #
            # Pinning matters as much as the name: save() only writes pinned profiles, so a
            # named-but-unpinned survivor is reported as named in the roster and then silently
            # forgotten on the next launch. That happens whenever a named speaker is folded
            # into a discovered one, which is the ordinary direction for "this stranger is
            # actually Priya". rename() and set_self() both keep named/self profiles pinned;
            # merge used to copy the name across without it.
            target.name = target.name or source.name
            target.is_self = target.is_self or source.is_self
            if target.name is not None or target.is_self:
                target.pinned = True
            # The source id is retired, not recycled. Every other speaker keeps the id their
            # existing captions were labelled with; only the merged-away person's lines need
            # moving, and app.py announces that mapping so the UI can move them deliberately
            # rather than inferring it from a renumbering.
            del self._profiles[source_id]
        self.save()
        return True

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        if len(audio) < self.min_identify_samples:
            return None
        stream = self._extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            return None
        return _l2_normalise(np.asarray(self._extractor.compute(stream), dtype=np.float32))

    def identify(self, audio: np.ndarray) -> tuple[int | None, float]:
        """Return (speaker_id, similarity). None when the clip is too short to judge."""
        embedding = self.embed(audio)
        if embedding is None:
            return None, 0.0

        with self._lock:
            if not self._profiles:
                if len(audio) < self.min_new_speaker_samples:
                    return None, 0.0
                return self._mint(embedding).id, 1.0

            # Compared in a fixed order so `best` indexes back into the same sequence, but
            # what leaves this method is the winner's durable id, never its position.
            candidates = sorted(self._profiles.values(), key=lambda p: p.id)
            scores = np.array([float(p.centroid @ embedding) for p in candidates])
            best = int(np.argmax(scores))
            best_score = float(scores[best])

            if best_score >= self.threshold:
                candidates[best].update(embedding)
                return candidates[best].id, best_score

            # Below threshold: only mint a new speaker given enough audio to trust it,
            # otherwise one short noisy turn fragments the roster.
            if (
                len(self._profiles) < self.max_speakers
                and len(audio) >= self.min_new_speaker_samples
            ):
                return self._mint(embedding).id, 1.0 - best_score

            return None, best_score

    # --- persistence -----------------------------------------------------
    def save(self) -> None:
        """Persist named speakers so they're recognised in later sessions.

        Ids are deliberately absent: they mean nothing outside the session that minted them,
        and writing them would only create a file format to migrate later.
        """
        if not self.profile_path:
            return
        with self._lock:
            payload = [
                {"name": p.name, "count": p.count, "is_self": p.is_self,
                 "centroid": p.centroid.tolist()}
                for p in sorted(self._profiles.values(), key=lambda p: p.id)
                if p.pinned
            ]
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self) -> None:
        if not self.profile_path or not self.profile_path.is_file():
            return
        try:
            payload = json.loads(self.profile_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return
        for entry in payload:
            profile = self._mint(
                np.asarray(entry["centroid"], dtype=np.float32),
                name=entry.get("name"),
                pinned=True,
                is_self=entry.get("is_self", False),
            )
            profile.count = entry.get("count", 1)
