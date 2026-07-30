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

    __slots__ = ("centroid", "count", "name", "pinned", "is_self")

    def __init__(self, embedding: np.ndarray, name: str | None = None,
                 pinned: bool = False, is_self: bool = False) -> None:
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
    """Assigns stable speaker ids to utterances, discovering speakers as they appear."""

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
        self._profiles: list[_Profile] = []
        self._lock = threading.Lock()
        if self.profile_path:
            self.load()

    @property
    def num_speakers(self) -> int:
        return len(self._profiles)

    def label(self, speaker_id: int | None) -> str | None:
        if speaker_id is None or speaker_id >= len(self._profiles):
            return None
        profile = self._profiles[speaker_id]
        return profile.name or f"Speaker {speaker_id + 1}"

    def roster(self) -> list[dict]:
        with self._lock:
            return [
                {"id": i, "label": p.name or f"Speaker {i + 1}",
                 "named": p.name is not None, "is_self": p.is_self}
                for i, p in enumerate(self._profiles)
            ]

    def reset(self) -> None:
        """Forget discovered speakers, keeping any the user has named."""
        with self._lock:
            self._profiles = [p for p in self._profiles if p.pinned]

    def rename(self, speaker_id: int, name: str) -> bool:
        """Name a speaker and pin their profile so it stops drifting."""
        with self._lock:
            if not (0 <= speaker_id < len(self._profiles)):
                return False
            cleaned = name.strip()
            self._profiles[speaker_id].name = cleaned or None
            self._profiles[speaker_id].pinned = bool(cleaned)
        self.save()
        return True

    def set_self(self, speaker_id: int, is_self: bool = True) -> bool:
        """Mark a speaker as the user themselves.

        Their speech is still transcribed - the user reads it back as clarity feedback for
        speech practice - but it is rendered distinctly and carries a clarity score, so
        their own lines never get confused with what other people said.
        """
        with self._lock:
            if not (0 <= speaker_id < len(self._profiles)):
                return False
            for i, profile in enumerate(self._profiles):
                # Only one speaker can be "you".
                profile.is_self = is_self and i == speaker_id
            if is_self:
                # Pin it: this profile must stay stable or the marking will drift off.
                self._profiles[speaker_id].pinned = True
        self.save()
        return True

    def is_self_speaker(self, speaker_id: int | None) -> bool:
        if speaker_id is None:
            return False
        with self._lock:
            return 0 <= speaker_id < len(self._profiles) and self._profiles[speaker_id].is_self

    def merge(self, source_id: int, target_id: int) -> bool:
        """Fold one speaker into another, for when discovery split one person in two."""
        with self._lock:
            n = len(self._profiles)
            if not (0 <= source_id < n and 0 <= target_id < n) or source_id == target_id:
                return False
            target = self._profiles[target_id]
            source = self._profiles[source_id]
            total = target.count + source.count
            target.centroid = _l2_normalise(
                (target.centroid * target.count + source.centroid * source.count) / total
            )
            target.count = total
            target.name = target.name or source.name
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
                self._profiles.append(_Profile(embedding))
                return 0, 1.0

            scores = np.array([float(p.centroid @ embedding) for p in self._profiles])
            best = int(np.argmax(scores))
            best_score = float(scores[best])

            if best_score >= self.threshold:
                self._profiles[best].update(embedding)
                return best, best_score

            # Below threshold: only mint a new speaker given enough audio to trust it,
            # otherwise one short noisy turn fragments the roster.
            if (
                len(self._profiles) < self.max_speakers
                and len(audio) >= self.min_new_speaker_samples
            ):
                self._profiles.append(_Profile(embedding))
                return len(self._profiles) - 1, 1.0 - best_score

            return None, best_score

    # --- persistence -----------------------------------------------------
    def save(self) -> None:
        """Persist named speakers so they're recognised in later sessions."""
        if not self.profile_path:
            return
        with self._lock:
            payload = [
                {"name": p.name, "count": p.count, "is_self": p.is_self,
                 "centroid": p.centroid.tolist()}
                for p in self._profiles
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
            profile = _Profile(
                np.asarray(entry["centroid"], dtype=np.float32),
                name=entry.get("name"),
                pinned=True,
                is_self=entry.get("is_self", False),
            )
            profile.count = entry.get("count", 1)
            self._profiles.append(profile)
