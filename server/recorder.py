"""Write a recording of the conversation to disk.

Sunno held audio in memory only until this existed. That was a promise as much as an
implementation, so the way this writes matters:

**Nothing is written until asked.** A Recorder is only constructed when the user presses
record, and the directory is only created then. An install that never records leaves no trace.

**A recording ends when the user says so, and not before.** Pausing, switching microphone,
changing model and a dropped USB cable all stop capture, and none of them should end a
recording: they should leave a gap in it. Pause and a transient failure keep the same process,
so the Recorder simply stops being fed. Changing device or model restarts the backend
entirely, so a recording has to be re-openable across processes -- which is why the audio is
appended as raw PCM rather than written through a container that owns its own header.

**A crash must not cost the meeting.** The scenario this exists for is a conversation that
cannot be re-run, so nothing is buffered for the end. Audio is appended as it arrives and each
finished line is appended immediately. Both survive the process dying at any point: raw PCM
has nothing to corrupt, and a JSONL file is valid up to its last complete line. AAC in an MP4
container is the opposite -- it needs its index written on close, and a killed encoder leaves
a file that will not play at all -- so the m4a is produced at the end, by transcoding.

**A killed process leaves recoverable work.** ``recover`` finds any recording that was never
finalised and completes it on the next launch, so the file appears rather than being lost.

The audio is the recogniser's own stream: 16 kHz mono, tapped ahead of ``preprocess`` so it is
what the microphone heard rather than what the model was fed. Fine for speech, and it means
the audio and the transcript can never disagree, because they are the same samples.
"""

from __future__ import annotations

import json
import shutil
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE

AUDIO_NAME = "audio.m4a"
PCM_NAME = "audio.pcm"
WAV_NAME = "audio.wav"          # written by 1.0.78.0 only; still recovered
JSONL_NAME = "lines.jsonl"
META_NAME = "recording.json"
TRANSCRIPT_JSON = "transcript.json"
TRANSCRIPT_TXT = "transcript.txt"

BYTES_PER_SAMPLE = 2


@dataclass
class Saved:
    """What a finished recording turned into."""

    name: str
    folder: Path
    duration_s: float
    lines: int
    audio: Path | None


def default_root() -> Path:
    """Where recordings go unless the user has chosen otherwise.

    Under the user profile rather than a machine-wide folder: two accounts on one PC must not
    share a recordings folder, and the profile root is one of the few writable places Windows
    does not redirect into OneDrive, so a meeting does not sync to a cloud tenant by accident.
    """
    return Path.home() / "Sunno" / "Recordings"


def next_name(root: Path) -> str:
    """`Recording`, then `Recording (2)`, matching the inbox Sound Recorder."""
    if not (root / "Recording").exists():
        return "Recording"
    n = 2
    while (root / f"Recording ({n})").exists():
        n += 1
    return f"Recording ({n})"


def _count_lines(folder: Path) -> int:
    path = folder / JSONL_NAME
    if not path.exists():
        return 0
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


class Recorder:
    """One recording, from the press of record to the file on disk.

    Args:
        root: the recordings folder.
        resume: an existing recording folder to continue appending to, used when the backend
            restarts for a new microphone or model part-way through a recording.
    """

    def __init__(self, root: Path, resume: Path | str | None = None) -> None:
        self.root = Path(root)
        self._closed = False

        if resume is not None and Path(resume).is_dir():
            self.folder = Path(resume)
            self.name = self.folder.name
            self.started_at = self._read_started_at()
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            self.name = next_name(self.root)
            self.folder = self.root / self.name
            self.folder.mkdir(parents=True, exist_ok=True)
            self.started_at = time.time()
            (self.folder / META_NAME).write_text(json.dumps({
                "name": self.name,
                "started_at": self.started_at,
                "sample_rate": SAMPLE_RATE,
            }), encoding="utf-8")

        # Append, both of them. Reopening a folder therefore continues the same recording
        # rather than starting a second one, and the gap where capture was stopped simply
        # does not appear in the audio.
        self._pcm = (self.folder / PCM_NAME).open("ab")
        self._jsonl = (self.folder / JSONL_NAME).open("a", encoding="utf-8")
        self._lines = _count_lines(self.folder)

    def _read_started_at(self) -> float:
        try:
            return float(json.loads(
                (self.folder / META_NAME).read_text(encoding="utf-8"))["started_at"])
        except Exception:
            return self.folder.stat().st_mtime

    @property
    def elapsed_s(self) -> float:
        """Length of the audio written, not wall-clock time since the button was pressed.

        Those differ whenever capture stops and starts within one recording, and the audio
        length is the one that matches the file the user ends up with. Measured from the file
        so a resumed recording continues counting from where it left off.
        """
        try:
            self._pcm.flush()
            return (self.folder / PCM_NAME).stat().st_size / BYTES_PER_SAMPLE / SAMPLE_RATE
        except OSError:
            return 0.0

    def add_audio(self, frame: np.ndarray) -> None:
        """Append one frame. Called from the capture thread, so it stays cheap."""
        if self._closed:
            return
        pcm = np.clip(frame, -1.0, 1.0)
        self._pcm.write((pcm * 32767.0).astype("<i2").tobytes())

    def add_line(self, event: dict) -> None:
        """Record a finished caption. Written immediately, not held for the end."""
        if self._closed:
            return
        # Clamped at zero. A line is emitted when decoding finishes but carries the time the
        # utterance *began*, so a sentence already under way when record was pressed reports
        # a start before the recording existed. Left alone that renders as "[-1:57]", which
        # reads as a broken file rather than as a sentence that straddled the button.
        at = max(0.0, event.get("started_at", 0.0) - self.started_at)
        self._jsonl.write(json.dumps({
            "at": round(at, 2),
            "speaker": event.get("speaker"),
            "speaker_id": event.get("speaker_id"),
            "text": event.get("text", ""),
            "words": event.get("words") or [],
        }, ensure_ascii=False) + "\n")
        self._jsonl.flush()
        self._lines += 1

    def detach(self) -> None:
        """Let go of the files without finalising, so another process can take over.

        Used when the backend restarts for a new microphone or model. The recording is not
        over; this process is simply no longer the one writing it.
        """
        self._closed = True
        for handle in (self._pcm, self._jsonl):
            try:
                handle.close()
            except Exception:
                pass

    def stop(self) -> Saved:
        """Finalise: close the streams, transcode, and write the readable transcript."""
        if not self._closed:
            self.detach()
        return finalise(self.folder, self.started_at)


def _read_lines(folder: Path) -> list[dict]:
    out: list[dict] = []
    path = folder / JSONL_NAME
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except ValueError:
            # A line torn in half by a kill. Everything before it is still good, and
            # discarding just the fragment is the whole reason for appending line by line.
            continue
    return out


def _clock(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 \
        else f"{s // 60:02d}:{s % 60:02d}"


def is_unfinished(folder: Path) -> bool:
    """True when a folder holds a recording that was never finalised."""
    folder = Path(folder)
    return (folder / PCM_NAME).exists() or (folder / WAV_NAME).exists()


def _raw_samples(folder: Path) -> np.ndarray:
    """The audio, from whichever working form this recording used."""
    pcm = folder / PCM_NAME
    if pcm.exists():
        return np.frombuffer(pcm.read_bytes(), dtype="<i2")
    wav = folder / WAV_NAME
    if wav.exists():
        try:
            with wave.open(str(wav)) as w:
                return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        except Exception:
            return np.empty(0, dtype="<i2")
    return np.empty(0, dtype="<i2")


def _encode(folder: Path, samples: np.ndarray) -> Path | None:
    """Write the m4a. Returns None if it could not be done, leaving the raw audio in place.

    A failure here must not delete anything: raw audio that can still be recovered is worth
    far more than a tidy folder.
    """
    if samples.size == 0:
        return None
    try:
        import av
    except Exception:
        return None

    out = folder / AUDIO_NAME
    try:
        with av.open(str(out), "w") as dst:
            stream = dst.add_stream("aac", rate=SAMPLE_RATE)
            stream.layout = "mono"
            # In chunks, so an hour-long meeting is not handed to the encoder as one frame.
            step = SAMPLE_RATE * 30
            for start in range(0, samples.size, step):
                chunk = np.ascontiguousarray(samples[start:start + step])
                frame = av.AudioFrame.from_ndarray(
                    chunk.reshape(1, -1), format="s16", layout="mono")
                frame.rate = SAMPLE_RATE
                frame.pts = None
                for packet in stream.encode(frame):
                    dst.mux(packet)
            for packet in stream.encode(None):
                dst.mux(packet)
    except Exception:
        out.unlink(missing_ok=True)
        return None

    (folder / PCM_NAME).unlink(missing_ok=True)
    (folder / WAV_NAME).unlink(missing_ok=True)
    return out


def finalise(folder: Path, started_at: float | None = None) -> Saved:
    """Turn a recording folder into its finished form.

    Separate from Recorder so the same path serves a normal stop and a recovery on the next
    launch after a crash.
    """
    folder = Path(folder)
    lines = _read_lines(folder)
    samples = _raw_samples(folder)
    duration = samples.size / SAMPLE_RATE

    audio = _encode(folder, samples)
    if audio is None and samples.size:
        # Fall back to a real WAV so there is always something playable, even when the
        # encoder is missing.
        wav = folder / WAV_NAME
        try:
            with wave.open(str(wav), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(BYTES_PER_SAMPLE)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(samples.tobytes())
            (folder / PCM_NAME).unlink(missing_ok=True)
            audio = wav
        except Exception:
            audio = folder / PCM_NAME if (folder / PCM_NAME).exists() else None

    if started_at is None:
        try:
            started_at = float(json.loads(
                (folder / META_NAME).read_text(encoding="utf-8"))["started_at"])
        except Exception:
            started_at = folder.stat().st_mtime

    (folder / TRANSCRIPT_JSON).write_text(json.dumps({
        "name": folder.name,
        "started_at": started_at,
        "duration_s": round(duration, 2),
        "sample_rate": SAMPLE_RATE,
        "lines": lines,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Plain text as well, because a recording nobody can read without Sunno is not much of a
    # record. Timestamps are offsets into the audio, so a line can be found by scrubbing.
    body = [folder.name,
            time.strftime("%Y-%m-%d %H:%M", time.localtime(started_at)),
            f"Length {_clock(duration)}",
            ""]
    for ln in lines:
        who = ln.get("speaker") or "Speaker"
        body.append(f"[{_clock(ln.get('at', 0.0))}] {who}: {ln.get('text', '')}".rstrip())
    (folder / TRANSCRIPT_TXT).write_text("\n".join(body) + "\n", encoding="utf-8")

    (folder / JSONL_NAME).unlink(missing_ok=True)
    (folder / META_NAME).unlink(missing_ok=True)

    return Saved(folder.name, folder, round(duration, 2), len(lines), audio)


def recover(root: Path, skip: Path | str | None = None) -> list[Saved]:
    """Finish any recording the last run did not.

    Args:
        skip: a folder that is about to be resumed rather than recovered. The backend passes
            this when restarting mid-recording for a new microphone or model, so the
            recording in progress is not finalised out from under itself.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    keep = Path(skip).resolve() if skip else None
    done: list[Saved] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not is_unfinished(folder):
            continue
        if keep is not None and folder.resolve() == keep:
            continue
        try:
            done.append(finalise(folder))
        except Exception:
            continue
    return done


def discard(folder: Path) -> None:
    """Throw away a recording that captured nothing worth keeping."""
    shutil.rmtree(Path(folder), ignore_errors=True)
