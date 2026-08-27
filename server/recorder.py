"""Write a recording of the conversation to disk.

Sunno held audio in memory only until this existed. That was a promise as much as an
implementation, so the way this writes matters:

**Nothing is written until asked.** A Recorder is only constructed when the user presses
record, and the directory is only created then. An install that never records leaves no trace.

**A crash must not cost the meeting.** The scenario this feature exists for is a conversation
that cannot be re-run, so nothing is buffered for the end. Audio streams to a WAV as it
arrives and each finished line is appended to a JSONL file immediately. Both survive the
process dying at any point: WAV is length-prefixed but a truncated one still decodes, and a
JSONL file is valid up to its last complete line. AAC in an MP4 container is the opposite --
it needs its index written on close, and a killed encoder leaves a file that will not play at
all -- so the m4a is produced at the end, by transcoding, rather than written live.

**A killed process leaves recoverable work.** ``recover`` finds any recording whose WAV was
never finalised and completes it on the next launch, so the file appears rather than being
silently lost.

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
WAV_NAME = "audio.wav"
JSONL_NAME = "lines.jsonl"
TRANSCRIPT_JSON = "transcript.json"
TRANSCRIPT_TXT = "transcript.txt"


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


class Recorder:
    """One recording, from the press of record to the file on disk."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.name = next_name(self.root)
        self.folder = self.root / self.name
        self.folder.mkdir(parents=True, exist_ok=True)
        self.started_at = time.time()
        self._samples = 0
        self._lines = 0
        self._closed = False

        self._wav = wave.open(str(self.folder / WAV_NAME), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(SAMPLE_RATE)
        self._jsonl = (self.folder / JSONL_NAME).open("a", encoding="utf-8")

    @property
    def elapsed_s(self) -> float:
        """Length of the audio written, not wall-clock time since the button was pressed.

        Those differ whenever capture stops and starts within one recording, and the audio
        length is the one that matches the file the user ends up with.
        """
        return self._samples / SAMPLE_RATE

    def add_audio(self, frame: np.ndarray) -> None:
        """Append one frame. Called from the capture thread, so it stays cheap."""
        if self._closed:
            return
        pcm = np.clip(frame, -1.0, 1.0)
        self._wav.writeframes((pcm * 32767.0).astype("<i2").tobytes())
        self._samples += len(frame)

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

    def stop(self) -> Saved:
        """Finalise: close the streams, transcode, and write the readable transcript."""
        if not self._closed:
            self._closed = True
            try:
                self._wav.close()
            except Exception:
                pass
            try:
                self._jsonl.close()
            except Exception:
                pass
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


def _transcode(folder: Path) -> Path | None:
    """WAV to m4a. Returns None if it could not be done, leaving the WAV in place.

    A failure here is not fatal and must not delete anything: a playable WAV is worth far
    more than a tidy folder.
    """
    wav = folder / WAV_NAME
    if not wav.exists() or wav.stat().st_size <= 44:
        return None
    try:
        import av
    except Exception:
        return None

    out = folder / AUDIO_NAME
    try:
        with av.open(str(wav)) as src, av.open(str(out), "w") as dst:
            stream = dst.add_stream("aac", rate=SAMPLE_RATE)
            stream.layout = "mono"
            for frame in src.decode(audio=0):
                frame.pts = None
                for packet in stream.encode(frame):
                    dst.mux(packet)
            for packet in stream.encode(None):
                dst.mux(packet)
    except Exception:
        out.unlink(missing_ok=True)
        return None

    wav.unlink(missing_ok=True)
    return out


def finalise(folder: Path, started_at: float | None = None) -> Saved:
    """Turn a recording folder into its finished form.

    Separate from Recorder so the same path serves a normal stop and a recovery on the next
    launch after a crash.
    """
    folder = Path(folder)
    lines = _read_lines(folder)

    wav = folder / WAV_NAME
    duration = 0.0
    if wav.exists():
        try:
            with wave.open(str(wav)) as w:
                duration = w.getnframes() / float(w.getframerate() or SAMPLE_RATE)
        except Exception:
            duration = 0.0

    audio = _transcode(folder)
    if audio is None and wav.exists():
        audio = wav

    when = started_at or folder.stat().st_mtime
    (folder / TRANSCRIPT_JSON).write_text(json.dumps({
        "name": folder.name,
        "started_at": when,
        "duration_s": round(duration, 2),
        "sample_rate": SAMPLE_RATE,
        "lines": lines,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Plain text as well, because a recording nobody can read without Sunno is not much of a
    # record. Timestamps are offsets into the audio, so a line can be found by scrubbing.
    body = [f"{folder.name}",
            time.strftime("%Y-%m-%d %H:%M", time.localtime(when)),
            f"Length {_clock(duration)}",
            ""]
    for ln in lines:
        who = ln.get("speaker") or "Speaker"
        body.append(f"[{_clock(ln.get('at', 0.0))}] {who}: {ln.get('text', '')}".rstrip())
    (folder / TRANSCRIPT_TXT).write_text("\n".join(body) + "\n", encoding="utf-8")

    (folder / JSONL_NAME).unlink(missing_ok=True)

    return Saved(folder.name, folder, round(duration, 2), len(lines), audio)


def recover(root: Path) -> list[Saved]:
    """Finish any recording the last run did not.

    A folder holding a WAV is one where the process died mid-recording: a normal stop
    transcodes and removes it. Completing these on launch is the difference between losing a
    meeting and finding it waiting.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    done: list[Saved] = []
    for folder in sorted(root.iterdir()):
        if folder.is_dir() and (folder / WAV_NAME).exists():
            try:
                done.append(finalise(folder))
            except Exception:
                continue
    return done


def discard(folder: Path) -> None:
    """Throw away a recording that captured nothing worth keeping."""
    shutil.rmtree(Path(folder), ignore_errors=True)
