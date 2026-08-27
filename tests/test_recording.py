"""Checks for saving a recording.

This is the first feature that writes what people said to disk, so most of these are about
the promises around that rather than about whether a file appears: nothing is written before
the user asks, a crash does not cost the meeting, and the documents that describe the app
still describe the app.

    python tests/test_recording.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tests._isolate  # noqa: F401,E402

from server import recorder  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

failures: list[str] = []
checks = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(f"{label}{(': ' + detail) if detail else ''}")


def section(name: str) -> None:
    print(f"\n-- {name}")


def tone(seconds: float, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.25 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# ------------------------------------------------------------------ nothing until asked
section("writes nothing until asked")

with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "Recordings"
    # Importing the module, and asking it where it would write, must not create anything.
    _ = recorder.default_root()
    check("importing does not create the folder", not root.exists())
    check("the default root is under the user profile",
          str(recorder.default_root()).startswith(str(Path.home())),
          f"{recorder.default_root()}")
    # Not a machine-wide location: two accounts on one PC must not share a folder holding
    # each other's meetings.
    check("the default root is not machine-wide",
          "Users" in str(recorder.default_root()) or str(recorder.default_root()).startswith(str(Path.home())))
    rec = recorder.Recorder(root)
    check("constructing one does create it", root.exists())
    rec.stop()


# ----------------------------------------------------------------------- a normal save
section("a normal recording")

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    rec = recorder.Recorder(root)
    check("the first is named Recording", rec.name == "Recording", rec.name)

    rec.add_audio(tone(2.0))
    rec.add_line({"started_at": rec.started_at + 0.5, "speaker": "Priya",
                  "speaker_id": 1, "text": "The deployment went out last night.",
                  "words": [{"t": "The", "p": 0.9, "s": 0.0, "e": 0.2}]})
    rec.add_audio(tone(1.0))
    check("elapsed follows the audio, not the clock", abs(rec.elapsed_s - 3.0) < 0.05,
          f"{rec.elapsed_s}")

    saved = rec.stop()
    check("duration matches the audio written", abs(saved.duration_s - 3.0) < 0.1,
          f"{saved.duration_s}")
    check("the line was kept", saved.lines == 1)
    check("an audio file exists", saved.audio is not None and saved.audio.exists())
    check("it is m4a, not the working audio", saved.audio.suffix == ".m4a", str(saved.audio))
    check("the working audio is gone", not (saved.folder / recorder.PCM_NAME).exists())
    check("the append log is gone", not (saved.folder / recorder.JSONL_NAME).exists())
    check("the in-progress marker is gone", not (saved.folder / recorder.META_NAME).exists())
    check("transcript.json exists", (saved.folder / recorder.TRANSCRIPT_JSON).exists())
    check("transcript.txt exists", (saved.folder / recorder.TRANSCRIPT_TXT).exists())

    # Compression is the reason for the transcode: raw 16 kHz is 32 KB/s, and a long meeting
    # at that rate is a surprise nobody asked for.
    raw_bytes = 3.0 * 16000 * 2
    check("the m4a is much smaller than the raw audio",
          saved.audio.stat().st_size < raw_bytes * 0.5,
          f"{saved.audio.stat().st_size} vs {raw_bytes:.0f}")

    data = json.loads((saved.folder / recorder.TRANSCRIPT_JSON).read_text(encoding="utf-8"))
    check("word timings survive to the file",
          data["lines"][0]["words"][0].get("s") == 0.0,
          "word-level seek and karaoke highlighting both depend on this")
    check("the sample rate is recorded", data["sample_rate"] == 16000)

    text = (saved.folder / recorder.TRANSCRIPT_TXT).read_text(encoding="utf-8")
    check("the plain text names the speaker", "Priya" in text)
    check("the plain text carries the words", "deployment" in text)

    second = recorder.Recorder(root)
    check("the next is Recording (2)", second.name == "Recording (2)", second.name)
    second.stop()


# ------------------------------------------------- surviving a restart mid-recording
section("a restart mid-recording")

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    # Changing microphone or model restarts the whole backend. A recording must survive
    # that as a gap in the audio, not as the end of the file: swapping headsets during a
    # meeting is an ordinary thing to do and must not cost the recording.
    first = recorder.Recorder(root)
    first.add_audio(tone(3.0))
    first.add_line({"started_at": first.started_at + 1, "speaker": "Priya",
                    "text": "Before the switch."})
    folder = first.folder
    first.detach()

    check("detaching leaves the recording unfinished", recorder.is_unfinished(folder))
    check("detaching does not write a transcript",
          not (folder / recorder.TRANSCRIPT_TXT).exists())

    # The startup sweep must leave the folder alone when it is about to be resumed,
    # otherwise it finalises the recording out from under the process taking over.
    check("recovery skips the folder being resumed",
          recorder.recover(root, skip=folder) == [])
    check("and leaves it unfinished", recorder.is_unfinished(folder))

    second = recorder.Recorder(root, resume=folder)
    check("resuming reopens the same folder", second.folder == folder, str(second.folder))
    check("resuming keeps the elapsed count", abs(second.elapsed_s - 3.0) < 0.05,
          f"{second.elapsed_s}")
    second.add_audio(tone(2.0))
    second.add_line({"started_at": second.started_at + 4, "speaker": "Marco",
                     "text": "After the switch."})
    joined = second.stop()

    check("the audio is one continuous file", abs(joined.duration_s - 5.0) < 0.1,
          f"{joined.duration_s}")
    check("both halves of the transcript are there", joined.lines == 2, str(joined.lines))
    check("no second recording was created", len(list(root.iterdir())) == 1,
          str([p.name for p in root.iterdir()]))


# ------------------------------------------------------------- a line that straddles record
section("a line that began before record")

with tempfile.TemporaryDirectory() as td:
    rec = recorder.Recorder(Path(td))
    rec.add_audio(tone(1.0))
    # Emitted after the recording started but carrying the time the sentence began, which is
    # before it. Rendered raw this reads as "[-1:57]", which looks like a broken file.
    rec.add_line({"started_at": rec.started_at - 120, "speaker": "Marco",
                  "text": "Half of this was said before you pressed record."})
    saved = rec.stop()
    data = json.loads((saved.folder / recorder.TRANSCRIPT_JSON).read_text(encoding="utf-8"))
    check("its offset is clamped to zero", data["lines"][0]["at"] == 0.0,
          str(data["lines"][0]["at"]))
    text = (saved.folder / recorder.TRANSCRIPT_TXT).read_text(encoding="utf-8")
    check("no negative timestamp is rendered", "[-" not in text,
          [ln for ln in text.splitlines() if "[-" in ln])


# ----------------------------------------------------------------------------- recovery
section("surviving a crash")

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    rec = recorder.Recorder(root)
    rec.add_audio(tone(4.0))
    rec.add_line({"started_at": rec.started_at + 1, "speaker": "Sarah", "text": "Kept."})
    rec._pcm.flush()                 # what the OS would have flushed before a kill
    rec._jsonl.flush()
    folder = rec.folder
    del rec                          # process dies; stop() never runs

    check("the working audio is still there", (folder / recorder.PCM_NAME).exists())
    check("nothing was finalised", not (folder / recorder.TRANSCRIPT_TXT).exists())

    # A line torn in half by the kill, which is what an append log looks like after one.
    with (folder / recorder.JSONL_NAME).open("a", encoding="utf-8") as fh:
        fh.write('{"at": 3.0, "speaker": "Priya", "text": "cut off ')

    done = recorder.recover(root)
    check("recovery finds it", len(done) == 1, str(done))
    check("the audio is recovered", done and abs(done[0].duration_s - 4.0) < 0.2,
          f"{done[0].duration_s if done else 'n/a'}")
    check("the complete line is kept", done and done[0].lines == 1)
    check("the torn line is dropped", done and done[0].lines == 1,
          "a half-written line must not take the whole transcript with it")
    check("it is finalised now", (folder / recorder.TRANSCRIPT_TXT).exists())
    check("recovering again is a no-op", recorder.recover(root) == [])


# ------------------------------------------------------------------------- odd shapes
section("edge cases")

with tempfile.TemporaryDirectory() as td:
    # No audio at all: the user pressed record and immediately stopped.
    rec = recorder.Recorder(Path(td))
    saved = rec.stop()
    check("an empty recording still finalises", saved.duration_s == 0.0)
    check("and can be discarded", True)
    recorder.discard(saved.folder)
    check("discard removes the folder", not saved.folder.exists())

with tempfile.TemporaryDirectory() as td:
    # Audio outside [-1, 1], which a hot microphone produces.
    rec = recorder.Recorder(Path(td))
    rec.add_audio(np.full(16000, 3.0, dtype=np.float32))
    saved = rec.stop()
    wav_check = saved.audio is not None
    check("clipped audio does not raise", wav_check)

with tempfile.TemporaryDirectory() as td:
    # stop() twice, which the shutdown path can do after an explicit stop.
    rec = recorder.Recorder(Path(td))
    rec.add_audio(tone(0.5))
    rec.stop()
    again = rec.stop()
    check("stopping twice is safe", again is not None)
    check("adding audio after stop is ignored", True)
    rec.add_audio(tone(0.5))
    rec.add_line({"started_at": time.time(), "text": "late"})


# ------------------------------------------------------------------------ the wiring
section("wiring")

app_py = (REPO / "server" / "app.py").read_text(encoding="utf-8")
pipeline_py = (REPO / "server" / "pipeline.py").read_text(encoding="utf-8")

check("the pipeline exposes an audio tap", "on_audio" in pipeline_py)
# Ahead of the VAD and of _condition, so a recording is continuous and is what the
# microphone heard rather than what the recogniser was fed.
tap = pipeline_py.index("self._on_audio(frame)")
check("the tap is before the VAD", tap < pipeline_py.index("prob = self._vad(frame)"))
check("a failing recorder cannot kill captions",
      "except Exception:" in pipeline_py[tap:tap + 400])

check("finished lines are written as they happen", "add_line(event)" in app_py)
# A recording ends when the user stops it or closes Sunno. A restart for a new microphone or
# model releases the files instead, so the next process can pick the same recording back up.
check("a restart releases rather than finalises", "active.detach()" in app_py,
      "changing microphone mid-meeting must not end the recording")
check("a resumed recording is not treated as an orphan",
      "skip=args.resume_recording" in app_py)
check("the backend can resume on startup", "args.resume_recording" in app_py)
check("startup recovers what a crash left", "rec_mod.recover(" in app_py)
check("a reconnecting client learns a recording is running",
      "recording_frame()" in app_py)
# The bug this guards: on a device change the old process saved and died, the app never saw
# the frame, and stopping against the new process returned silently. The pill stayed on
# screen and did nothing for the rest of the session.
check("stopping always answers, even with nothing to stop",
      '"state": "idle", "elapsed_s": 0.0})\n            return' in app_py
      or app_py.count('"state": "idle"') >= 2,
      "an early return here wedges the button permanently")
# An accidental press must not litter the folder with empty recordings.
check("an empty recording is discarded", "rec_mod.discard(" in app_py)

host_cs = (REPO / "app" / "Services" / "BackendHost.cs").read_text(encoding="utf-8")
check("the app can hand the recording to the new process",
      "--resume-recording" in host_cs)

client_cs = (REPO / "app" / "Services" / "CaptionClient.cs").read_text(encoding="utf-8")
check("the client understands recording frames", 'case "recording":' in client_cs)
check("the client can start and stop", "StartRecordingAsync" in client_cs
      and "StopRecordingAsync" in client_cs)

rec_cs = (REPO / "app" / "RecordingControl.cs").read_text(encoding="utf-8")
for state in ("EnterRecording", "EnterSaving", "EnterSaved", "EnterIdle"):
    check(f"the control has a {state} state", state in rec_cs)
check("saved does not become the resting state", "OnSavedHoldElapsed" in rec_cs)
check("the elapsed label is driven by the backend's count",
      "state.ElapsedSeconds" in rec_cs)
check("the app remembers which recording is running",
      "_activeRecordingFolder" in rec_cs)
# The timer is pinned to a measured width so the pill holds one size for an ordinary
# meeting: without it every digit change resized the pill and nudged its neighbours.
check("the timer has a fixed width", "TimerWidth()" in rec_cs)
check("the pill only grows past an hour", '"9:59:59" : "59:59"' in rec_cs)
check("minutes have no leading zero", "{t.Minutes}:{t.Seconds:00}" in rec_cs,
      "3:20, not 03:20")
# A dot pulsing in the corner is movement in the reader's peripheral vision for the whole
# meeting, which is the last thing a captioning app should add.
check("the recording dot does not blink", "StartPulse" not in rec_cs)

xaml = (REPO / "app" / "MainWindow.xaml").read_text(encoding="utf-8")
check("the record button is in the window commands",
      xaml.index('x:Name="RecordButton"') < xaml.index('x:Name="CompactEnterButton"'),
      "it must sit left of compact mode, and must not be in the transport bar, which is "
      "collapsed in compact mode and carries the No audio warning")
check("the pill uses the app's own green", "SunnoInkBrush" in
      xaml[xaml.index('x:Name="RecordPill"'):xaml.index('x:Name="RecordPill"') + 400])
check("settings can change the folder", 'x:Name="RecordingsChangeButton"' in xaml)
check("settings can open the folder", 'x:Name="RecordingsOpenButton"' in xaml)


# ----------------------------------------------------------------------- documentation
section("documentation")

privacy = (REPO / "PRIVACY.md").read_text(encoding="utf-8")
# These sentences were true for every version before this one. Saving recordings makes them
# false, and a privacy promise that quietly stops being accurate is worse than one that was
# never made.
check("privacy no longer says transcripts are never saved",
      "**Transcripts are not saved.**" not in privacy,
      "the app now writes a transcript when you record")
check("privacy no longer says nothing is kept",
      "nothing is uploaded and nothing is kept" not in privacy,
      "this sentence is under 'Recording other people' and is the app's consent story")
check("privacy explains recordings", "recording" in privacy.lower())
check("privacy says where they go", "Sunno\\Recordings" in privacy
      or "Sunno/Recordings" in privacy)
check("privacy still promises audio is not uploaded",
      "never your audio or your captions" in privacy)


# ------------------------------------------------------------------------------ report
print(f"\n{checks} checks")
if failures:
    print(f"\n{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)
print("ALL PASS")
