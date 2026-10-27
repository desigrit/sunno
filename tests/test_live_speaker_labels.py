"""Deterministic speaker-state regressions, not a diarization accuracy benchmark."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import tests._isolate  # noqa: F401
from server.config import Settings, SAMPLE_RATE, FRAME_SAMPLES
from server.speaker import SpeakerIdentifier, _l2_normalise
from server.speaker_labels import LiveSpeakerLabels
from server.recorder import Recorder, _read_lines, _count_lines


def identifier():
    speaker = object.__new__(SpeakerIdentifier)
    speaker._profiles = {}
    speaker._next_id = 0
    speaker._lock = threading.Lock()
    speaker.threshold = .5
    speaker.max_speakers = 8
    speaker.min_identify_samples = SAMPLE_RATE
    speaker.min_new_speaker_samples = SAMPLE_RATE * 2
    speaker.profile_path = None
    speaker.embed = Mock(return_value=np.array([1., 0., 0.], np.float32))
    return speaker


class LiveLabelTests(unittest.TestCase):
    def setUp(self):
        self.speaker = identifier()
        self.events = []
        self.labels = LiveSpeakerLabels(self.speaker, self.events.append)
        self.addCleanup(self.labels.close)

    def observe(self, uid=1, seconds=3, final=True):
        self.labels.observe(uid, (np.ones(int(seconds * SAMPLE_RATE), np.float32),), final)
        self.assertTrue(self.labels.drain(2))

    def test_first_words_have_a_label_without_waiting_for_inference(self):
        self.labels.begin(1, 100.)
        self.labels.emit_caption({"type": "partial", "id": 1, "text": "hello"})
        self.assertEqual(self.events[-1]["speaker"], "Speaker 1")
        self.speaker.embed.assert_not_called()

    def test_longer_evidence_corrects_same_line_without_rewriting_words(self):
        self.labels.begin(1, 100.)
        self.observe()
        self.labels.begin(2, 200.)
        self.labels.emit_caption({"type": "partial", "id": 2, "text": "new voice"})
        self.assertEqual(self.events[-1]["speaker_id"], 0)
        self.speaker.embed.return_value = np.array([0., 1., 0.], np.float32)
        self.observe(uid=2)
        correction = [e for e in self.events if e["type"] == "speaker_label"][-1]
        self.assertEqual((correction["id"], correction["started_at"], correction["speaker_id"]),
                         (2, 200., 1))
        self.assertNotIn("text", correction)
        self.labels.emit_caption({"type": "final", "id": 2, "text": "new voice", "speaker_id": 0})
        self.assertEqual(self.events[-1]["speaker_id"], 1)

    def test_off_stops_inference_and_removes_labels_from_new_captions(self):
        self.labels.set_enabled(False)
        self.labels.begin(1, 1.)
        self.observe()
        self.labels.emit_caption({"type": "final", "id": 1, "speaker_id": 8})
        self.assertIsNone(self.events[-1]["speaker_id"])
        self.speaker.embed.assert_not_called()
        self.assertEqual(self.labels.state()["mode"], "off")

    def test_off_then_on_rejects_inflight_result(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def slow(_):
            entered.set()
            release.wait(2)
            return np.array([1., 0., 0.], np.float32)
        self.speaker.embed.side_effect = slow
        self.labels.begin(1, 1.)
        self.labels.observe(1, (np.ones(48000, np.float32),), True)
        self.assertTrue(entered.wait(1))
        self.labels.set_enabled(False)
        self.labels.set_enabled(True)
        release.set()
        self.assertTrue(self.labels.drain(2))
        self.assertTrue(self.speaker.is_placeholder(0))

    def test_short_or_borderline_turn_does_not_split_roster(self):
        self.labels.begin(1, 1.)
        self.observe()
        noisy = np.array([0., 1., 0.], np.float32)
        self.assertEqual(self.speaker.assign(noisy, 1.5, 0, True)[0], 0)
        borderline = _l2_normalise(np.array([.45, .893, 0.], np.float32))
        self.assertEqual(self.speaker.assign(borderline, 3., 0, True)[0], 0)
        self.assertEqual(self.speaker.num_speakers, 1)

    def test_off_on_between_audio_snapshot_and_enqueue_drops_old_audio(self):
        self.labels.begin(1, 1.)
        old_generation = self.labels.generation
        self.labels.set_enabled(False)
        self.labels.set_enabled(True)
        self.labels.observe(1, (np.ones(48000, np.float32),), True,
                            expected_generation=old_generation)
        self.assertTrue(self.labels.drain(2))
        self.speaker.embed.assert_not_called()
        self.labels.observe(1, (np.ones(48000, np.float32),), True,
                            expected_generation=self.labels.generation)
        self.assertTrue(self.labels.drain(2))
        self.speaker.embed.assert_called_once()

    def test_partials_do_not_repeatedly_train_reference(self):
        self.labels.begin(1, 1.)
        self.observe()
        before = self.speaker._profiles[0].count
        for _ in range(4):
            self.observe(final=False)
        self.assertEqual(self.speaker._profiles[0].count, before)

    def test_recent_weak_assignment_is_revisited_after_better_reference(self):
        weak = _l2_normalise(np.array([.48, .88, 0.], np.float32))
        self.speaker._mint(np.array([1., 0., 0.], np.float32))
        self.labels._last_speaker = 0
        self.labels.begin(1, 1.)
        self.speaker.embed.return_value = weak
        self.observe(seconds=1.5)
        # A new, speech-rich turn identifies this voice unambiguously.
        self.speaker._mint(weak)
        self.labels.begin(2, 2.)
        self.observe(uid=2)
        self.assertEqual(self.labels._assignments[1].speaker_id, 1)
        self.assertTrue(any(e["type"] == "speaker_label" and e["id"] == 1 for e in self.events))

    def test_named_people_are_not_automatically_merged(self):
        for name in ("A", "B"):
            p = self.speaker._mint(np.array([1., 0., 0.], np.float32), name=name, pinned=True)
            p.count = 10
        self.assertIsNone(self.speaker.duplicate())

    def test_strong_duplicate_references_merge_without_reusing_ids(self):
        for _ in range(2):
            p = self.speaker._mint(np.array([1., 0., 0.], np.float32))
            p.count = 3
        self.assertEqual(self.speaker.duplicate(), (1, 0))
        self.labels._last_speaker = 1
        self.labels.begin(1, 1.)
        self.labels.merge(1, 0, automatic=True)
        self.assertEqual(self.labels._assignments[1].speaker_id, 0)
        self.assertEqual(self.speaker._next_id, 2)

    def test_empty_provisional_can_merge_into_named_reference(self):
        known = self.speaker._mint(np.array([1., 0., 0.], np.float32), name="Known", pinned=True)
        self.labels.begin(1, 1.)
        self.observe()
        self.assertEqual(self.speaker.num_speakers, 1)
        self.assertEqual(self.labels._assignments[1].speaker_id, known.id)

    def test_bounded_history_and_failed_embedding_leave_captions_working(self):
        self.speaker.embed.side_effect = RuntimeError("unavailable")
        for uid in range(300):
            self.labels.begin(uid, float(uid))
        self.observe(uid=299)
        self.assertLessEqual(len(self.labels._assignments), 256)
        self.labels.emit_caption({"type": "final", "id": 299, "text": "still here"})
        self.assertEqual(self.events[-1]["text"], "still here")


class PipelineVoiceSelectionTests(unittest.TestCase):
    def test_only_voiced_frames_reach_speaker_model_and_snapshots_stay_bounded(self):
        from server.pipeline import CaptionPipeline
        from tests.test_streaming_pipeline import RecordingEngine

        class VAD:
            def __init__(self, *_): pass
            def reset(self): pass
            def __call__(self, frame): return float(frame[0])

        for speech_frames in (80, 320):
            with self.subTest(speech_frames=speech_frames):
                speaker = identifier()
                events = []
                settings = Settings(start_frames=1, end_silence_ms=64, min_utterance_ms=32,
                                    min_partial_ms=999999, max_utterance_s=20)
                with patch("server.vad.StreamingSileroVAD", VAD):
                    pipeline = CaptionPipeline(settings, RecordingEngine(), events.append,
                                               speaker=speaker)
                pipeline._condition = Mock(side_effect=lambda frame: frame)
                pipeline._condition.reset = Mock()
                frames = [np.full(FRAME_SAMPLES, .1, np.float32) for _ in range(6)]
                for i in range(speech_frames):
                    frames.append(np.full(FRAME_SAMPLES, .9, np.float32))
                    if i % 10 == 0:  # Brief gaps within a turn should not enter its embedding.
                        frames.append(np.full(FRAME_SAMPLES, .1, np.float32))
                frames.extend([np.full(FRAME_SAMPLES, .1, np.float32) for _ in range(2)])
                try:
                    pipeline.run(iter(frames))
                    self.assertTrue(pipeline.drain(2))
                finally:
                    pipeline.close()
                speaker.embed.assert_called_once()
                audio = speaker.embed.call_args.args[0]
                self.assertEqual(len(audio), min(speech_frames * FRAME_SAMPLES, 8 * SAMPLE_RATE))
                self.assertTrue(np.all(audio == np.float32(.9)))
                self.assertEqual(len([e for e in events if e["type"] == "final"]), 1)


class RecordingCorrectionTests(unittest.TestCase):
    def test_corrections_and_merges_survive_recovery_and_do_not_cross_restarts(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = Recorder(Path(folder))
            recorder.add_line({"id": 1, "text": "first", "speaker_id": 0, "speaker": "Speaker 1"})
            recorder.add_speaker_event({"type": "speaker_label", "id": 1, "speaker_id": 1, "speaker": "Speaker 2"})
            recorder.add_speaker_event({"type": "speaker_merged", "from": 1, "into": 2})
            recorder.add_speaker_event({"type": "roster", "speakers": [{"id": 2, "label": "Presenter"}]})
            recorder.detach()
            resumed = Recorder(Path(folder), resume=recorder.folder)
            resumed.add_line({"id": 1, "text": "second", "speaker_id": 0, "speaker": "New person"})
            resumed.add_speaker_event({"type": "speaker_label", "id": 1, "speaker_id": 1, "speaker": "Different"})
            resumed.detach()
            rows = _read_lines(recorder.folder)
            self.assertEqual([r["speaker"] for r in rows], ["Presenter", "Different"])
            self.assertEqual(_count_lines(recorder.folder), 2)
            self.assertNotIn("_session", rows[0])


if __name__ == "__main__":
    unittest.main()
