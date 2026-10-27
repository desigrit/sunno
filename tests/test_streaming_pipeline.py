"""Streaming handoff correctness, independent of hardware and model downloads."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from server.config import FRAME_SAMPLES, SAMPLE_RATE, Settings
from server.engine import Transcript
from server.pipeline import AsrWorker, CaptionPipeline, _Job
from server.preprocess import StreamingAudioConditioner


class RecordingEngine:
    def __init__(self):
        self.settings = Settings(model_size="stream-en", device="cpu")
        self.streams = []
        self.active = None
        self.closed = False
        self.fail_next = False

    def start_stream(self):
        self.active = []
        self.streams.append(self.active)

    def stream(self, audio, *, final=False):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("decode interrupted")
        self.active.extend(audio.tolist())
        return Transcript(str(len(self.active)), len(self.active) / SAMPLE_RATE, 1, final)

    def reset_stream(self):
        self.active = None

    def close(self):
        self.closed = True


class ConditionerTests(unittest.TestCase):
    def test_previous_frames_are_immutable_when_loud_audio_arrives(self):
        conditioner = StreamingAudioConditioner(Settings())
        rng = np.random.default_rng(5)
        first = conditioner((rng.normal(size=512) * .01).astype(np.float32))
        saved = first.copy()
        for _ in range(30):
            conditioner((rng.normal(size=512) * .5).astype(np.float32))
        np.testing.assert_array_equal(first, saved)
        self.assertFalse(first.flags.writeable)

    def test_reset_restores_original_output_and_input_is_not_modified(self):
        conditioner = StreamingAudioConditioner(Settings())
        data = np.random.default_rng(1).normal(0, .02, 512).astype(np.float32)
        original = data.copy()
        first = conditioner(data)
        conditioner(data * 10)
        conditioner.reset()
        np.testing.assert_array_equal(first, conditioner(data))
        np.testing.assert_array_equal(data, original)

    def test_filter_state_is_preserved_across_chunk_boundaries(self):
        settings = Settings(target_rms=0)
        signal = np.random.default_rng(3).normal(0, .02, 4096).astype(np.float32)
        whole = StreamingAudioConditioner(settings)(signal)
        conditioner = StreamingAudioConditioner(settings)
        split = np.concatenate([conditioner(chunk) for chunk in np.split(signal, 8)])
        np.testing.assert_array_equal(split, whole)

    def test_bad_audio_rejected_and_silence_not_amplified(self):
        conditioner = StreamingAudioConditioner(Settings())
        self.assertEqual(conditioner(np.zeros(0)).size, 0)
        self.assertFalse(np.any(conditioner(np.zeros(512))))
        for invalid in (np.array([np.nan]), np.array([np.inf]), np.zeros((2, 2))):
            with self.assertRaises(ValueError):
                conditioner(invalid)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.engine = RecordingEngine()
        self.events = []
        self.worker = AsrWorker(self.engine, self.events.append)

    def tearDown(self):
        if self.worker._thread.is_alive():
            self.worker.stop()

    def test_skipped_refresh_does_not_skip_audio(self):
        chunks = tuple(np.full(512, i, np.float32) for i in range(5))
        self.worker.submit_partial(_Job(1, None, False, chunks=chunks[:1]))
        self.worker.submit_partial(_Job(1, None, False, chunks=chunks[:3]))
        self.worker.start()
        self.assertTrue(self.worker.drain(2))
        self.worker.submit_final(_Job(1, None, True, chunks=chunks))
        self.assertTrue(self.worker.drain(2))
        self.assertEqual(len(self.engine.streams), 1)
        np.testing.assert_array_equal(self.engine.streams[0], np.concatenate(chunks))
        self.assertEqual(self.events[-1]["text"], str(5 * 512))

    def test_final_supersedes_pending_partial_without_duplicate_decode(self):
        audio = np.arange(512, dtype=np.float32)
        self.worker.submit_partial(_Job(7, audio[:100], False))
        self.worker.submit_final(_Job(7, audio, True))
        self.worker.start()
        self.assertTrue(self.worker.drain(2))
        np.testing.assert_array_equal(self.engine.streams[0], audio)
        self.assertEqual([event["type"] for event in self.events], ["final"])

    def test_new_utterance_starts_empty(self):
        self.worker.start()
        for index in (1, 2):
            self.worker.submit_final(_Job(index, np.ones(512), True))
        self.assertTrue(self.worker.drain(2))
        self.assertEqual([len(stream) for stream in self.engine.streams], [512, 512])

    def test_flush_suppresses_result_already_in_flight(self):
        entered, release = threading.Event(), threading.Event()
        original = self.engine.stream

        def blocked(audio, *, final=False):
            entered.set()
            self.assertTrue(release.wait(2))
            return original(audio, final=final)

        self.engine.stream = blocked
        self.worker.start()
        self.worker.submit_partial(_Job(1, np.ones(512), False))
        self.assertTrue(entered.wait(2))
        self.worker.flush()
        release.set()
        self.assertTrue(self.worker.drain(2))
        self.assertEqual(self.events, [])
        self.assertIsNone(self.engine.active)
        self.worker.submit_final(_Job(2, np.ones(512), True))
        self.assertTrue(self.worker.drain(2))
        self.assertEqual(self.events[-1]["id"], 2)

    def test_decoder_failure_replays_immutable_snapshot_on_next_job(self):
        self.engine.fail_next = True
        self.worker.start()
        self.worker.submit_partial(_Job(1, np.ones(512), False))
        self.assertTrue(self.worker.drain(2))
        self.worker.submit_final(_Job(1, np.ones(1024), True))
        self.assertTrue(self.worker.drain(2))
        self.assertEqual(self.events[0]["type"], "error")
        self.assertEqual(self.events[-1]["text"], "1024")

    def test_batch_engine_still_receives_complete_snapshots(self):
        class BatchEngine:
            settings = Settings(model_size="base", device="cpu")
            seen = []

            def partial(self, audio):
                self.seen.append(audio.copy())
                return Transcript("partial", len(audio) / SAMPLE_RATE, 1, False)

            def final(self, audio):
                self.seen.append(audio.copy())
                return Transcript("final", len(audio) / SAMPLE_RATE, 1, True)

        engine = BatchEngine()
        worker = AsrWorker(engine, self.events.append)
        with patch("server.hardware.record_latency"):
            worker.start()
            try:
                chunks = (np.ones(512), np.ones(512) * 2)
                worker.submit_partial(_Job(1, None, False, chunks=chunks[:1]))
                self.assertTrue(worker.drain(2))
                worker.submit_final(_Job(1, None, True, chunks=chunks))
                self.assertTrue(worker.drain(2))
            finally:
                worker.stop()
        self.assertEqual([len(audio) for audio in engine.seen], [512, 1024])

    def test_shutdown_closes_engine(self):
        self.worker.start()
        self.worker.stop()
        self.assertTrue(self.engine.closed)

    def test_failed_reset_does_not_kill_worker(self):
        original = self.engine.reset_stream
        failures = [True]

        def interrupted_reset():
            if failures:
                failures.pop()
                raise RuntimeError("helper exited during reset")
            original()

        self.engine.reset_stream = interrupted_reset
        self.worker.flush()
        self.worker.start()
        self.assertTrue(self.worker.drain(2))
        self.worker.submit_final(_Job(2, np.ones(512), True))
        self.assertTrue(self.worker.drain(2))
        self.assertEqual(self.events[-1]["text"], "512")

    def test_shutdown_closes_engine_even_if_reset_fails(self):
        def broken_reset():
            raise RuntimeError("reset failed")

        self.engine.reset_stream = broken_reset
        self.worker.start()
        self.worker.stop()
        self.assertTrue(self.engine.closed)

    def test_adaptive_cadence_never_runs_faster_than_configured(self):
        self.worker._last_decode_ms = 10
        self.assertEqual(self.worker.partial_interval_ms(450), 450)
        self.worker._last_decode_ms = 800
        self.assertEqual(self.worker.partial_interval_ms(450), 1000)


class PipelineTests(unittest.TestCase):
    def test_recording_keeps_raw_audio_and_streaming_receives_conditioned_frames(self):
        class VAD:
            def __init__(self, *_): pass
            def reset(self): pass
            def __call__(self, frame): return 1.0 if np.any(frame) else 0.0

        engine = RecordingEngine()
        raw, events = [], []
        settings = Settings(enable_speakers=False, min_partial_ms=32, start_frames=1,
                            end_silence_ms=64, min_utterance_ms=32)
        with patch("server.vad.StreamingSileroVAD", VAD):
            pipeline = CaptionPipeline(settings, engine, events.append,
                                       on_audio=lambda audio: raw.append(audio.copy()))
        frames = [np.full(FRAME_SAMPLES, .1, np.float32) for _ in range(4)]
        frames += [np.zeros(FRAME_SAMPLES, np.float32) for _ in range(2)]
        try:
            pipeline.run(iter(frames))
            self.assertTrue(pipeline.drain(2))
        finally:
            pipeline.close()
        np.testing.assert_array_equal(np.concatenate(raw), np.concatenate(frames))
        self.assertEqual(len(engine.streams[0]), len(frames) * FRAME_SAMPLES)
        self.assertLess(abs(float(np.mean(engine.streams[0]))), .01)
        self.assertEqual(events[-1]["type"], "final")


if __name__ == "__main__":
    unittest.main()
