"""Validate delay arithmetic and observation hooks without model downloads."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from bench.bench_caption_delay import caption_rows, distribution, measured_pipeline
from server.config import Settings
from server.engine import Transcript


class CaptionDelayMetricTests(unittest.TestCase):
    def test_delays_are_relative_to_speech_not_file_start(self):
        segments = [{"id": 1, "vad_onset_s": 0.5, "vad_offset_s": 3.5}]
        events = [
            {"type": "speech_start", "id": 1, "elapsed_s": 0.6},
            {"type": "partial", "id": 1, "elapsed_s": 1.2, "text": "Hello"},
            {"type": "partial", "id": 1, "elapsed_s": 1.7, "text": "Hello"},
            {"type": "final", "id": 1, "elapsed_s": 4.1, "text": "Hello there",
             "latency_ms": 60, "queue_ms": 10},
        ]
        row, = caption_rows(segments, events)
        self.assertAlmostEqual(row["first_caption_delay_ms"], 700)
        self.assertAlmostEqual(row["final_caption_delay_ms"], 600)
        self.assertEqual(row["changed_partial_updates"], 1)
        self.assertEqual(row["final_decode_ms"], 60)

    def test_final_only_and_empty_results_are_not_hidden(self):
        segments = [{"id": 1, "vad_onset_s": 0, "vad_offset_s": 1},
                    {"id": 2, "vad_onset_s": 2, "vad_offset_s": 3}]
        events = [{"type": "final", "id": 1, "elapsed_s": 1.6, "text": "Hello"},
                  {"type": "final", "id": 2, "elapsed_s": 3.6, "text": ""}]
        first, empty = caption_rows(segments, events)
        self.assertEqual(first["first_caption_type"], "final")
        self.assertAlmostEqual(first["first_caption_delay_ms"], 1600)
        self.assertIsNone(empty["first_caption_delay_ms"])
        self.assertIsNone(empty["final_caption_delay_ms"])
        self.assertEqual(distribution([None, 10, 20])["n"], 2)
        self.assertIsNone(distribution([])["median_ms"])

    def test_results_do_not_leak_across_utterances(self):
        rows = caption_rows([
            {"id": 4, "vad_onset_s": 1, "vad_offset_s": 2},
            {"id": 5, "vad_onset_s": 3, "vad_offset_s": 4},
        ], [
            {"type": "final", "id": 5, "elapsed_s": 4.7, "text": "Second"},
            {"type": "final", "id": 4, "elapsed_s": 4.8, "text": "First"},
        ])
        self.assertAlmostEqual(rows[0]["final_caption_delay_ms"], 2800)
        self.assertAlmostEqual(rows[1]["final_caption_delay_ms"], 700)


class ObserverTests(unittest.TestCase):
    def test_onset_and_offset_hooks_follow_real_segmentation(self):
        class FakeVAD:
            def __init__(self, *args):
                self.reset()

            def reset(self):
                self.probabilities = iter([0, 0] + [1] * 30 + [0] * 16)

            def __call__(self, frame):
                return next(self.probabilities)

        class FakeEngine:
            settings = Settings(model_size="moonshine-tiny")

            def start_stream(self):
                pass

            def stream(self, audio, *, final=False):
                return Transcript("test", 1, 1, final)

            def reset_stream(self):
                pass

        with patch("server.vad.StreamingSileroVAD", FakeVAD):
            events = []
            pipeline = measured_pipeline(Settings(), FakeEngine(), events.append)
            try:
                pipeline.run([np.zeros(512, np.float32) for _ in range(48)])
                self.assertTrue(pipeline.drain(2))
                segment, = pipeline.segments
                self.assertAlmostEqual(segment["vad_onset_s"], 0.064)
                self.assertAlmostEqual(segment["vad_offset_s"], 1.024)
                self.assertAlmostEqual(segment["final_submitted_audio_s"], 1.536)
                self.assertEqual(segment["finish_reason"], "silence_endpoint")
                self.assertEqual(events[-1]["type"], "final")
            finally:
                pipeline.close()


if __name__ == "__main__":
    unittest.main()
