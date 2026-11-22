"""Deterministic startup behavior, without downloads, capture or real model loading."""
import asyncio
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tests._isolate  # noqa: F401
from server import hardware, models, recorder
from server.asr import CTranslate2Engine
from server.config import Settings
from server.startup import StartupTimings


class StartupTests(unittest.TestCase):
    def test_cpu_only_models_skip_unused_gpu_initialization(self):
        from server.app import parse_args

        for model in ("moonshine-small", "stream-en", "windows-speech"):
            with self.subTest(model=model), \
                 patch("sys.argv", ["sunno", "--model", model]), \
                 patch("server.distribution.IS_STORE_BUILD", False), \
                 patch("server.arm_whisper.enabled", return_value=False), \
                 patch.object(hardware, "resolve_device") as resolve:
                settings, _ = parse_args()
                self.assertEqual(settings.model_size, model)
                self.assertEqual(settings.device, "cpu")
                resolve.assert_not_called()

    def test_timing_buffer_keeps_non_caption_commands_quiet(self):
        lines = []
        timing = StartupTimings(write=lambda line, **_: lines.append(line), enabled=False)
        timing.mark("imports_ready")
        self.assertEqual(lines, [])
        timing.enable()
        timing.enable()
        self.assertEqual(len(lines), 1)

    def test_cached_model_is_passed_to_whisper_without_remote_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("config.json", "model.bin", "tokenizer.json"):
                (Path(directory) / name).write_text("{}")
            whisper = Mock()
            with patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=whisper)}), \
                 patch("huggingface_hub.snapshot_download", return_value=directory) as resolve:
                CTranslate2Engine(Settings(model_size="large-v3", device="cpu", compute_type="int8"))
            self.assertTrue(resolve.call_args.kwargs["local_files_only"])
            whisper.assert_called_once_with(directory, device="cpu", compute_type="int8", local_files_only=True)

    def test_incomplete_cache_cannot_trigger_an_implicit_tokenizer_download(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("config.json", "model.bin"):
                (Path(directory) / name).write_text("{}")
            whisper = Mock()
            with patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=whisper)}):
                with self.assertRaises(FileNotFoundError):
                    CTranslate2Engine(Settings(model_size=directory))
            whisper.assert_not_called()

    def test_complete_custom_folder_never_queries_the_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("config.json", "model.bin", "tokenizer.json"):
                (Path(directory) / name).write_text("{}")
            with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("Network lookup")):
                self.assertEqual(models.local_whisper_path(directory), directory)

    def test_cached_cpu_estimate_does_not_benchmark_again(self):
        hardware.cpu_score.cache_clear()
        self.addCleanup(hardware.cpu_score.cache_clear)
        with patch.object(hardware, "_read_cached_score", return_value=123.0), \
             patch.object(hardware, "_measure_cpu_score") as measure:
            self.assertEqual(hardware.cpu_score(), 123.0)
            measure.assert_not_called()

    def test_recovery_snapshot_excludes_resumed_and_new_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, resumed, new = (root / name for name in ("old", "resumed", "new"))
            old.mkdir()
            resumed.mkdir()
            with patch.object(recorder, "is_unfinished", return_value=True), \
                 patch.object(recorder, "finalise") as finish:
                candidates = recorder.recovery_candidates(root, skip=resumed)
                new.mkdir()
                recorder.recover(root, candidates=candidates)
                finish.assert_called_once_with(old.resolve())

    def test_timing_failure_is_numeric_and_readiness_is_emitted_once(self):
        lines = []
        clock = iter([1., 2., 3., 4.]).__next__
        timing = StartupTimings(clock=clock, write=lambda line, **_: lines.append(line))
        timing.mark("control_ready")
        timing.mark("control_ready")
        with self.assertRaisesRegex(ValueError, "private"):
            with timing.measure("model_load"):
                raise ValueError("private device and path")
        self.assertEqual(len(lines), 2)
        self.assertIn("stage=control_ready duration_ms=1000 elapsed_ms=1000 result=ok", lines[0])
        self.assertIn("stage=model_load duration_ms=1000 elapsed_ms=3000 result=failed", lines[1])
        self.assertNotIn("private", " ".join(lines))

    def test_background_timed_work_keeps_control_loop_responsive(self):
        import threading

        entered, release = threading.Event(), threading.Event()
        def blocked():
            entered.set()
            release.wait(3)
        async def run():
            timing = StartupTimings(write=lambda *args, **kwargs: None)
            task = asyncio.create_task(timing.run("recording_recovery", blocked))
            try:
                while not entered.is_set():
                    await asyncio.sleep(.005)
                self.assertFalse(task.done())
            finally:
                release.set()
                await task
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
