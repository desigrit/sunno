"""Exercise the ARM model path on the build host without loading ARM binaries."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server import arm_whisper, hardware, models


class ArmReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {"Sunno_DATA_DIR": self.temp.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.arm = patch("server.arm_whisper.enabled", return_value=True)
        self.arm.start()
        self.addCleanup(self.arm.stop)

    def fill_model(self, name="base"):
        root = Path(self.temp.name) / "onnx-models" / name
        root.mkdir(parents=True, exist_ok=True)
        for filename in arm_whisper.FILES:
            (root / filename).write_bytes(b"model fixture")
        return root

    def test_catalog_excludes_ct2_models_and_keeps_streaming(self):
        self.assertEqual([x["id"] for x in models.catalog_for_platform()],
                         ["base", "tiny", "stream-en", "stream-en-kroko"])
        self.assertFalse(models.uses_cuda("base"))
        self.assertFalse(hardware.has_cuda())

    def test_x64_catalog_and_cuda_are_preserved(self):
        with patch("server.arm_whisper.enabled", return_value=False):
            self.assertIs(models.catalog_for_platform(), models.CATALOG)
            self.assertTrue(models.uses_cuda("large-v3"))
            self.assertFalse(models.uses_cuda("stream-en"))

    def test_incomplete_model_is_not_available(self):
        root = self.fill_model()
        (root / "decoder.onnx.data").unlink()
        self.assertFalse(models.is_available("base").available)
        with self.assertRaises(FileNotFoundError): arm_whisper.model_path("base")
        (root / "decoder.onnx.data").touch()
        self.assertFalse(models.is_available("base").available)

    def test_complete_model_resolves_without_ct2_cache(self):
        root = self.fill_model()
        with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("CT2 cache")):
            self.assertEqual(models.is_available("base").path, str(root))
            self.assertEqual(models.onnx_model_path("base"), root)

    def test_wrong_architecture_model_is_rejected_before_download(self):
        with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("download")):
            with self.assertRaises(ValueError): models.download("large-v3")

    def test_download_uses_pinned_native_model_and_validates_completion(self):
        progress = []
        with patch("server.arm_whisper.total_bytes", return_value=123), \
             patch("huggingface_hub.snapshot_download", side_effect=lambda *a, **kw: self.fill_model()) as fetch:
            self.assertEqual(models.download("base", lambda d, t: progress.append((d, t))),
                             str(Path(self.temp.name) / "onnx-models" / "base"))
        self.assertEqual(fetch.call_args.args, (arm_whisper.REPO,))
        self.assertEqual(fetch.call_args.kwargs["revision"], arm_whisper.REVISION)
        self.assertEqual(set(fetch.call_args.kwargs["allow_patterns"]),
                         {"base/" + name for name in arm_whisper.FILES})
        self.assertEqual(progress[-1], (123, 123))

    def test_old_x64_model_selection_migrates_on_arm(self):
        from server.app import parse_args
        with patch("sys.argv", ["sunno", "--model", "large-v3"]), \
             patch("server.hardware._read_state", return_value={}):
            settings, _ = parse_args()
        self.assertEqual(settings.model_size, "base")
        self.assertEqual(settings.device, "cpu")

    def test_latency_does_not_reuse_emulated_whisper_measurements(self):
        with patch("server.hardware._read_state", return_value={"observed_lag_ms": {"cpu:base": 9000}}), \
             patch.dict(hardware._OBSERVED, {}, clear=True):
            self.assertEqual(hardware.estimated_lag_ms("base", "cpu"), 519)
        self.assertEqual(hardware._latency_key("base", "cpu"), "onnx:cpu:base")
        self.assertEqual(hardware._latency_key("stream-en", "cpu"), "cpu:stream-en")


if __name__ == "__main__":
    unittest.main()
