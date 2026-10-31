"""Opt-in selection and native helper recovery, without downloading system models."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from server import asr_native, distribution, experimental_models, models
from server.config import Settings
from server.engine import resolve_engine


class CatalogTests(unittest.TestCase):
    def test_every_new_engine_is_opt_in(self):
        for model_id in experimental_models.MODEL_IDS:
            self.assertFalse(models.auto_selectable(model_id), model_id)
        if hasattr(models, "uses_cuda"):
            for model_id in experimental_models.MODEL_IDS:
                self.assertFalse(models.uses_cuda(model_id), model_id)

    def test_platform_filter_and_unknown_latency(self):
        capability = {"supported": True, "available": False}
        for platform, present, absent in (("win32", "windows-speech", "apple-speech"),
                                          ("darwin", "apple-speech", "windows-speech")):
            with patch.object(experimental_models.sys, "platform", platform), \
                 patch.object(experimental_models, "status", return_value=capability):
                entries = experimental_models.catalog_entries()
            ids = {entry["id"] for entry in entries}
            self.assertIn(present, ids)
            self.assertNotIn(absent, ids)
            for entry in entries:
                self.assertEqual(entry["lag_ms"], 0)
                self.assertFalse(entry["auto_select"])
                self.assertFalse(entry["uses_gpu"])
                self.assertTrue(entry["supported"])
                self.assertFalse(entry["available"])

    def test_unsupported_engine_has_reason_and_cannot_be_auto_selected(self):
        with patch.object(experimental_models, "status", return_value={
                "supported": False, "available": False, "reason": "Requires an OS update"}):
            entries = experimental_models.catalog_entries()
        for entry in entries:
            self.assertFalse(entry["supported"])
            self.assertFalse(entry["auto_select"])
            self.assertEqual(entry["detail"], "Requires an OS update")

    def test_requested_model_selects_its_own_engine(self):
        with patch("server.engine.available_engines", return_value={"moonshine": True}):
            for model_id in ("moonshine-tiny", "moonshine-small", "moonshine-medium"):
                self.assertEqual(resolve_engine("ct2", model_id), "moonshine")
            for model_id in asr_native.NATIVE_MODELS:
                self.assertEqual(resolve_engine("ct2", model_id), "native")

    def test_missing_moonshine_does_not_fall_through_to_whisper(self):
        with patch("server.engine.available_engines", return_value={"ct2": True}):
            with self.assertRaisesRegex(RuntimeError, "Moonshine"):
                resolve_engine("auto", "moonshine-tiny")

    def test_native_model_download_size_is_not_fabricated(self):
        for model_id in asr_native.NATIVE_MODELS:
            self.assertEqual(models.total_size_bytes(model_id), 0)


class StoreDistributionTests(unittest.TestCase):
    def test_store_catalog_keeps_moonshine_but_never_probes_windows_ai(self):
        with patch.object(distribution, "IS_STORE_BUILD", True), \
             patch.object(experimental_models.sys, "platform", "win32"), \
             patch.object(experimental_models, "status", return_value={
                 "supported": True, "available": True}) as status:
            ids = {entry["id"] for entry in experimental_models.catalog_entries()}
        self.assertNotIn("windows-speech", ids)
        self.assertTrue({"moonshine-small", "moonshine-medium", "moonshine-tiny"} <= ids)
        self.assertNotIn("windows-speech", {call.args[0] for call in status.call_args_list})

    def test_store_blocks_direct_helper_and_download_requests(self):
        with patch.object(distribution, "IS_STORE_BUILD", True), \
             patch.object(asr_native.subprocess, "Popen") as spawn:
            self.assertIsNone(asr_native.service_path("windows-speech"))
            self.assertFalse(experimental_models.status("windows-speech")["supported"])
            with self.assertRaisesRegex(RuntimeError, "not included"):
                asr_native.download("windows-speech")
        spawn.assert_not_called()

    def test_store_migrates_only_the_unsupported_saved_selection(self):
        with patch.object(distribution, "IS_STORE_BUILD", True):
            self.assertIsNone(distribution.startup_model("windows-speech"))
            for model in ("large-v3", "stream-en", "moonshine-medium", None):
                self.assertEqual(distribution.startup_model(model), model)
        with patch.object(distribution, "IS_STORE_BUILD", False):
            self.assertEqual(distribution.startup_model("windows-speech"), "windows-speech")

    def test_store_startup_resolves_a_supported_hardware_default(self):
        from server.app import parse_args
        with patch.object(distribution, "IS_STORE_BUILD", True), \
             patch("sys.argv", ["sunno", "--model", "windows-speech"]), \
             patch("server.hardware.resolve_device", return_value="cpu"), \
             patch("server.hardware.compute_type_for", return_value="int8"), \
             patch("server.hardware.default_model", return_value="small") as choose:
            settings, _ = parse_args()
        self.assertEqual(settings.model_size, "small")
        choose.assert_called_once()


class FakeService:
    def __init__(self, *_):
        self._closed = False
        self._process = SimpleNamespace(poll=lambda: 1 if self._closed else None)
        self.calls = []
        self.fail = None

    def request(self, command, **values):
        self.calls.append((command, values))
        if command == self.fail:
            raise RuntimeError("helper unavailable")
        return {"ok": True, "text": "hello"}

    def close(self):
        self._closed = True


class NativeEngineTests(unittest.TestCase):
    def setUp(self):
        self.helpers = []

        def create(*args):
            helper = FakeService(*args)
            self.helpers.append(helper)
            return helper

        self.factory = patch.object(asr_native, "NativeService", side_effect=create)
        self.factory.start()
        self.addCleanup(self.factory.stop)
        self.engine = asr_native.NativeSpeechEngine(Settings(model_size="windows-speech"))
        self.addCleanup(self.engine.close)

    def test_load_never_provisions_or_opens_an_independent_microphone(self):
        self.assertEqual([call[0] for call in self.helpers[0].calls], ["load"])
        self.assertEqual(self.engine.warmup(), 0)
        self.assertEqual(len(self.helpers[0].calls), 1)

    def test_pcm_encoding_final_duration_and_reset(self):
        self.engine.start_stream()
        self.engine.stream(np.array([-2., 0., 2.], np.float32))
        _, values = self.helpers[0].calls[-1]
        pcm = np.frombuffer(base64.b64decode(values["samples"]), dtype="<i2")
        np.testing.assert_array_equal(pcm, [-32767, 0, 32767])
        result = self.engine.stream(np.zeros(157, np.float32), final=True)
        self.assertEqual(result.duration_s, .01)
        self.assertEqual(result.text, "hello")
        self.assertTrue(result.is_final)
        self.assertIsNone(result.clarity)
        self.assertEqual(result.words, [])
        self.assertFalse(self.engine._active)

    def test_failed_helper_reloads_on_the_next_stream(self):
        self.engine.start_stream()
        self.helpers[0].fail = "audio"
        with self.assertRaises(RuntimeError):
            self.engine.stream(np.zeros(512))
        self.assertTrue(self.helpers[0]._closed)
        self.engine.start_stream()
        self.assertEqual(len(self.helpers), 2)
        self.assertEqual([call[0] for call in self.helpers[1].calls], ["load", "start"])
        self.assertEqual(self.engine.stream(np.zeros(512), final=True).text, "hello")

    def test_failed_reset_closes_helper_and_allows_restart(self):
        self.engine.start_stream()
        self.helpers[0].fail = "reset"
        self.engine.reset_stream()
        self.assertTrue(self.helpers[0]._closed)
        self.engine.start_stream()
        self.assertEqual(len(self.helpers), 2)

    def test_model_provisioning_is_an_explicit_action_with_indeterminate_progress(self):
        progress = Mock()
        asr_native.download("windows-speech", progress)
        self.assertEqual([call[0] for call in self.helpers[-1].calls], ["prepare"])
        self.assertTrue(self.helpers[-1]._closed)
        self.assertEqual(progress.call_args_list[0].args, (0, 0))
        self.assertEqual(progress.call_args_list[-1].args, (1, 1))

    def test_failed_model_setup_invalidates_cached_os_readiness(self):
        helper = FakeService()
        helper.fail = "prepare"
        with patch.object(asr_native, "NativeService", return_value=helper), \
             patch.object(asr_native, "clear_availability") as clear:
            with self.assertRaisesRegex(RuntimeError, "helper unavailable"):
                asr_native.download("windows-speech")
        clear.assert_called_once()
        self.assertTrue(helper._closed)

    def test_readiness_is_rechecked_after_windows_changes_it(self):
        asr_native.clear_availability()
        self.addCleanup(asr_native.clear_availability)
        with patch.object(asr_native.time, "monotonic", return_value=0), \
             patch.object(asr_native, "service_path", return_value=Path("helper.exe")):
            asr_native.availability("windows-speech")
            asr_native.availability("windows-speech")
        self.assertEqual(len(self.helpers), 2)  # engine helper and one shared probe
        with patch.object(asr_native.time, "monotonic", return_value=16), \
             patch.object(asr_native, "service_path", return_value=Path("helper.exe")):
            asr_native.availability("windows-speech")
        self.assertEqual(len(self.helpers), 3)


class NativeArchitectureTests(unittest.TestCase):
    def test_snapdragon_uses_arm_helper_even_from_x64_python(self):
        with patch.object(asr_native.sys, "platform", "win32"), \
             patch("server.hardware.native_machine", return_value="ARM64", create=True), \
             patch.object(Path, "is_file", return_value=True):
            path = asr_native.service_path("windows-speech")
        self.assertIn("arm64", path.parts)


if __name__ == "__main__":
    unittest.main()
