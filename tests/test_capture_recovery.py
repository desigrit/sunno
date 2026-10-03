"""Deterministic capture lifecycle, IPC, format and pipeline continuity checks.

Run directly. SUNNO_TEST_NATIVE_AUDIO=1 also exercises real Windows enumeration and
a short loopback capture in memory. No microphone, model download or saved audio.
"""
from __future__ import annotations

import contextlib
import asyncio
import io
import os
from pathlib import Path
import queue
import subprocess
import socket
import sys
import time
import types
import unittest
import json
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tests._isolate  # noqa: F401,E402

import numpy as np

from server.capture import CaptureManager, CaptureProcess
from server.capture_target import AudioTarget
from server.config import FRAME_SAMPLES, Settings
from server.coreaudio import CaptureError, EndpointStream, decode_packet
from server.engine import Transcript
from server.pipeline import CaptionPipeline, SessionController
from server.recorder import Recorder


def target(identity="a", kind="microphone", follow=False):
    return AudioTarget(kind, identity, "Test input", 0, follow)


def failure(code="device_unavailable", retryable=True):
    return {"code": code, "retryable": retryable, "message": "Test input unavailable."}


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now

    def advance(self, seconds=5):
        self.now += seconds


class Worker:
    def __init__(self, selection, probe=False, clock=None):
        self.target, self.probe = selection, probe
        self.ready = self.failure = None
        self.closed = False
        self.audio = queue.Queue()

    def poll_failure(self):
        return self.failure

    def frame(self):
        try:
            return self.audio.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        self.closed = True


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.controller = SessionController()
        self.clock = Clock()
        self.workers, self.events = [], []
        self.default = "a"

        def factory(*args, **kwargs):
            worker = Worker(*args, **kwargs)
            self.workers.append(worker)
            return worker

        self.manager = CaptureManager(target(), self.controller, self.events.append,
                                      "test-model", factory, self.clock, lambda _: self.default)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.controller.shutdown()
        self.manager.step()
        self.output.__exit__(None, None, None)

    def start(self):
        self.manager.step()
        worker = self.workers[-1]
        worker.ready = worker.target
        self.manager.step()
        return worker

    def test_confirm_before_commit(self):
        self.manager.step()
        self.assertIsNone(self.manager.committed)
        self.assertFalse(self.manager.snapshot()["committed"])
        self.workers[-1].ready = target()
        self.manager.step()
        self.assertEqual(self.manager.committed, target())
        self.assertTrue(self.manager.snapshot()["running"])

    def test_make_before_break(self):
        old = self.start()
        old.audio.put(np.ones(FRAME_SAMPLES, dtype=np.float32))
        frames = self.manager.frames(self.manager.session)
        self.manager.select(target("b"), "switch")
        self.assertEqual(next(frames).shape, (FRAME_SAMPLES,))
        new = self.workers[-1]
        self.assertIs(self.manager.active, old)
        self.assertFalse(old.closed)
        new.ready = target("b")
        self.manager.step()
        self.assertTrue(old.closed)
        self.assertIs(self.manager.active, new)
        self.assertEqual(self.manager.snapshot()["request_id"], "switch")
        self.assertIsNone(next(frames, None))

    def test_failed_switch_keeps_healthy_input(self):
        old = self.start()
        self.manager.select(target("missing"), "switch")
        self.manager.step()
        self.workers[-1].failure = failure()
        self.manager.step()
        self.assertIs(self.manager.active, old)
        self.assertFalse(old.closed)
        self.assertEqual(self.manager.target, target())
        self.assertEqual(self.manager.snapshot()["state"], "failed")
        self.assertTrue(self.manager.snapshot()["running"])
        self.assertTrue(self.manager.snapshot()["committed"])
        self.clock.advance(100)
        self.manager.step()
        self.assertEqual(len(self.workers), 2)

    def test_successful_handover_drops_overlapping_candidate_backlog(self):
        self.start()
        self.manager.select(target("b"), "switch")
        self.manager.step()
        new = self.workers[-1]
        for value in range(3):
            new.audio.put(np.full(FRAME_SAMPLES, value, np.float32))
        new.ready = new.target
        self.manager.step()
        self.assertEqual(new.audio.qsize(), 1)
        self.assertEqual(new.frame()[0], 2)

    def test_latest_request_wins(self):
        old = self.start()
        self.manager.select(target("b"), "older")
        self.manager.step()
        obsolete = self.workers[-1]
        obsolete.ready = target("b")
        self.manager.select(target("c"), "latest")
        self.manager.step()
        self.assertTrue(obsolete.closed)
        self.assertIs(self.manager.active, old)
        self.workers[-1].ready = target("c")
        self.manager.step()
        self.assertEqual(self.manager.target.endpoint_id, "c")
        self.assertEqual(self.manager.snapshot()["request_id"], "latest")

    def test_reconnecting_client_replays_same_request_without_reopening_input(self):
        self.start()
        selection = target("b")
        self.manager.select(selection, "same-request")
        self.manager.step()
        worker = self.workers[-1]
        self.manager.select(selection, "same-request")
        self.assertIs(self.manager.candidate, worker)
        worker.ready = selection
        self.manager.step()
        self.manager.select(selection, "same-request")
        self.manager.step()
        self.assertIs(self.manager.active, worker)
        self.assertEqual(len(self.workers), 2)
        self.assertEqual(self.events[-1]["request_id"], "same-request")

    def test_request_ids_cannot_be_reused_for_different_targets(self):
        self.manager.select(target("b"), "request")
        with self.assertRaises(ValueError):
            self.manager.select(target("c"), "request")

    def test_unplug_recovers_without_changing_user_intent(self):
        old = self.start()
        old.failure = failure()
        self.manager.step()
        self.assertTrue(old.closed)
        self.assertIsNone(self.manager.active)
        self.assertTrue(self.controller.is_running)
        self.clock.advance(.25)
        replacement = self.start()
        self.assertIs(self.manager.active, replacement)
        self.assertEqual(replacement.target.endpoint_id, "a")

    def test_old_failure_does_not_cancel_pending_switch(self):
        old = self.start()
        self.manager.select(target("b"), "switch")
        self.manager.step()
        new = self.workers[-1]
        old.failure = failure()
        new.ready = target("b")
        self.manager.step()
        self.assertIs(self.manager.active, new)
        self.assertFalse(new.closed)

    def test_retry_backoff_is_bounded(self):
        delays = []
        for _ in range(8):
            self.manager.step()
            self.workers[-1].failure = failure()
            self.manager.step()
            delays.append(self.manager._retry_at - self.clock())
            count = len(self.workers)
            self.clock.advance(.1)
            self.manager.step()
            self.assertEqual(count, len(self.workers))
            self.clock.advance(5)
        self.assertEqual(delays, [.25, .5, 1, 2, 5, 5, 5, 5])
        self.assertEqual(self.manager.snapshot()["state"], "waiting")

    def test_access_denied_waits_for_explicit_retry(self):
        self.manager.step()
        self.workers[-1].failure = failure("capture_denied", False)
        self.manager.step()
        self.assertEqual(self.manager.snapshot()["state"], "blocked")
        self.clock.advance(100)
        self.manager.step()
        self.assertEqual(len(self.workers), 1)
        self.manager.devices_changed()  # An unrelated audio event cannot defeat the denial.
        self.manager.step()
        self.assertEqual(len(self.workers), 1)
        self.manager.retry()
        self.manager.step()
        self.assertEqual(len(self.workers), 2)

    def test_spawn_failure_does_not_stop_recognition_service(self):
        self.manager.factory = lambda *a, **k: (_ for _ in ()).throw(OSError("spawn"))
        self.manager.step()
        self.assertEqual(self.manager.snapshot()["code"], "capture_spawn")
        self.assertTrue(self.controller.is_running)

    def test_pause_releases_capture_without_stream_reopening(self):
        old = self.start()
        self.controller.pause()
        for _ in range(5):
            self.manager.step()
            self.clock.advance(5)
        self.assertTrue(old.closed)
        self.assertFalse(self.manager.snapshot()["wanted"])
        self.assertEqual(len(self.workers), 1)

    def test_pause_acknowledges_intent_during_blocked_or_backoff_recovery(self):
        for retryable in (True, False):
            with self.subTest(retryable=retryable):
                self.manager.step()
                self.workers[-1].failure = failure("capture_denied", retryable)
                self.manager.step()
                self.controller.pause()
                self.manager.step()
                self.assertFalse(self.manager.snapshot()["wanted"])
                self.assertEqual(self.manager.snapshot()["state"], "paused")
                self.controller.start()
                self.manager.retry()

    def test_selection_while_paused_only_probes_metadata(self):
        self.controller.pause()
        self.manager.select(target("b"), "paused-switch")
        worker = self.start()
        self.assertTrue(worker.probe)
        self.assertTrue(worker.closed)
        self.assertIsNone(self.manager.active)
        self.assertFalse(self.controller.is_running)
        self.assertEqual(self.manager.snapshot()["state"], "selected")
        self.controller.start()
        active = self.start()
        self.assertFalse(active.probe)
        self.assertEqual(active.target.endpoint_id, "b")

    def test_pause_during_switch_cannot_ack_unvalidated_target(self):
        self.start()
        self.manager.select(target("b"), "switch")
        self.manager.step()
        streaming = self.workers[-1]
        self.controller.pause()
        self.manager.step()
        self.assertTrue(streaming.closed)
        self.assertTrue(self.workers[-1].probe)
        self.assertFalse(self.manager.snapshot()["committed"])
        self.assertEqual(self.manager.committed.endpoint_id, "a")
        self.workers[-1].ready = target("b")
        self.manager.step()
        self.assertEqual(self.manager.committed.endpoint_id, "b")
        self.assertFalse(self.manager.snapshot()["running"])

    def test_resume_cancels_metadata_probe_and_waits_for_real_audio(self):
        self.controller.pause()
        self.manager.step()
        probe = self.workers[-1]
        probe.ready = target()
        self.controller.start()
        self.manager.step()
        self.assertTrue(probe.closed)
        self.assertFalse(self.workers[-1].probe)
        self.assertIsNone(self.manager.active)
        self.assertFalse(self.manager.snapshot()["committed"])

    def test_default_changes_make_before_break(self):
        self.manager.target = target(follow=True)
        old = self.start()
        self.default = "b"
        self.manager.devices_changed()
        self.manager.step()
        new = self.workers[-1]
        self.assertIs(self.manager.active, old)
        new.ready = target("b", follow=True)
        self.manager.step()
        self.assertIs(self.manager.active, new)
        self.assertTrue(old.closed)
        self.assertTrue(self.manager.target.follow_default)

    def test_default_changes_while_paused_only_refresh_metadata(self):
        self.manager.target = target(follow=True)
        self.start()
        self.controller.pause()
        self.manager.step()
        self.manager.devices_changed()
        self.manager.step()
        probe = self.workers[-1]
        self.assertTrue(probe.probe)
        probe.ready = target("new-default", follow=True)
        self.manager.step()
        self.assertEqual(self.manager.committed.endpoint_id, "new-default")
        self.assertIsNone(self.manager.active)
        self.assertFalse(self.manager.snapshot()["wanted"])

    def test_failed_default_change_keeps_retrying_with_old_capture(self):
        self.manager.target = target(follow=True)
        old = self.start()
        self.default = "b"
        self.manager.devices_changed()
        self.manager.step()
        self.workers[-1].failure = failure()
        self.manager.step()
        self.assertIs(self.manager.active, old)
        self.assertEqual(self.manager.snapshot()["state"], "recovering")
        self.clock.advance(5)
        self.manager.step()
        self.assertEqual(len(self.workers), 3)

    def test_pinned_selection_does_not_follow_default(self):
        old = self.start()
        self.default = "b"
        self.manager.devices_changed()
        self.manager.step()
        self.assertIs(self.manager.active, old)
        self.assertEqual(len(self.workers), 1)

    def test_new_follow_default_request_can_roll_back(self):
        old = self.start()
        self.default = "b"
        self.manager.select(target(follow=True), "default-request")
        self.manager.step()
        self.workers[-1].failure = failure()
        self.manager.step()
        self.assertIs(self.manager.active, old)
        self.assertFalse(self.manager.target.follow_default)
        self.assertEqual(self.manager.snapshot()["state"], "failed")

    def test_device_arrival_bypasses_retry_delay(self):
        self.manager.step()
        self.workers[-1].failure = failure()
        self.manager.step()
        self.manager.devices_changed()
        self.manager.step()
        self.assertEqual(len(self.workers), 2)

    def test_shutdown_closes_active_and_candidate(self):
        old = self.start()
        self.manager.select(target("b"), "switch")
        self.manager.step()
        new = self.workers[-1]
        self.controller.shutdown()
        self.manager.step()
        self.assertTrue(old.closed and new.closed)


class IdentityAndFormatTests(unittest.TestCase):
    devices = [
        {"index": 7, "name": "USB input", "endpoint_id": "a", "loopback": False,
         "is_default_input": True},
        {"index": 0, "name": "Other input", "endpoint_id": "b", "loopback": False},
        {"index": 7, "name": "USB input", "endpoint_id": "out", "loopback": True,
         "is_default_output": True},
    ]

    def test_identity_is_authoritative_over_stale_index_and_name(self):
        selected = AudioTarget("microphone", "a", "Other input", 0).resolve(self.devices)
        self.assertEqual(selected.endpoint_id, "a")
        self.assertEqual(selected.index, 7)

    def test_missing_identity_never_falls_back_to_different_device(self):
        with self.assertRaises(CaptureError):
            AudioTarget("microphone", "missing", "USB input", 7).resolve(self.devices)

    def test_legacy_name_precedes_index(self):
        selected = AudioTarget("microphone", name="USB input", index=0).resolve(self.devices)
        self.assertEqual(selected.endpoint_id, "a")

    def test_legacy_index_resolves_in_same_worker(self):
        old = [{"index": 90, "name": "USB input"}]
        selected = AudioTarget(index=90).resolve(self.devices, old)
        self.assertEqual(selected.endpoint_id, "a")

    def test_legacy_index_cannot_be_reinterpreted_as_a_native_collection_index(self):
        with self.assertRaises(CaptureError):
            AudioTarget(index=0).resolve(self.devices, [])

    def test_resource_string_names_migrate_from_cleaned_display_names(self):
        devices = [{"index": 0, "endpoint_id": "resource",
                    "name": "Headset (@System32/driver.sys,#2;%1 Hands-Free%0 ;(Test device))",
                    "loopback": False}]
        selection = AudioTarget(name="Headset (Test device)").resolve(devices)
        self.assertEqual(selection.endpoint_id, "resource")

    def test_duplicate_names_need_explicit_identity(self):
        devices = self.devices + [dict(self.devices[0], endpoint_id="duplicate")]
        with self.assertRaises(CaptureError) as caught:
            AudioTarget(name="USB input").resolve(devices)
        self.assertEqual(caught.exception.code, "device_ambiguous")
        self.assertFalse(caught.exception.retryable)

    def test_default_is_separate_for_input_and_output(self):
        self.assertEqual(AudioTarget(follow_default=True).resolve(self.devices).endpoint_id, "a")
        self.assertEqual(AudioTarget("loopback", follow_default=True).resolve(self.devices).endpoint_id, "out")

    def test_selection_validation(self):
        for invalid in (None, [], {"kind": "file"}, {"index": True},
                        {"name": 42}, {"endpoint_id": "x" * 1025}, {"follow_default": "yes"}):
            with self.subTest(invalid=type(invalid)):
                with self.assertRaises(ValueError):
                    AudioTarget.from_dict(invalid)

    def test_mix_formats_downmix_without_changing_level(self):
        for tag, bits, dtype, values in (
            (3, 32, "<f4", [.5, -.5, .25, .75]),
            (1, 16, "<i2", [16384, -16384, 8192, 24576]),
            (1, 32, "<i4", [1073741824, -1073741824, 536870912, 1610612736]),
        ):
            with self.subTest(bits=bits, tag=tag):
                result = decode_packet(np.array(values, dtype=dtype).tobytes(), tag, bits, 2)
                np.testing.assert_allclose(result, [0, .5])
                self.assertEqual(result.dtype, np.float32)

    def test_signed_24_bit_pcm(self):
        result = decode_packet(bytes.fromhex("000080000000ffff7f"), 1, 24, 1)
        np.testing.assert_allclose(result, [-1, 0, 1 - 1 / 8388608])

    def test_invalid_float_samples_are_sanitized(self):
        result = decode_packet(np.array([float("nan"), float("inf")], "<f4").tobytes(), 3, 32, 1)
        self.assertTrue(np.isfinite(result).all())

    def test_unsupported_format_is_not_retried_forever(self):
        with self.assertRaises(CaptureError) as caught:
            decode_packet(bytes(8), 3, 64, 1)
        self.assertFalse(caught.exception.retryable)

    def test_failed_native_open_balances_com_and_releases_every_pointer(self):
        calls = []
        @contextlib.contextmanager
        def apartment():
            calls.append("init")
            try:
                yield
            finally:
                calls.append("uninit")
        with patch("server.coreaudio.apartment", apartment), \
             patch("server.coreaudio._enumerator", return_value="enumerator"), \
             patch("server.coreaudio._call", side_effect=CaptureError("capture_denied", "Denied", False)), \
             patch("server.coreaudio._release", side_effect=lambda ptr: calls.append(ptr) if ptr else None):
            stream = EndpointStream("test", "microphone")
            with self.assertRaises(CaptureError):
                stream.__enter__()
        self.assertEqual(calls, ["init", "enumerator", "uninit"])
        self.assertIsNone(stream._device)

    def test_enumeration_survives_endpoint_disappearance_and_missing_name(self):
        import ctypes
        from server.coreaudio import list_endpoints
        def call(ptr, slot, kinds, *args):
            if ptr.value == 100:
                args[-1]._obj.value = 1000
            elif slot == 3:
                args[-1]._obj.value = 3
            else:
                args[-1]._obj.value = 200 + args[0]
        def identity(ptr):
            if ptr.value == 200:
                raise CaptureError("device_unavailable", "Removed")
            return str(ptr.value)
        def name(ptr):
            if ptr.value == 201:
                raise CaptureError("device_unavailable", "Missing property")
            return "Test input"
        with patch("server.coreaudio.apartment", contextlib.nullcontext), \
             patch("server.coreaudio._enumerator", return_value=ctypes.c_void_p(100)), \
             patch("server.coreaudio._default_id", return_value="202"), \
             patch("server.coreaudio._call", side_effect=call), \
             patch("server.coreaudio._release"), \
             patch("server.coreaudio._device_id", side_effect=identity), \
             patch("server.coreaudio._device_name", side_effect=name):
            devices = list_endpoints("microphone")
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0]["name"], "Audio device")
        self.assertEqual(devices[1]["endpoint_id"], "202")
        self.assertTrue(devices[1]["is_default_input"])

    def test_idle_loopback_reports_healthy_silence_and_checks_endpoint(self):
        stream = EndpointStream("test", "loopback")
        stream.capture_rate = 16000
        stream._client, stream._capture = "client", "capture"
        with patch("server.coreaudio._call") as call, \
             patch("server.coreaudio.time.monotonic", side_effect=[100, 100.064]):
            frames = stream.frames(lambda: True)
            frame = next(frames)
            frames.close()
        self.assertEqual(frame.shape, (FRAME_SAMPLES,))
        self.assertEqual(np.count_nonzero(frame), 0)
        self.assertTrue(any(args.args[:2] == ("client", 6) for args in call.call_args_list))

    def test_idle_loopback_does_not_hide_endpoint_invalidation(self):
        stream = EndpointStream("test", "loopback")
        stream.capture_rate = 16000
        stream._client, stream._capture = "client", "capture"
        def call(ptr, slot, *args):
            if ptr == "client":
                raise CaptureError("device_unavailable", "Removed")
        with patch("server.coreaudio._call", side_effect=call):
            with self.assertRaises(CaptureError):
                next(stream.frames(lambda: True))


class LegacyCleanupTests(unittest.TestCase):
    def test_full_microphone_queue_never_blocks_exit(self):
        from server.audio import MicrophoneStream
        stream = MicrophoneStream(0)
        while not stream._queue.full():
            stream._queue.put_nowait(np.zeros(FRAME_SAMPLES, np.float32))
        started = time.monotonic()
        stream.__exit__()
        self.assertLess(time.monotonic() - started, .5)
        self.assertIsNone(next(stream.frames(), None))

    def test_dead_microphone_does_not_wait_forever_for_callbacks(self):
        from server.audio import MicrophoneStream
        stream = MicrophoneStream(0)
        stream._stream = types.SimpleNamespace(active=False)
        with self.assertRaises(RuntimeError):
            next(stream.frames(lambda: True))

    def test_loopback_closes_even_if_stop_fails_and_queue_is_full(self):
        from server.loopback import LoopbackStream
        stream = LoopbackStream(0)
        closed, terminated = [], []
        stream._stream = types.SimpleNamespace(
            stop_stream=lambda: (_ for _ in ()).throw(OSError("removed")),
            close=lambda: closed.append(True))
        stream._audio = types.SimpleNamespace(terminate=lambda: terminated.append(True))
        while not stream._queue.full():
            stream._queue.put_nowait(bytes(2048))
        stream.__exit__()
        self.assertEqual(closed, [True])
        self.assertEqual(terminated, [True])


class DeviceCacheTests(unittest.TestCase):
    def request(self, path):
        from server import app
        handler = app._UiRequestHandler.__new__(app._UiRequestHandler)
        handler.path = path
        responses = []
        handler._json = responses.append
        handler.do_GET()
        return responses[0]

    def test_first_enumeration_is_isolated_and_later_startup_calls_use_cache(self):
        from server import app
        devices = [{"index": 0, "name": "Test", "endpoint_id": "test"}]
        with patch.object(app._UiRequestHandler, "_devices_cache", None), \
             patch.object(app, "_fresh_devices", return_value=devices) as fresh:
            self.assertEqual(self.request("/devices.json")["devices"], devices)
            self.assertEqual(self.request("/devices.json")["devices"], devices)
            fresh.assert_called_once()

    def test_failed_refresh_serves_last_good_list_without_native_calls_in_parent(self):
        from server import app
        devices = [{"index": 0, "name": "Test", "endpoint_id": "test"}]
        with patch.object(app._UiRequestHandler, "_devices_cache", devices), \
             patch.object(app, "_fresh_devices", return_value=None):
            response = self.request("/devices.json?fresh=1")
        self.assertEqual(response["devices"], devices)
        self.assertTrue(response["stale"])

    def test_successful_empty_enumeration_replaces_old_hardware(self):
        from server import app
        with patch.object(app._UiRequestHandler, "_devices_cache", [{"index": 0}]), \
             patch.object(app, "_fresh_devices", return_value=[]):
            response = self.request("/devices.json?fresh=1")
            self.assertEqual(response, {"devices": []})
            self.assertEqual(app._UiRequestHandler._devices_cache, [])


class ProcessTests(unittest.TestCase):
    def peer(self, mode, clock=None):
        popen = subprocess.Popen
        def spawn(command, **kwargs):
            return popen([sys.executable, "-B", str(ROOT / "tests/capture_child_stub.py"), mode], **kwargs)
        with patch("server.capture.subprocess.Popen", side_effect=spawn):
            process = CaptureProcess(target(), clock=clock or time.monotonic)
        self.addCleanup(process.close)
        return process

    def wait_for(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.01)
        self.fail("Timed out waiting for capture protocol.")

    def test_unresponsive_driver_can_be_killed(self):
        peer = self.peer("hang")
        self.wait_for(lambda: peer.ready is not None)
        started = time.monotonic()
        peer.close()
        self.assertLess(time.monotonic() - started, 2)
        self.assertIsNotNone(peer.process.poll())
        self.assertFalse(peer.reader.is_alive())
        peer.close()  # Idempotent.

    def test_parent_death_after_stop_still_exits_a_hung_native_worker(self):
        peer = self.peer("worker_hang")
        self.wait_for(lambda: peer.ready is not None)
        peer.process.stdin.write(b"stop\n")
        peer.process.stdin.flush()
        time.sleep(.03)
        peer.process.stdin.close()
        peer.process.wait(timeout=2)
        self.assertEqual(peer.process.returncode, 0)

    def test_protocol_failure_and_unexpected_exit_are_recoverable(self):
        for mode in ("crash", "malformed", "oversized", "bad_audio", "nan_audio"):
            with self.subTest(mode=mode):
                peer = self.peer(mode)
                self.wait_for(lambda: peer.failure is not None)
                self.assertTrue(peer.poll_failure()["retryable"])
                peer.close()

    def test_missing_callback_heartbeat_triggers_recovery(self):
        clock = Clock()
        peer = self.peer("hang", clock)
        self.wait_for(lambda: peer.ready is not None)
        self.assertIsNone(peer.poll_failure())
        clock.advance(3.1)
        self.assertEqual(peer.poll_failure()["code"], "capture_timeout")

    def test_audio_queue_never_blocks_child_or_close(self):
        peer = self.peer("overflow")
        self.wait_for(lambda: peer.audio.qsize() == 16)
        self.assertEqual(peer.frame().shape, (FRAME_SAMPLES,))
        peer.close()
        self.assertIsNotNone(peer.process.poll())

    def test_initial_open_has_a_deadline_without_any_protocol_message(self):
        peer = CaptureProcess.__new__(CaptureProcess)
        peer.failure = peer.ready = None
        peer.created = 0
        peer.clock = lambda: 6.1
        self.assertEqual(peer.poll_failure()["code"], "capture_timeout")


class PipelineContinuityTests(unittest.TestCase):
    def test_switches_keep_one_asr_worker_utterance_ids_speaker_and_recording(self):
        class VAD:
            def __init__(self, *args): pass
            def reset(self): pass
            def __call__(self, frame): return 1

        settings = Settings(start_frames=1, min_utterance_ms=32, min_partial_ms=100000)
        engine = types.SimpleNamespace(settings=settings)
        engine.final = lambda audio: Transcript("test caption", len(audio) / 16000, 1, True)
        engine.partial = lambda audio: Transcript("partial", len(audio) / 16000, 1, False)
        speaker = types.SimpleNamespace(
            identify=lambda audio: (1, 1.), label=lambda _: "Test speaker",
            roster=lambda: [{"id": 1, "label": "Test speaker", "is_self": False}])
        recorder = Recorder(Path(tests._isolate.DATA_DIR) / "capture-continuity")
        self.addCleanup(recorder.detach)
        events = []
        def emit(event):
            events.append(event)
            if event.get("type") == "final":
                recorder.add_line(event)
        with patch("server.vad.StreamingSileroVAD", VAD):
            pipeline = CaptionPipeline(settings, engine, emit, speaker=speaker, on_audio=recorder.add_audio)
        self.addCleanup(pipeline.close)
        asr = pipeline._worker
        with patch("server.hardware.record_latency"):
            for _ in range(3):
                pipeline.run([np.full(FRAME_SAMPLES, .01, np.float32) for _ in range(4)])
                self.assertTrue(pipeline.drain(3))
        finals = [event for event in events if event.get("type") == "final"]
        self.assertEqual([event["id"] for event in finals], [1, 2, 3])
        self.assertTrue(all(event["speaker_id"] == 1 for event in finals))
        self.assertIs(pipeline._worker, asr)
        self.assertTrue(asr._thread.is_alive())
        self.assertIs(pipeline._speaker, speaker)
        self.assertEqual(recorder.elapsed_s, 12 * FRAME_SAMPLES / 16000)


class ServerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_websocket_switch_pause_recording_and_reconnect(self):
        from websockets.asyncio.client import connect
        from server import app, capture, engine, hardware, models
        managers, pipelines = [], []

        class ReadyWorker(Worker):
            def __init__(self, selection, **kwargs):
                super().__init__(selection, **kwargs)
                self.ready = selection
            def frame(self):
                time.sleep(.005)
                return np.zeros(FRAME_SAMPLES, np.float32)

        class Manager(CaptureManager):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, factory=ReadyWorker, default_id=lambda _: "a", **kwargs)
                managers.append(self)

        class Pipeline:
            def __init__(self, settings, model, emit, should_run, speaker, on_audio):
                self.on_audio = on_audio
                self.closed = False
                pipelines.append(self)
            def run(self, frames):
                for frame in frames:
                    self.on_audio(frame)
            def close(self): self.closed = True
            def stop(self): pass

        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        settings = Settings(device="cpu", enable_speakers=False, ws_port=port,
                            input_endpoint_id="a", recordings_path=str(Path(tests._isolate.DATA_DIR) / "integration"))
        args = types.SimpleNamespace(no_speakers=True, start_stopped=False, resume_recording=None,
                                     wav=None, compute_device="cpu", compute_type=None,
                                     echo_transcript=False)
        model = types.SimpleNamespace(warmup=lambda: 0)
        with contextlib.ExitStack() as stack:
            for obj, name, value in (
                (capture, "CaptureManager", Manager),
                (app, "CaptionPipeline", Pipeline), (app, "_serve_ui", lambda *a: None),
                (engine, "create_engine", lambda *a: model),
                (hardware, "engine_importable", lambda: True),
                (hardware, "gpu_capable", lambda: False),
                (models, "is_available", lambda *a: types.SimpleNamespace(available=True)),
            ):
                stack.enter_context(patch.object(obj, name, value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            task = asyncio.create_task(app.run(settings, args))
            ws = None
            async def receive(kind, **fields):
                async with asyncio.timeout(5):
                    while True:
                        message = json.loads(await ws.recv())
                        if message.get("type") == kind and all(message.get(k) == v for k, v in fields.items()):
                            return message
            try:
                for _ in range(100):
                    try:
                        ws = await connect(f"ws://127.0.0.1:{port}")
                        break
                    except OSError:
                        await asyncio.sleep(.02)
                self.assertIsNotNone(ws)
                await receive("input", state="ready")
                await ws.send("[]")  # A malformed command cannot drop the connection.
                await ws.send(json.dumps({"cmd": "start_recording"}))
                recording = await receive("recording", state="recording")
                await ws.send(json.dumps({"cmd": "set_input", "request_id": "switch",
                                          "target": target("b").to_dict()}))
                ready = await receive("input", request_id="switch", state="ready")
                self.assertEqual(ready["target"]["endpoint_id"], "b")
                self.assertEqual(len(pipelines), 1)
                await ws.send(json.dumps({"cmd": "stop"}))
                await receive("input", state="selected", wanted=False)
                self.assertIsNone(managers[0].active)
                await ws.send(json.dumps({"cmd": "set_input", "request_id": "paused",
                                          "target": target("c").to_dict()}))
                selected = await receive("input", state="selected", request_id="paused")
                self.assertFalse(selected["running"])
                await ws.close()
                ws = await connect(f"ws://127.0.0.1:{port}")
                replay = await receive("input", state="selected", request_id="paused")
                self.assertFalse(replay["wanted"])
                resumed_recording = await receive("recording", state="recording")
                self.assertEqual(recording["folder"], resumed_recording["folder"])
                await ws.send(json.dumps({"cmd": "start"}))
                await receive("input", state="ready", wanted=True)
                self.assertEqual(managers[0].active.ready.endpoint_id, "c")
                self.assertEqual(len(pipelines), 1)
            finally:
                if ws is not None:
                    await ws.close()
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                for _ in range(100):
                    if pipelines and pipelines[0].closed:
                        break
                    await asyncio.sleep(.01)
                self.assertTrue(pipelines[0].closed)


@unittest.skipUnless(sys.platform == "win32" and os.environ.get("SUNNO_TEST_NATIVE_AUDIO") == "1",
                     "opt-in Windows hardware smoke test")
class NativeTests(unittest.TestCase):
    wait_for = ProcessTests.wait_for
    def test_real_default_loopback_and_cleanup(self):
        process = CaptureProcess(AudioTarget("loopback", follow_default=True))
        self.addCleanup(process.close)
        self.wait_for(lambda: process.ready is not None or process.failure is not None)
        self.assertIsNone(process.poll_failure())
        self.assertTrue(process.ready.follow_default)
        self.assertIsNotNone(process.ready.endpoint_id)
        self.wait_for(lambda: process.frame() is not None)
        process.close()
        self.assertIsNotNone(process.process.poll())
        self.assertFalse(process.reader.is_alive())

    def test_paused_microphone_probe_never_opens_capture(self):
        process = CaptureProcess(AudioTarget(follow_default=True), probe=True)
        self.addCleanup(process.close)
        self.wait_for(lambda: process.ready is not None or process.failure is not None)
        self.assertIsNone(process.poll_failure())
        self.assertTrue(process.audio.empty())
        self.wait_for(lambda: process.process.poll() is not None)

    def test_parent_eof_releases_capture_even_without_stop_command(self):
        process = CaptureProcess(AudioTarget("loopback", follow_default=True))
        self.addCleanup(process.close)
        self.wait_for(lambda: process.ready is not None or process.failure is not None)
        self.assertIsNone(process.poll_failure())
        process.process.stdin.close()
        self.wait_for(lambda: process.process.poll() is not None)


if __name__ == "__main__":
    unittest.main()
