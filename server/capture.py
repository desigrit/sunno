"""Supervise replaceable audio children while recognition and recording stay alive."""
from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .capture_target import AudioTarget
from .config import FRAME_SAMPLES


class CaptureProcess:
    def __init__(self, target, probe=False, clock=time.monotonic):
        self.target, self.probe, self.clock = target, probe, clock
        self.created = self.last_audio = clock()
        self.ready = self.failure = None
        self.closed = False
        self.audio = queue.Queue(maxsize=16)
        root = str(Path(__file__).resolve().parents[1])
        env = dict(os.environ)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUTF8"] = env["PYTHONUNBUFFERED"] = "1"
        env["SUNNO_AUDIO_CHILD"] = "1"
        command = [sys.executable, "-u", "-m", "server.capture_worker", json.dumps(target.to_dict())]
        if probe:
            command.append("--probe")
        self.process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.reader = threading.Thread(target=self._read, daemon=True, name="capture-reader")
        self.reader.start()

    def _read(self):
        try:
            while not self.closed:
                raw = self.process.stdout.readline(16385)
                if not raw:
                    break
                if len(raw) > 16384:
                    raise ValueError("oversized capture frame")
                msg = json.loads(raw)
                if msg.get("type") == "ready":
                    self.ready = AudioTarget.from_dict(msg["target"])
                    self.last_audio = self.clock()
                elif msg.get("type") == "error":
                    self.failure = msg
                    break
                elif msg.get("type") == "audio":
                    data = base64.b64decode(msg["data"], validate=True)
                    if len(data) != FRAME_SAMPLES * 4:
                        raise ValueError("invalid capture frame")
                    self.last_audio = self.clock()
                    frame = np.frombuffer(data, dtype="<f4").copy()
                    if not np.isfinite(frame).all():
                        raise ValueError("non-finite capture frame")
                    try:
                        self.audio.put_nowait(frame)
                    except queue.Full:
                        # Preserve live latency under inference backpressure.
                        try:
                            self.audio.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.audio.put_nowait(frame)
                        except queue.Full:
                            pass
        except Exception:
            self.failure = {"code": "capture_protocol", "retryable": True,
                            "message": "The audio connection was interrupted. Sunno will reconnect."}
        finally:
            if not self.closed and not self.failure and not (self.probe and self.ready):
                self.failure = {"code": "capture_exited", "retryable": True,
                                "message": "The audio input stopped. Sunno will reconnect."}

    def poll_failure(self):
        if self.failure:
            return self.failure
        now = self.clock()
        if (self.ready is None and now - self.created > 6) or (
                not self.probe and self.ready is not None and now - self.last_audio > 3):
            return {"code": "capture_timeout", "retryable": True,
                    "message": "The audio input did not respond. Sunno is reconnecting."}
        return None

    def frame(self):
        try:
            return self.audio.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write(b"stop\n")
                    self.process.stdin.flush()
                    self.process.wait(timeout=0.15)
                except (OSError, subprocess.TimeoutExpired):
                    self.process.kill()
                    self.process.wait(timeout=1)
        finally:
            for handle in (self.process.stdin, self.process.stdout):
                if handle is not None:
                    handle.close()
            self.reader.join(timeout=0.2)


class CaptureManager:
    def __init__(self, target, controller, emit, model, factory=CaptureProcess,
                 clock=time.monotonic, default_id=None):
        self.target, self.controller, self.emit, self.model = target, controller, emit, model
        self.factory, self.clock = factory, clock
        if default_id is None:
            from .coreaudio import default_endpoint_id
            default_id = default_endpoint_id if sys.platform == "win32" else lambda kind: None
        self.default_id = default_id
        self._lock = threading.RLock()
        self.active = self.candidate = None
        self.committed = None
        self.request_id = "startup"
        self._requested_target = target
        self.revision = self.session = 0
        self._candidate_revision = -1
        self._paused_revision = -1
        self._attempts = 0
        self._retry_at = self._default_at = 0
        self._blocked = False
        self._blocked_code = None
        self._devices_changed = False
        self._last_signature = None
        self._latest = None
        self._automatic = False

    def select(self, target, request_id):
        with self._lock:
            if request_id == self.request_id:
                if target != self._requested_target:
                    raise ValueError("This input request was already used.")
                if self._latest is not None and self._latest["request_id"] == request_id:
                    self.emit(self._latest)
                return
            self._requested_target = target
            self.target, self.request_id = target, request_id
            self.revision += 1
            self._retry_at = 0
            self._attempts = 0
            self._blocked = self._automatic = False
            self._blocked_code = None

    def retry(self):
        with self._lock:
            self._blocked = False
            self._blocked_code = None
            self._retry_at = 0
            self._paused_revision = -1

    def devices_changed(self):
        with self._lock:
            self._devices_changed = True
            self._retry_at = 0
            if self._blocked_code not in ("capture_denied", "capture_dependency", "device_ambiguous"):
                self._blocked = False
            self._paused_revision = -1

    def snapshot(self):
        with self._lock:
            return self._latest

    def _announce(self, state, *, message=None, code=None, committed=False):
        running = self.active is not None and self.controller.is_running
        signature = (state, self.request_id, self.revision, self.controller.is_running,
                     running, committed, self.target, message, code)
        if signature == self._last_signature:
            return
        self._last_signature = signature
        active = self.active.ready.to_dict() if self.active and self.active.ready else None
        self._latest = {"type": "input", "state": state, "request_id": self.request_id,
                        "target": self.target.to_dict(), "active": active,
                        "wanted": self.controller.is_running, "running": running,
                        "committed": committed, "message": message, "code": code}
        self.emit(self._latest)
        self.emit({"type": "status", "state": "listening" if running else
                   "stopped" if not self.controller.is_running else state,
                   "running": running, "model": self.model,
                   "wanted": self.controller.is_running,
                   "device": self.active.ready.name if running else None})
        # Device names and IDs belong in the local UI, never persistent diagnostics.
        print(f"[capture] {state}; attempt {self._attempts}", flush=True)

    def _close(self, field):
        worker = getattr(self, field)
        setattr(self, field, None)
        if worker is not None:
            try:
                worker.close()
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _failed(self, failure):
        self._close("candidate")
        if self.active is not None and self.committed is not None and not self._automatic:
            self.target = self.committed
            self._candidate_revision = self.revision
            self._announce("failed", message="Sunno could not switch inputs. The previous input is still running.",
                           code=failure.get("code"), committed=True)
            return
        self._attempts += 1
        self._blocked = not failure.get("retryable", True)
        self._blocked_code = failure.get("code") if self._blocked else None
        delay = (0.25, 0.5, 1, 2, 5)[min(self._attempts - 1, 4)]
        self._retry_at = self.clock() + delay
        self._announce("blocked" if self._blocked else "waiting" if self._attempts >= 3 else "recovering",
                       message=failure.get("message"), code=failure.get("code"))

    def step(self):
        """Advance lifecycle once. Stream driver calls stay in disposable children."""
        with self._lock:
            if self.controller.is_shutdown:
                self._close("candidate")
                self._close("active")
                return
            wanted = self.controller.is_running
            if wanted and self.candidate is not None and self.candidate.probe:
                self._close("candidate")
            if self.active is not None:
                failure = self.active.poll_failure()
                if failure:
                    self._close("active")
                    self.session += 1
                    if self.candidate is None:
                        self._candidate_revision = -1
                        self._failed(failure)
                    else:
                        self._announce("recovering", message=failure.get("message"),
                                       code=failure.get("code"))
            if not wanted:
                if self.active is not None:
                    self._close("active")
                    self.session += 1
                if self.candidate is not None and not self.candidate.probe:
                    self._close("candidate")
                if (self.candidate is None and self.target == self.committed
                        and self._candidate_revision == self.revision
                        and not self._devices_changed and not self._automatic):
                    self._paused_revision = self.revision
                    self._announce("selected", committed=True)
                    return
                if self._latest is not None and self._latest["wanted"]:
                    self._announce("paused")
                if self._devices_changed:
                    self._devices_changed = False
                    self._paused_revision = -1
                if self._paused_revision == self.revision:
                    return
            if self.candidate is not None and self._candidate_revision != self.revision:
                self._close("candidate")
            if wanted and self.active and self.target.follow_default and self.candidate is None and (
                    self._candidate_revision == self.revision) and (
                    self._devices_changed or self.clock() >= self._default_at):
                self._default_at = self.clock() + 2
                self._devices_changed = False
                try:
                    identity = self.default_id(self.target.kind)
                except Exception:
                    identity = self.active.ready.endpoint_id
                if identity != self.active.ready.endpoint_id and self.candidate is None:
                    self._candidate_revision = -1
                    self._automatic = True
            if self.candidate is None:
                if self._blocked or self.clock() < self._retry_at:
                    return
                if wanted and self.active is not None and self._candidate_revision == self.revision:
                    return
                try:
                    self.candidate = self.factory(self.target, probe=not wanted, clock=self.clock)
                    self._candidate_revision = self.revision
                    self._announce("switching" if self.active else "recovering")
                except Exception:
                    self._failed({"code": "capture_spawn", "retryable": True,
                                  "message": "Sunno could not start audio capture. It will try again."})
                    return
            failure = self.candidate.poll_failure()
            if failure:
                if not wanted and not failure.get("retryable", True):
                    self._paused_revision = self.revision
                self._failed(failure)
                return
            if self.candidate.ready is not None:
                worker = self.candidate
                self.target = self.committed = worker.ready
                if worker.probe:
                    self._attempts = 0
                    self._close("candidate")
                    self._paused_revision = self.revision
                    self._announce("selected", committed=True)
                elif wanted:
                    had_active = self.active is not None
                    self._close("active")
                    if had_active:
                        # Candidate audio overlaps the source that was still captioning.
                        # Start from its freshest frame rather than replaying the overlap.
                        for _ in range(max(0, worker.audio.qsize() - 1)):
                            try:
                                worker.audio.get_nowait()
                            except queue.Empty:
                                break
                    self.active, self.candidate = worker, None
                    self.session += 1
                    self._attempts = 0
                    self._announce("ready", committed=True)
                self._automatic = False

    def frames(self, session):
        while self.controller.is_running and not self.controller.is_shutdown:
            self.step()
            if self.session != session or self.active is None:
                return
            frame = self.active.frame()
            if frame is None:
                time.sleep(0.01)
            else:
                yield frame

    def run(self, pipeline):
        try:
            while not self.controller.is_shutdown:
                self.step()
                if self.controller.is_running and self.active is not None:
                    pipeline.run(self.frames(self.session))
                else:
                    time.sleep(0.02)
        finally:
            self._close("candidate")
            self._close("active")
