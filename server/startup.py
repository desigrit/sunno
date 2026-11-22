"""Small, numeric startup diagnostics. No transcript, paths or device names."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import threading
import time


STAGES = frozenset({
    "imports_ready", "hardware", "control_ready", "cache_lookup", "model_load",
    "model_warmup", "pipeline_ready", "capture_ready", "speaker_load",
    "recording_recovery", "catalog", "correction_load", "correction_warmup",
})


class StartupTimings:
    def __init__(self, clock=time.perf_counter, write=print, *, enabled=True):
        self._clock = clock
        self._write = write
        self._started = clock()
        self._seen = set()
        self._lock = threading.Lock()
        self._enabled = enabled
        self._pending = []

    def enable(self):
        with self._lock:
            self._enabled = True
            for line in self._pending:
                self._write(line, flush=True)
            self._pending.clear()

    def _report(self, stage, started, result):
        if stage not in STAGES:
            raise ValueError("Unknown startup stage")
        now = self._clock()
        line = (f"[startup] stage={stage} duration_ms={max(0, round((now-started)*1000))} "
                f"elapsed_ms={max(0, round((now-self._started)*1000))} result={result}")
        if self._enabled:
            self._write(line, flush=True)
        else:
            self._pending.append(line)
            self._pending = self._pending[-32:]

    def mark(self, stage):
        with self._lock:
            if stage in self._seen:
                return
            self._seen.add(stage)
            self._report(stage, self._started, "ok")

    @contextmanager
    def measure(self, stage):
        started = self._clock()
        result = "failed"
        try:
            yield
            result = "ok"
        finally:
            with self._lock:
                self._report(stage, started, result)

    async def run(self, stage, function, *args, **kwargs):
        with self.measure(stage):
            return await asyncio.to_thread(function, *args, **kwargs)


# Commands such as --list-devices must keep machine-readable stdout clean.
startup = StartupTimings(enabled=False)
