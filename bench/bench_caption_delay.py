"""Measure live caption emission, not just the time spent inside a decoder.

Replay prerecorded 16 kHz speech in real time through Sunno's unchanged VAD,
audio conditioner and serial ASR worker. No microphone, recording, model download
or user-profile write is needed. Caption events are timed before transport/UI.
Speech boundaries are VAD estimates with 32 ms frame resolution, not phonetic
annotations. Speaker identification is disabled to isolate the speech pipeline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def distribution(values):
    present = [value for value in values if value is not None]
    return {
        "n": len(present),
        "median_ms": statistics.median(present) if present else None,
        "min_ms": min(present) if present else None,
        "max_ms": max(present) if present else None,
    }


def caption_rows(segments, events):
    """Match worker results to their own utterance, including final-only captions."""
    rows = []
    for segment in segments:
        captions = [event for event in events
                    if event.get("id") == segment["id"]
                    and event["type"] in ("partial", "final")
                    and str(event.get("text", "")).strip()]
        first = captions[0] if captions else None
        final = next((event for event in reversed(captions)
                      if event["type"] == "final"), None)
        previous, changed_partials = None, 0
        for event in captions:
            text = event["text"].strip()
            if event["type"] == "partial" and text != previous:
                changed_partials += 1
            previous = text
        onset, offset = segment["vad_onset_s"], segment.get("vad_offset_s")
        rows.append(dict(
            segment,
            first_caption_type=first["type"] if first else None,
            first_caption_delay_ms=(first["elapsed_s"] - onset) * 1000 if first else None,
            final_caption_delay_ms=(final["elapsed_s"] - offset) * 1000
            if final and offset is not None else None,
            final_decode_ms=final.get("latency_ms") if final else None,
            final_queue_ms=final.get("queue_ms") if final else None,
            changed_partial_updates=changed_partials,
        ))
    return rows


def measured_pipeline(settings, engine, emit):
    from server.config import FRAME_SAMPLES, SAMPLE_RATE
    from server.pipeline import CaptionPipeline

    class MeasuredPipeline(CaptionPipeline):
        # Only observe existing state transitions. Do not replace VAD probabilities,
        # ASR scheduling, endpoint rules or the production event callback.
        def reset_measurement(self):
            self.frame_count = 0
            self.segments = []

        def _process(self, frame):
            self.frame_count += 1
            super()._process(frame)

        def _begin_utterance(self):
            self.segments.append({
                "id": self._utterance_id + 1,
                "vad_onset_s": (self.frame_count - self.settings.start_frames)
                * FRAME_SAMPLES / SAMPLE_RATE,
            })
            super()._begin_utterance()

        def _finalise(self, continued=False):
            segment = self.segments[-1]
            segment["vad_offset_s"] = (self.frame_count - self._silence_run) \
                * FRAME_SAMPLES / SAMPLE_RATE
            segment["final_submitted_audio_s"] = self.frame_count * FRAME_SAMPLES / SAMPLE_RATE
            segment["finish_reason"] = (
                "max_duration" if continued else
                "silence_endpoint" if self._silence_run >= self._end_silence_frames else
                "source_end"
            )
            super()._finalise(continued)

    pipeline = MeasuredPipeline(settings, engine, emit, speaker=None)
    pipeline.reset_measurement()
    return pipeline


def replay(pipeline, audio, event_log, *, tail_s=2.0):
    import numpy as np
    from server.config import FRAME_SAMPLES, SAMPLE_RATE

    # A real silence tail lets Sunno detect the endpoint. EOF must not shortcut
    # its silence wait, as a direct engine.final() benchmark would.
    count = len(audio) + int(tail_s * SAMPLE_RATE)
    count = ((count + FRAME_SAMPLES - 1) // FRAME_SAMPLES) * FRAME_SAMPLES
    padded = np.zeros(count, np.float32)
    padded[:len(audio)] = audio
    pipeline.reset_measurement()
    lateness = []
    event_log.clear()
    origin = time.monotonic()

    def frames():
        for stop in range(FRAME_SAMPLES, count + 1, FRAME_SAMPLES):
            due = origin + stop / SAMPLE_RATE
            time.sleep(max(0.0, due - time.monotonic()))
            lateness.append(max(0.0, time.monotonic() - due) * 1000)
            yield padded[stop - FRAME_SAMPLES:stop]

    pipeline.run(frames())
    if not pipeline.drain(60):
        raise TimeoutError("Caption worker did not finish within 60 seconds")
    events = [dict(event, elapsed_s=event["wall_s"] - origin)
              for event in event_log]
    return {
        "audio_s": len(audio) / SAMPLE_RATE,
        "replay_s": time.monotonic() - origin,
        "feed_lateness_ms": distribution(lateness),
        "utterances": caption_rows(pipeline.segments, events),
        "events": events,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=("moonshine-tiny", "moonshine-small", "moonshine-medium"))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--clips", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--machine", required=True)
    args = parser.parse_args()
    if min(args.clips, args.repeats, args.threads) < 1:
        parser.error("clips, repeats and threads must be positive")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["Sunno_DATA_DIR"] = str(args.data_dir.resolve())

    from bench.bench_engines import platform_description, read_wav
    from server.config import FRAME_MS, Settings
    from server.engine import create_engine

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))[:args.clips]
    if not corpus:
        parser.error("corpus is empty")
    clips = []
    for item in corpus:
        path = args.corpus.parent / item["wav"]
        clips.append((item["id"], read_wav(path), hashlib.sha256(path.read_bytes()).hexdigest()))
    settings = Settings(model_size=args.model, device="cpu", compute_type="int8",
                        enable_speakers=False)
    events, lock = [], threading.Lock()

    def emit(event):
        if event["type"] in ("speech_start", "partial", "final", "error", "discard"):
            measured = dict(event, wall_s=time.monotonic())
            with lock:
                events.append(measured)

    start = time.monotonic()
    engine = create_engine(settings)
    load_ms = (time.monotonic() - start) * 1000
    pipeline = measured_pipeline(settings, engine, emit)
    runs = []
    try:
        print(f"{args.machine}: {args.model} loaded in {load_ms:.0f} ms", flush=True)
        # Warm the real pipeline, including the worker thread and VAD. Discard it.
        warmup = replay(pipeline, clips[0][1], events)
        if any(event["type"] == "error" for event in warmup["events"]):
            raise RuntimeError("Warmup emitted an ASR error")
        for repetition in range(1, args.repeats + 1):
            for clip_id, audio, digest in clips:
                result = replay(pipeline, audio, events)
                result.update(clip_id=clip_id, sha256=digest, repetition=repetition)
                runs.append(result)
                rows = result["utterances"]
                print(json.dumps({
                    "model": args.model, "clip": clip_id, "repetition": repetition,
                    "first_ms": [row["first_caption_delay_ms"] for row in rows],
                    "final_ms": [row["final_caption_delay_ms"] for row in rows],
                    "feed_max_ms": result["feed_lateness_ms"]["max_ms"],
                }), flush=True)
    finally:
        pipeline.close()

    rows = [row for run in runs for row in run["utterances"]]
    failures = [event for run in runs for event in run["events"]
                if event["type"] in ("error", "discard")]
    versions = {}
    for package in ("moonshine-voice", "numpy", "onnxruntime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    result = {
        "schema": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "machine": args.machine, "platform": platform_description(),
        "python": sys.version.split()[0], "versions": versions,
        "model": args.model, "provider": "CPU", "threads_requested": args.threads,
        "load_ms": load_ms, "clips": len(clips), "repeats": args.repeats,
        "settings": {
            "frame_ms": FRAME_MS, "end_silence_ms": settings.end_silence_ms,
            "effective_end_silence_ms": pipeline._end_silence_frames * FRAME_MS,
            "min_partial_ms": settings.min_partial_ms,
            "partial_interval_ms": settings.partial_interval_ms,
            "speaker_identification": False,
        },
        "summary": {
            "utterances": len(rows), "failures": len(failures),
            "first_caption": distribution(row["first_caption_delay_ms"] for row in rows),
            "final_caption": distribution(row["final_caption_delay_ms"] for row in rows),
            "final_decode": distribution(row["final_decode_ms"] for row in rows),
            "final_queue": distribution(row["final_queue_ms"] for row in rows),
            "final_only": sum(row["first_caption_type"] == "final" for row in rows),
            "non_silence_endpoints": sum(row["finish_reason"] != "silence_endpoint" for row in rows),
        },
        "runs": runs,
        "limits": [
            "VAD-estimated boundaries, not human phonetic annotations; 32 ms frames.",
            "Caption emission before WebSocket transport and UI painting.",
            "No microphone buffering, speaker identification or cold model load in caption delay.",
            "In-use host; background applications are not paused.",
            "Requested thread environment does not prove native ONNX thread count.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"]), flush=True)
    if (failures or not rows or result["summary"]["non_silence_endpoints"]
            or any(row["final_caption_delay_ms"] is None for row in rows)):
        raise RuntimeError("Incomplete replay; inspect the saved results before using measurements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
