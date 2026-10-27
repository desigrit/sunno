"""Measure Sunno's real engine calls, not a vendor's advertised latency.

Run one model per process to isolate memory and avoid concurrent benchmarks.
Batch engines use Sunno's actual partial/final decoding settings. Incremental
engines receive only new audio, at Sunno's caption cadence. Timings exclude
microphone capture, endpoint detection, speaker identification and UI rendering.
The harness does not open a microphone or publish transcriptions.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time
import wave
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower().replace("’", "'"))


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    expected, actual = normalized_words(reference), normalized_words(hypothesis)
    previous = list(range(len(actual) + 1))
    for i, word in enumerate(expected, 1):
        current = [i]
        for j, heard in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (word != heard)))
        previous = current
    return previous[-1], len(expected)


def platform_description() -> str:
    # platform.platform() performs a WMI CPU query on Windows. That optional
    # query can time out and discard an otherwise completed benchmark.
    if sys.platform == "win32":
        version = sys.getwindowsversion()
        return f"Windows-{version.major}.{version.minor}.{version.build}"
    info = os.uname()
    return f"{info.sysname}-{info.release}-{info.machine}"


def read_wav(path: Path):
    import numpy as np
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != 16000 or wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise ValueError("Benchmark audio must be 16 kHz, mono, 16-bit PCM")
        raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def streaming_pass(engine, audio, settings, *, paced=False):
    import numpy as np
    start = time.perf_counter()
    engine.start_stream()
    start_ms = (time.perf_counter() - start) * 1000
    playback_start = time.perf_counter()
    offset = 0
    step = int(16000 * settings.partial_interval_ms / 1000)
    first_step = int(16000 * settings.min_partial_ms / 1000)
    costs, first_text_audio_s = [], None
    stop = min(first_step, len(audio))
    while offset < len(audio):
        if paced:
            time.sleep(max(0, playback_start + stop / 16000 - time.perf_counter()))
        t0 = time.perf_counter()
        result = engine.stream(audio[offset:stop])
        costs.append((time.perf_counter() - t0) * 1000)
        if result.text and first_text_audio_s is None:
            first_text_audio_s = stop / 16000
        offset = stop
        if paced:
            # Sunno backs off expensive refreshes and coalesces queued audio.
            adaptive_ms = max(settings.partial_interval_ms, min(2000, costs[-1] * 1.25))
            due = max(offset + int(16 * adaptive_ms),
                      int((time.perf_counter() - playback_start) * 16000))
            stop = min(len(audio), due)
        else:
            stop = min(len(audio), offset + step)
    t0 = time.perf_counter()
    result = engine.stream(np.empty(0, np.float32), final=True)
    final_ms = (time.perf_counter() - t0) * 1000
    metrics = {
        "final_ms": final_ms, "total_compute_ms": start_ms + sum(costs) + final_ms,
        "partial_median_ms": statistics.median(costs), "partial_max_ms": max(costs),
        "first_text_audio_s": first_text_audio_s, "updates": len(costs),
    }
    if paced:
        metrics["final_available_delay_ms"] = max(0, (
            time.perf_counter() - playback_start - len(audio) / 16000) * 1000)
    return metrics, result.text


def batch_pass(engine, audio):
    clear = getattr(engine, "clear_context", None)
    if clear:
        clear()
    t0 = time.perf_counter()
    partial = engine.partial(audio)
    partial_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    final = engine.final(audio)
    final_ms = (time.perf_counter() - t0) * 1000
    return {"partial_ms": partial_ms, "final_ms": final_ms}, final.text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--wav", type=Path, default=ROOT / "testdata/1-two-speakers-en.wav")
    parser.add_argument("--corpus", type=Path, help="JSON list with wav and reference fields")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--paced", action="store_true", help="Replay streaming audio in real time with adaptive refreshes")
    parser.add_argument("--download", action="store_true", help="Fetch missing model weights, never system AI models")
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    if args.data_dir:
        os.environ["Sunno_DATA_DIR"] = str(args.data_dir)

    def memory_info():
        if sys.platform == "darwin":
            import resource
            import subprocess
            rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) * 1024
            return SimpleNamespace(rss=rss, peak_wset=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        import psutil
        return psutil.Process().memory_info()
    from server.config import Settings
    from server.engine import create_engine, IncrementalSpeechEngine
    from server import models
    if args.device == "cuda":
        from server.cuda_setup import register_cuda_dlls
        register_cuda_dlls(required=True)
    status = models.is_available(args.model)
    if not status.available:
        if not args.download or args.model in ("windows-speech", "apple-speech"):
            print(f"SKIP {args.model}: model is not installed", flush=True)
            return 2
        print(f"Downloading {args.model} weights", flush=True)
        models.download(args.model)
    baseline_mb = memory_info().rss / 1048576
    settings = Settings(model_size=args.model, device=args.device,
                        compute_type="float16" if args.device == "cuda" else "int8")
    t0 = time.perf_counter()
    engine = create_engine(settings)
    load_ms = (time.perf_counter() - t0) * 1000
    incremental = isinstance(engine, IncrementalSpeechEngine)
    loaded_mb = memory_info().rss / 1048576
    audio = read_wav(args.wav)
    if len(audio) < 16000 * 8:
        raise ValueError("Latency clip must contain at least eight seconds")
    timing = (lambda e, a, cfg: streaming_pass(e, a, cfg, paced=args.paced)) if incremental \
        else lambda e, a, _: batch_pass(e, a)
    rows = []
    try:
        # One discarded warmup uses real speech and the same final decode path.
        timing(engine, audio[:32000], settings)
        for seconds in (2, 4, 8):
            runs = [timing(engine, audio[:16000 * seconds], settings)[0]
                    for _ in range(args.repeats)]
            row = {"audio_s": seconds, "runs": runs}
            for key in runs[0]:
                values = [run[key] for run in runs if run[key] is not None]
                row[key] = statistics.median(values) if values else None
            cost = row.get("total_compute_ms", row["final_ms"])
            row["rtf"] = cost / (seconds * 1000)
            rows.append(row)
            print(f"{args.model} {args.device}: {seconds}s, final {row['final_ms']:.1f}ms, "
                  f"RTF {row['rtf']:.3f}", flush=True)
        accuracy = []
        if args.corpus:
            entries = json.loads(args.corpus.read_text(encoding="utf-8"))
            for entry in entries:
                sample = read_wav(args.corpus.parent / entry["wav"])
                metrics, text = timing(engine, sample, settings)
                errors, words = word_errors(entry["reference"], text)
                accuracy.append({"id": entry["id"], "audio_s": len(sample) / 16000,
                                 "word_errors": errors, "reference_words": words,
                                 "hypothesis": text, **metrics})
            words = sum(row["reference_words"] for row in accuracy)
            wer = sum(row["word_errors"] for row in accuracy) / words if words else None
            print(f"Accuracy sanity check: {len(accuracy)} clips, {words} words, WER {wer:.1%}", flush=True)
        else:
            wer = None
        memory = memory_info()
        result = {
            "model": args.model, "device": args.device, "threads_requested": args.threads,
            "incremental": incremental, "platform": platform_description(),
            "paced": args.paced, "engine_class": type(engine).__name__,
            "python": platform.python_version(), "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "clip": args.wav.name, "clip_sha256": hashlib.sha256(args.wav.read_bytes()).hexdigest(),
            "load_ms": load_ms, "baseline_rss_mb": baseline_mb, "loaded_rss_mb": loaded_mb,
            "rss_mb": memory.rss / 1048576,
            "peak_rss_mb": getattr(memory, "peak_wset", memory.rss) / 1048576,
            "accuracy_wer": wer, "accuracy": accuracy, "rows": rows,
            "notes": "Engine calls only, not end-to-end caption delay. Median of warmed repetitions. CPU thread request may not control every runtime. RAM excludes GPU memory and native helper processes.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved {args.output}", flush=True)
    finally:
        close = getattr(engine, "close", None)
        if close:
            close()
        del engine
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
