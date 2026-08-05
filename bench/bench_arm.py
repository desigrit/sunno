"""Measure ONNX Whisper decode latency on this machine, especially Windows on ARM.

Why this exists
---------------
The x64 build runs Whisper through CTranslate2, which publishes no win_arm64 wheel. The
replacement is ONNX Runtime via onnxruntime-genai, and the question that decides the whole
ARM product is one nobody has published an answer to: how fast does the Whisper *decoder*
run on a Snapdragon X?

The encoder is fixed-shape and NPU-friendly. The decoder is autoregressive - its KV cache
grows one step at a time - so QNN's static-shape requirement pushes it back to the CPU. That
means decode speed on ARM is a CPU question, and ARM CPU decode numbers for Whisper are not
published anywhere I could find. Guessing costs weeks aimed at the wrong model tier, so this
measures instead.

What it decides: whether ARM gets large-v3-turbo (best accented-speech accuracy of the fast
models) or has to drop to something smaller.

Usage
-----
Copy the repo to the ARM machine, then:

    py -3.12 bench\\bench_arm.py --setup

That creates a venv beside the repo, installs what it needs, and runs. Without ``--setup`` it
assumes the packages are already present. Add ``--qnn`` to also try the NPU, ``--models`` to
limit which are measured, and ``--wav`` to decode a specific clip.

The run writes ``bench/arm-hardware.json``. That file is not read by the app - the shipped lag
tables in ``server/hardware.py`` are source constants - so its numbers are transcribed by hand
into the ARM catalog once they are known.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Ordered fastest-first, because on a slow machine the later entries may not be worth waiting
# for and a partial table is still a useful answer.
#
# These are genai-format repos, which is not the same thing as "ONNX Whisper on HuggingFace".
# The popular onnx-community/whisper-* repos are Optimum/Transformers.js exports: they carry
# encoder_model.onnx and decoder_model_merged.onnx in every quantisation, but no
# genai_config.json, and onnxruntime-genai refuses to load without it (verified - it fails with
# "Error opening ...genai_config.json"). A genai model is a different shape:
#
#     genai_config.json  audio_processor_config.json  encoder.onnx  decoder.onnx  tokenizer.json
#
# Producing that from an upstream Whisper needs onnxruntime_genai.models.builders.whisper, which
# pulls onnx_ir/onnx/torch/transformers - a heavy build-time chain, but a one-off, and the result
# can be published or bundled.
#
# Note what is missing below: large-v3-turbo. No genai-format build of it was found, and it is
# the model that matters most for accented speech. Converting it is the open item this benchmark
# is meant to justify or rule out.
CANDIDATES: list[dict] = [
    {
        "id": "whisper-tiny",
        "repo": "tonythethompson/Whisper-Tiny-GenAI-ONNX",
        "note": "floor - if this is slow, the machine cannot caption live at all",
    },
    {
        "id": "whisper-base",
        "repo": "tonythethompson/Whisper-Base-GenAI-ONNX",
        "note": "low tier candidate for a machine with no usable NPU",
    },
    {
        "id": "whisper-small",
        "repo": "tonythethompson/Whisper-Small-GenAI-ONNX",
        "note": "mid tier",
    },
    {
        "id": "whisper-medium",
        "repo": "tonythethompson/Whisper-Medium-GenAI-ONNX",
        "note": "upper bound available in genai format today",
    },
]

# Matches the x64 measurement in bench_latency.py so the two tables can be compared directly.
DURATIONS = [2.0, 4.0, 8.0]

# server/hardware.py treats anything at or under this as responsive enough to follow a
# conversation rather than merely transcribe a recording.
RESPONSIVE_LAG_MS = 1000

# Whisper's forced decoder prefix. The app pins language to English and does not use
# timestamps, so the benchmark decodes the same way - otherwise it would be timing a different
# amount of work than the product does.
PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"


REQUIREMENTS = ["onnxruntime-genai", "huggingface_hub", "numpy"]


def bootstrap(cache_root: Path) -> int:
    """Create a venv, install what the benchmark needs, and re-run inside it.

    Exists so the first thing he does on the ARM laptop is run one command, not assemble an
    environment. Deliberately a separate venv rather than the repo's .venv: that one is staged
    from x64 wheels and its ctranslate2 cannot load here at all.
    """
    import subprocess

    venv = cache_root / ".venv-arm"
    python = venv / "Scripts" / "python.exe"
    if not python.is_file():
        print(f"  creating {venv}")
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)

    print(f"  installing {', '.join(REQUIREMENTS)}")
    subprocess.run([str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=False)
    result = subprocess.run([str(python), "-m", "pip", "install", "--quiet", *REQUIREMENTS])
    if result.returncode != 0:
        print("\n  Install failed. The most likely cause on ARM64 is that one of these has no")
        print("  win_arm64 wheel for this Python version. Try python 3.12 specifically.")
        return result.returncode

    # Hand over to the venv, minus --setup so this does not recurse.
    forwarded = [a for a in sys.argv[1:] if a != "--setup"]
    print(f"  re-running inside {python.name}\n")
    return subprocess.run([str(python), str(Path(__file__).resolve()), *forwarded]).returncode


def describe_machine() -> dict:
    """Record what this actually ran on, including emulation, which changes the numbers."""
    info = {
        "python": platform.python_version(),
        "platform_machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        # Imported for native_machine() only. server/__init__ registers the CUDA DLL
        # directories on import and warns when they are absent, which is noise here and on
        # every ARM machine by definition - so the warning is suppressed rather than left to
        # look like a benchmark failure.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from server import hardware
        info["process_machine"] = hardware.process_machine()
        info["native_machine"] = hardware.native_machine()
        info["emulated"] = hardware.is_emulated()
    except Exception as exc:  # noqa: BLE001
        info["native_machine"] = f"unavailable ({exc})"
    return info


def make_speech_clip(dest: Path) -> Path | None:
    """Synthesise a speech clip with the Windows voice, so the harness needs no assets.

    testdata/ is gitignored - it holds real recordings of real people - so "copy the repo to
    the ARM laptop" would otherwise deliver a benchmark with nothing to decode. Every Windows
    machine has SAPI, so the clip can be generated on the spot, identically on both machines,
    with nothing to download and nobody's voice in it.

    Synthetic speech is not a WER benchmark and is not being used as one. What is being
    measured here is decode *latency*, which depends on audio length and model size rather
    than on who is speaking. The decoded text is printed only so a run that silently produces
    nonsense is visible rather than being reported as a fast success.
    """
    if dest.is_file():
        return dest

    # Long enough to fill the longest window measured below without SAPI padding it out.
    line = (
        "The quick brown fox jumps over the lazy dog. "
        "She sells sea shells by the sea shore on a bright and windy afternoon. "
        "Please read this sentence aloud so the model has something to transcribe. "
        "We are measuring how long the decoder takes, not how well it hears."
    )
    script = (
        "$ErrorActionPreference = 'Stop';"
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        # 16 kHz mono 16-bit is exactly what the pipeline feeds Whisper, so no resampling
        # sits between the benchmark and the number it reports.
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,"
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,"
        "[System.Speech.AudioFormat.AudioChannel]::Mono);"
        f"$s.SetOutputToWaveFile('{dest}', $f);"
        f"$s.Speak('{line}');"
        "$s.Dispose();"
    )
    try:
        import subprocess

        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            check=True, capture_output=True, timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    could not synthesise a clip: {str(exc)[:160]}")
        return None
    return dest if dest.is_file() else None


def resolve_clip(explicit: str | None) -> Path | None:
    """Pick the audio to decode: what was asked for, else real test data, else synthesised."""
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None

    recorded = ROOT / "testdata" / "1-two-speakers-en.wav"
    if recorded.is_file():
        return recorded

    return make_speech_clip(Path(__file__).with_name("bench-speech.wav"))


def load_clip(path: Path, seconds: float) -> "object":
    """Read a mono 16 kHz float32 clip of the requested length."""
    import numpy as np

    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    want = int(seconds * rate)
    if len(audio) < want:
        audio = np.pad(audio, (0, want - len(audio)))
    return audio[:want], rate


def to_wav_bytes(audio, rate: int) -> bytes:
    """genai takes a file or bytes, never an array, so the live path must encode in memory.

    Worth knowing before committing to this engine: the real app holds numpy frames from the
    microphone, so every utterance pays this encode. It is small next to decode, but it is not
    nothing, and it is measured separately below.
    """
    import io

    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def ensure_model(repo: str, dest: Path) -> Path | None:
    """Fetch a genai-format ONNX Whisper from HuggingFace, resuming if it is partly there.

    Restricted to the files genai actually loads. Without this, snapshot_download pulls every
    quantisation a repo happens to carry - about 1.5 GB for a model whose useful weights are
    under 100 MB - which on a laptop being benchmarked is both slow and misleading.
    """
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(
            repo_id=repo,
            local_dir=str(dest),
            allow_patterns=[
                "genai_config.json",
                "audio_processor_config.json",
                "tokenizer*.json",
                "*.onnx",
                "*.onnx.data",
            ],
        )
        return Path(path)
    except Exception as exc:  # noqa: BLE001
        print(f"    could not fetch {repo}: {str(exc)[:160]}")
        return None


def measure(model_dir: Path, clip_path: Path, use_qnn: bool) -> dict | None:
    """Decode the clips and report per-duration latency, or None if the model will not load."""
    import onnxruntime_genai as og

    result: dict = {"provider": "qnn" if use_qnn else "cpu"}

    try:
        started = time.perf_counter()
        # Config is only touched for QNN so the default path stays exactly what genai would do
        # on its own - a bad provider override would otherwise look like a slow model.
        if use_qnn:
            config = og.Config(str(model_dir))
            config.clear_providers()
            config.append_provider("qnn")
            model = og.Model(config)
        else:
            model = og.Model(str(model_dir))
        result["load_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"load failed: {str(exc)[:200]}", **result}

    try:
        processor = model.create_multimodal_processor()
        tokenizer = og.Tokenizer(model)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"processor failed: {str(exc)[:200]}", **result}

    per_duration: dict[str, dict] = {}
    for seconds in DURATIONS:
        audio, rate = load_clip(clip_path, seconds)

        encode_started = time.perf_counter()
        wav = to_wav_bytes(audio, rate)
        encode_ms = (time.perf_counter() - encode_started) * 1000.0

        best_ms = None
        text = ""
        # Best of three, matching bench_latency.py. The first pass also absorbs any lazy
        # kernel init, so taking the minimum is what makes runs comparable.
        for _ in range(3):
            try:
                started = time.perf_counter()
                audios = og.Audios.open_bytes(wav)
                # Whisper decodes from a forced token prefix that declares language and task;
                # genai surfaces that as the processor's prompt rather than inferring it, and
                # omitting it fails with "No prompt or prompts provided to WhisperProcessor".
                # notimestamps matches how the app decodes: timestamps are not used, and asking
                # for them would make the decoder emit extra tokens and look slower than it is.
                inputs = processor(prompt=PROMPT, audios=audios)
                params = og.GeneratorParams(model)
                generator = og.Generator(model, params)
                generator.set_inputs(inputs)
                while not generator.is_done():
                    generator.generate_next_token()
                elapsed = (time.perf_counter() - started) * 1000.0
                best_ms = elapsed if best_ms is None else min(best_ms, elapsed)
                text = processor.decode(generator.get_sequence(0))
            except Exception as exc:  # noqa: BLE001
                return {"error": f"decode failed: {str(exc)[:200]}", **result}

        per_duration[f"{seconds:g}s"] = {
            "decode_ms": round(best_ms, 1) if best_ms else None,
            "wav_encode_ms": round(encode_ms, 2),
            "text": text.strip()[:120],
        }

    result["by_duration"] = per_duration
    lags = [v["decode_ms"] for v in per_duration.values() if v["decode_ms"]]
    if lags:
        result["mean_decode_ms"] = round(sum(lags) / len(lags), 1)
        result["responsive"] = result["mean_decode_ms"] <= RESPONSIVE_LAG_MS
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qnn", action="store_true",
                        help="also try the Qualcomm NPU (encoder is expected to offload, decoder to fall back)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="subset of model ids to measure; default is all")
    parser.add_argument("--cache", default=str(ROOT / ".onnx-models"),
                        help="where to keep downloaded ONNX weights")
    parser.add_argument("--wav", default=None,
                        help="clip to decode; defaults to testdata, else one synthesised with the Windows voice")
    parser.add_argument("--setup", action="store_true",
                        help="create a venv, install what is needed, and run inside it")
    args = parser.parse_args()

    print("Sunno ARM decode benchmark")
    print("=" * 62)

    if args.setup:
        return bootstrap(ROOT)

    machine = describe_machine()
    for key, value in machine.items():
        print(f"  {key:18} {value}")

    try:
        import onnxruntime_genai as og
    except ImportError:
        print("\n  onnxruntime-genai is not installed. Run:")
        print("    python -m pip install onnxruntime-genai huggingface_hub numpy")
        return 2

    print(f"  genai              {getattr(og, '__version__', '?')}")
    print(f"  qnn available      {og.is_qnn_available()}")

    if machine.get("emulated"):
        print()
        print("  STOP: this Python is emulated, so every number below would be wrong.")
        print(f"        You are running a {machine.get('process_machine')} Python on a "
              f"{machine.get('native_machine')} machine.")
        print("        Install the ARM64 build from python.org and use that instead:")
        print("          https://www.python.org/ftp/python/3.12.10/python-3.12.10-arm64.exe")
        return 2

    if machine.get("native_machine") not in ("ARM64",):
        print("\n  NOTE: this is not an ARM64 machine. The run still works and is useful as a")
        print("        baseline, but it does not answer the Snapdragon question.")

    clip = resolve_clip(args.wav)
    if clip is None:
        print("\n  no audio to decode, and synthesising one failed.")
        print("  Pass one explicitly:  python bench\\bench_arm.py --wav some-speech.wav")
        return 2
    print(f"  clip               {clip.name}")

    wanted = CANDIDATES if not args.models else [c for c in CANDIDATES if c["id"] in args.models]
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for entry in wanted:
        print(f"\n  {entry['id']}  - {entry['note']}")
        model_dir = ensure_model(entry["repo"], cache / entry["id"])
        if model_dir is None:
            results[entry["id"]] = {"error": "download failed"}
            continue

        for use_qnn in ([False, True] if args.qnn else [False]):
            label = "qnn" if use_qnn else "cpu"
            measured = measure(model_dir, clip, use_qnn)
            results[f"{entry['id']}:{label}"] = measured
            if measured is None or "error" in measured:
                print(f"    {label:4} {measured.get('error', 'unknown failure') if measured else 'failed'}")
                continue
            mean = measured.get("mean_decode_ms")
            verdict = "responsive" if measured.get("responsive") else "too slow to follow a conversation"
            print(f"    {label:4} load {measured['load_ms']:>8.1f} ms   mean decode {mean:>8.1f} ms   {verdict}")
            for dur, row in measured["by_duration"].items():
                print(f"         {dur:>4}  {row['decode_ms']:>8.1f} ms   (+{row['wav_encode_ms']:.2f} ms wav encode)")
                print(f"               \"{row['text']}\"")

    out = Path(__file__).with_name("arm-hardware.json")
    out.write_text(json.dumps({"machine": machine, "results": results}, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    print("\n  Send this file back - it decides which models the ARM build offers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
