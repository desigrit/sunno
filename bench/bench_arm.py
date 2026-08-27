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
No checkout needed - this is a single self-contained file. On the ARM machine, with the
**ARM64** build of Python 3.12 installed (python.org, ``python-3.12.10-arm64.exe``):

    curl -L -o bench_arm.py https://raw.githubusercontent.com/desigrit/sunno/master/bench/bench_arm.py
    py -3.12 bench_arm.py --setup

``--setup`` creates a venv beside the script, installs what it needs, and runs. Without it the
packages are assumed present. Add ``--qnn`` to also try the NPU, ``--models`` to limit which are
measured, and ``--wav`` to decode a specific clip.

It refuses to run under an emulated interpreter, because an x64 Python on a Snapdragon measures
emulation rather than ARM - which is the thing being replaced, not the thing being measured.

The run writes ``arm-hardware.json`` beside the script. That file is not read by the app - the
shipped lag tables in ``server/hardware.py`` are source constants - so its numbers are
transcribed by hand into the ARM catalog once they are known.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import sysconfig
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "server").is_dir():
    # Running as a standalone download rather than from a checkout. Keep the venv, the model
    # cache and the results beside the script instead of scattering them into whatever folder
    # the file happened to land in.
    ROOT = Path(__file__).resolve().parent

_MACHINE_NAMES: dict[int, str] = {
    0x0000: "unknown",
    0x014C: "x86",
    0x8664: "x64",
    0x01C4: "ARM32",
    0xAA64: "ARM64",
}


def wow64_machines() -> tuple[str, str]:
    """(process, native) from IsWow64Process2. DIAGNOSTIC ONLY - see :func:`machines`."""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.IsWow64Process2.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.USHORT),
            ctypes.POINTER(wintypes.USHORT),
        ]
        kernel32.IsWow64Process2.restype = wintypes.BOOL

        process = wintypes.USHORT()
        native = wintypes.USHORT()
        if not kernel32.IsWow64Process2(
            kernel32.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(native)
        ):
            return "unknown", "unknown"

        native_name = _MACHINE_NAMES.get(native.value, f"0x{native.value:04x}")
        if process.value == 0:
            return native_name, native_name
        return _MACHINE_NAMES.get(process.value, f"0x{process.value:04x}"), native_name
    except Exception:
        return "unknown", "unknown"


_CANONICAL = {
    "AMD64": "x64",
    "X86_64": "x64",
    "X64": "x64",
    "ARM64": "ARM64",
    "AARCH64": "ARM64",
    "ARM": "ARM32",
    "ARM32": "ARM32",
    "X86": "x86",
    "I386": "x86",
    "I686": "x86",
}

_TAG_TO_MACHINE = {"win-amd64": "AMD64", "win-arm64": "ARM64", "win32": "x86"}


def machines() -> tuple[str, str]:
    """(process machine, native machine).

    Duplicated from server/hardware.py on purpose. This file has to run on a machine with no
    checkout - downloading one script is a much smaller ask than cloning a repo with a 3 GB
    model history - and the check it performs is the one that decides whether the numbers below
    mean anything at all. A copy that always runs beats an import that sometimes does.

    The process half comes from ``sysconfig.get_platform()``, fixed when CPython was compiled
    and therefore immune to emulation; the native half from ``platform.machine()``, which on
    Windows answers from WMI's processor architecture and so describes the host CPU.

    Deliberately NOT IsWow64Process2, which this function used until the ARM64 startup crash of
    1.0.77 forced the question. Its pProcessMachine is documented as IMAGE_FILE_MACHINE_UNKNOWN
    for a process that "is not a WOW64 process", and reports disagree over what an emulated x64
    process on ARM64 returns. If it answers UNKNOWN there, the old code returned (native, native),
    read as not-emulated, and this script would have published emulated-x64 timings labelled
    ARM64 - the precise outcome its own refusal-to-run guard exists to prevent. It is still
    recorded in the JSON through :func:`wow64_machines`, but it no longer decides anything.
    """
    tag = sysconfig.get_platform()
    process = _TAG_TO_MACHINE.get(tag, "AMD64")
    native = platform.machine()
    return (
        _CANONICAL.get(process.strip().upper(), process),
        _CANONICAL.get(native.strip().upper(), native),
    )


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

# QNN lives in a separate distribution. The plain wheel bundles no Qualcomm provider at all -
# verified by listing it: onnxruntime-genai.dll and nothing else - so asking for --qnn without
# this installs a build that can only answer "QNN execution provider is not supported in this
# build", which says nothing about the machine.
QNN_REQUIREMENTS = ["onnxruntime-qnn"]


def bootstrap(cache_root: Path, want_qnn: bool) -> int:
    """Create a venv, install what the benchmark needs, and re-run inside it.

    Exists so the first thing to do on a test machine is run one command, not assemble an
    environment. Deliberately a separate venv rather than the repo's .venv: that one is staged
    from x64 wheels and its ctranslate2 cannot load here at all.
    """
    import subprocess

    venv = cache_root / ".venv-arm"
    python = venv / "Scripts" / "python.exe"
    if not python.is_file():
        print(f"  creating {venv}")
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)

    wanted = REQUIREMENTS + (QNN_REQUIREMENTS if want_qnn else [])
    print(f"  installing {', '.join(wanted)}")
    subprocess.run([str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=False)
    result = subprocess.run([str(python), "-m", "pip", "install", "--quiet", *wanted])
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
    process, native = machines()
    return {
        "python": platform.python_version(),
        "platform_machine": platform.machine(),
        "processor": platform.processor(),
        "process_machine": process,
        "native_machine": native,
        # Recorded so a run from real ARM64 hardware finally settles what this API returns
        # there. It decides nothing - see machines().
        "wow64_probe": list(wow64_machines()),
        "platform_tag": sysconfig.get_platform(),
        # An unreadable answer degrades to "not emulated" rather than putting a slowness
        # warning on a machine nothing is known about. The empty string counts as
        # unreadable: platform.machine() returns "" when the WMI query raises and neither
        # PROCESSOR_* variable is set, and treating that as emulation would make this
        # script refuse to run on exactly the failure path the guard exists for.
        "emulated": bool(process)
        and bool(native)
        and process != native
        and "unknown" not in (process, native),
    }


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


def enable_qnn() -> tuple[bool, str]:
    """Make the Qualcomm NPU available to genai, and say plainly whether it worked.

    QNN is not built into onnxruntime-genai - its wheel contains one DLL and no Qualcomm
    provider at all. It ships separately as onnxruntime-qnn, which is not a replacement for
    onnxruntime but a plugin package: it installs as ``onnxruntime_qnn`` and carries
    onnxruntime_providers_qnn.dll plus the QnnHtp libraries. genai loads it through
    register_execution_provider_library.

    Returns (ready, detail) so a failure can be reported as what it is. "QNN execution provider
    is not supported in this build" reads like a verdict on the machine, and is not one.
    """
    try:
        import onnxruntime_qnn
    except ImportError:
        return False, "onnxruntime-qnn is not installed (re-run with --setup --qnn)"

    try:
        import onnxruntime_genai as og

        name = onnxruntime_qnn.get_ep_name()
        library = onnxruntime_qnn.get_library_path()
        if not Path(library).is_file():
            return False, f"provider library missing at {library}"
        og.register_execution_provider_library(name, library)
        return True, name
    except Exception as exc:  # noqa: BLE001
        return False, f"could not register the QNN provider: {str(exc)[:160]}"


def measure(model_dir: Path, clip_path: Path, provider: str | None) -> dict | None:
    """Decode the clips and report per-duration latency, or None if the model will not load."""
    import onnxruntime_genai as og

    result: dict = {"provider": provider or "cpu"}

    try:
        started = time.perf_counter()
        # Config is only touched when a provider is named, so the default path stays exactly
        # what genai would do on its own - a bad provider override would otherwise look like a
        # slow model.
        if provider:
            config = og.Config(str(model_dir))
            config.clear_providers()
            config.append_provider(provider)
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
    # Single-dash spellings accepted too. PowerShell's own flags use one dash, so -qnn is a
    # natural thing to type, and failing on it wastes a run on a machine that may be borrowed.
    parser.add_argument("--qnn", "-qnn", action="store_true",
                        help="also try the Qualcomm NPU (encoder is expected to offload, decoder to fall back)")
    parser.add_argument("--models", "-models", nargs="*", default=None,
                        help="subset of model ids to measure; default is all")
    parser.add_argument("--cache", "-cache", default=str(ROOT / ".onnx-models"),
                        help="where to keep downloaded ONNX weights")
    parser.add_argument("--wav", "-wav", default=None,
                        help="clip to decode; defaults to testdata, else one synthesised with the Windows voice")
    parser.add_argument("--out", "-out", default=None,
                        help="where to write the results; defaults to arm-hardware.json beside this script")
    parser.add_argument("--setup", "-setup", action="store_true",
                        help="create a venv, install what is needed, and run inside it")
    args = parser.parse_args()

    print("Sunno ARM decode benchmark")
    print("=" * 62)

    if args.setup:
        return bootstrap(ROOT, args.qnn)

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

    qnn_provider = None
    if args.qnn:
        ready, detail = enable_qnn()
        machine["qnn"] = detail if ready else f"unavailable: {detail}"
        if ready:
            qnn_provider = detail
            print(f"  qnn provider       {detail}")
        else:
            print(f"  qnn provider       unavailable - {detail}")
            print("                     (this is about the install, not about this PC)")

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

        for provider in ([None, qnn_provider] if qnn_provider else [None]):
            label = "qnn" if provider else "cpu"
            measured = measure(model_dir, clip, provider)
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

    # Named after the machine by default, because these files come from several laptops and
    # get compared against each other. A fixed name means the second run silently overwrites
    # the first, and the numbers are only meaningful next to the machine that produced them.
    if args.out:
        out = Path(args.out)
    else:
        tag = (machine.get("processor") or "unknown").split()[0].strip(",")
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in tag)[:24]
        out = Path(__file__).with_name(f"arm-hardware-{machine.get('native_machine','?')}-{safe}.json")
    out.write_text(json.dumps({"machine": machine, "results": results}, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    print("\n  Send this file back - it decides which models the ARM build offers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
