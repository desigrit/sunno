"""Convert an upstream Whisper into the genai format the ARM engine loads.

Why this exists
---------------
server/asr_onnx.py runs Whisper through onnxruntime-genai, which will only load a model laid
out its own way:

    genai_config.json  audio_processor_config.json  tokenizer.json
    encoder.onnx (+ .data)  decoder.onnx (+ .data)

The popular onnx-community/whisper-* repos are Optimum/Transformers.js exports and carry no
genai_config.json, so genai refuses them outright. Of 189 Whisper repos on HuggingFace only a
handful are in genai format, and those are one unaffiliated individual's uploads - not something
to put under a captioning app people rely on.

Converting from openai/whisper-* ourselves removes that question. The weights are MIT and the
conversion is reproducible, so the ARM build depends on OpenAI and Microsoft rather than on a
stranger's account staying up.

Two workarounds are load-bearing, both found the hard way
--------------------------------------------------------
1. transformers 5.x asks the Hub with token=True even for public models, so the builder's own
   -m/--model_name download fails with LocalTokenNotFoundError. Fetching the weights first with
   huggingface_hub (which is happy anonymous) and pointing the builder at the folder avoids it.

2. onnxruntime_genai/models/builders/base.py deletes the cache directory when it is empty, and
   the Whisper builder calls save_model twice - once for the encoder, once for the decoder. The
   second call then dies on a directory the first one removed. Keeping a file in there stops it
   being empty and the build completes.

Usage
-----
Needs an x64 machine: the builder imports torch, which publishes no win_arm64 wheel. That is
fine - this is a build-time step, and its output is what ships.

    python -m venv .venv-convert
    .venv-convert\\Scripts\\python.exe -m pip install onnxruntime-genai onnx onnx_ir transformers torch
    .venv-convert\\Scripts\\python.exe bench\\convert_whisper_genai.py --model base --out dist/onnx

Verified: base at int8 converts and decodes real speech correctly in about 363 ms on an
i9-14900K CPU.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Upstream weights, not a CTranslate2 conversion of them. MIT, and the source everything else
# downstream is derived from.
SOURCES = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
}

# int8 on CPU: the accuracy cost is small and it is the quantisation that actually pays there,
# which is where the ARM decoder runs.
DEFAULT_PRECISION = "int8"


def fetch(repo: str, dest: Path) -> Path:
    """Download upstream weights anonymously, sidestepping the builder's token requirement."""
    from huggingface_hub import snapshot_download

    print(f"  fetching {repo}")
    path = snapshot_download(
        repo_id=repo,
        local_dir=str(dest),
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"],
    )
    return Path(path)


def convert(source: Path, out: Path, precision: str, provider: str) -> bool:
    """Run the genai builder, keeping its cache directory alive across both save passes."""
    cache = out.parent / f".cache-{out.name}"
    cache.mkdir(parents=True, exist_ok=True)
    # See the module docstring: an empty cache dir gets removed after the encoder is written,
    # and the decoder pass then fails listing it.
    (cache / ".keep").write_text("keep", encoding="utf-8")

    out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "onnxruntime_genai.models.builder",
        "-i", str(source), "-o", str(out),
        "-p", precision, "-e", provider, "-c", str(cache),
    ]
    print(f"  converting to {precision} for {provider}")
    result = subprocess.run(command)
    shutil.rmtree(cache, ignore_errors=True)
    return result.returncode == 0


def verify(out: Path) -> bool:
    """Confirm genai will load it, which is the only completeness check that means anything."""
    required = ["genai_config.json", "encoder.onnx", "decoder.onnx", "tokenizer.json"]
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        print(f"  INCOMPLETE - missing {', '.join(missing)}")
        return False

    try:
        import onnxruntime_genai as og

        model = og.Model(str(out))
        model.create_multimodal_processor()
        print("  loads in onnxruntime-genai")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  built but will not load: {str(exc)[:180]}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="*", default=["base"], choices=sorted(SOURCES),
                        help="which sizes to convert")
    parser.add_argument("--out", default="dist/onnx", help="where to write the converted models")
    parser.add_argument("--precision", default=DEFAULT_PRECISION,
                        choices=["int4", "int8", "fp16", "fp32"])
    parser.add_argument("--provider", default="cpu", choices=["cpu", "cuda"],
                        help="target for the conversion; cpu is what ARM runs")
    parser.add_argument("--work", default="dist/.whisper-src", help="where upstream weights land")
    args = parser.parse_args()

    print("Whisper -> onnxruntime-genai")
    print("=" * 62)

    failures = 0
    for name in args.model:
        print(f"\n  {name}  ({SOURCES[name]})")
        source = fetch(SOURCES[name], Path(args.work) / name)
        out = Path(args.out) / name
        if not convert(source, out, args.precision, args.provider) or not verify(out):
            failures += 1
            continue
        size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        print(f"  done - {size / 1024 / 1024:.0f} MB at {out}")

    print()
    if failures:
        print(f"{failures} model(s) failed")
        return 1
    print("All converted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
