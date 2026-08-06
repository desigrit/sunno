"""The ARM picker and downloader must describe the genai artifacts, not CT2 models."""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import engine, hardware, models  # noqa: E402

failures: list[str] = []
root = Path(__file__).resolve().parents[1]


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"PASS  {name}")
    else:
        failures.append(f"{name}: {detail}")
        print(f"FAIL  {name}  {detail}")


catalog = models.catalog_for("onnx")
check("ARM offers base then tiny", [entry["id"] for entry in catalog] == ["base", "tiny"])
check("ARM model repo is first-party", models._repo_id("base", "onnx") == "desigrit/Sunno")
check(
    "base download patterns stay in its directory",
    all(pattern.startswith("base/") for pattern in models._allow_patterns("base", "onnx")),
)
check("base lag is responsive", hardware.estimated_lag_ms("base", "cpu", "onnx") == 519)
check("tiny lag is responsive", hardware.estimated_lag_ms("tiny", "cpu", "onnx") == 203)
check("base is the ARM default", hardware.default_model(["base", "tiny"], "cpu", "onnx") == "base")
requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
check(
    "GenAI runtime is declared",
    any(line.strip().startswith("onnxruntime-genai") for line in requirements),
)
arm_requirements = (root / "requirements-arm64.txt").read_text(encoding="utf-8").splitlines()
check(
    "ARM dependency set excludes unavailable native packages",
    not any(
        line.strip().lower().startswith(("faster-whisper", "sherpa-onnx", "nvidia-"))
        for line in arm_requirements
    ),
)
check(
    "CT2 latency keys remain compatible",
    hardware._latency_key("large-v3", "cuda", "ct2") == "cuda:large-v3",
)

real_import = __import__


def fail_genai_import(name: str, *args: object, **kwargs: object) -> object:
    if name == "onnxruntime_genai":
        raise ImportError("test missing GenAI runtime")
    return real_import(name, *args, **kwargs)


hardware.engine_importable.cache_clear()

no_engine_log = StringIO()
with patch.object(
    engine,
    "available_engines",
    return_value={"ct2": False, "onnx": False},
), redirect_stdout(no_engine_log):
    try:
        engine.resolve_engine()
        no_engine_raised = False
    except RuntimeError:
        no_engine_raised = True
check("missing all engines aborts resolution", no_engine_raised)
check(
    "missing all engines emits the host failure marker",
    "speech engine could not be loaded" in no_engine_log.getvalue(),
    no_engine_log.getvalue(),
)
engine_log = StringIO()
with patch("builtins.__import__", side_effect=fail_genai_import), redirect_stdout(engine_log):
    onnx_importable = hardware.engine_importable("onnx")
check("missing ONNX engine is reported", not onnx_importable)
check(
    "ONNX load failure uses architecture-neutral wording",
    "onnx speech engine could not be loaded" in engine_log.getvalue()
    and "Intel or AMD" not in engine_log.getvalue(),
    engine_log.getvalue(),
)
hardware.engine_importable.cache_clear()

with tempfile.TemporaryDirectory() as temp:
    previous = os.environ.get("LOCALAPPDATA")
    os.environ["LOCALAPPDATA"] = temp
    try:
        check("missing ONNX model is unavailable", not models.is_available("base", "onnx").available)
        model_dir = Path(temp) / "Sunno" / "onnx-models" / "base"
        model_dir.mkdir(parents=True)
        for name in models._ONNX_FILENAMES:
            (model_dir / name).touch()
        status = models.is_available("base", "onnx")
        check("complete ONNX model is available", status.available, str(status))
        check("availability returns the model directory", status.path == str(model_dir), str(status))
    finally:
        if previous is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = previous

print()
if failures:
    print(f"{len(failures)} FAILED")
    for failure in failures:
        print(f"  {failure}")
    raise SystemExit(1)

print("ALL PASS")
