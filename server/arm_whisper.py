"""Native Windows ARM64 Whisper models and their separate GenAI cache."""

from pathlib import Path
import threading

REPO = "desigrit/Sunno"
REVISION = "f57a056cc4f7044e7528af439f69e8deb8c86c03"
FILES = ("audio_processor_config.json", "decoder.onnx", "decoder.onnx.data",
         "encoder.onnx", "encoder.onnx.data", "genai_config.json",
         "tokenizer_config.json", "tokenizer.json")
CATALOG = [
    {"id": "base", "name": "Whisper base", "detail": "Multilingual speech recognition, including English.",
     "approx_mb": 202, "languages": "multilingual"},
    {"id": "tiny", "name": "Whisper tiny", "detail": "Smaller and faster, with lower accuracy.",
     "approx_mb": 129, "languages": "multilingual"},
]


def enabled() -> bool:
    from .native import process_tag
    return process_tag() == "win-arm64"


def model_path(model_id: str) -> Path:
    from .paths import data_dir
    if model_id not in {entry["id"] for entry in CATALOG}:
        raise ValueError(f"Whisper model '{model_id}' is not available in the ARM64 build.")
    root = data_dir() / "onnx-models" / model_id
    if not all((root / name).is_file() and (root / name).stat().st_size > 0 for name in FILES):
        raise FileNotFoundError(f"Whisper {model_id} needs to be downloaded before it can start.")
    return root


def total_bytes(model_id: str) -> int:
    from huggingface_hub import HfApi
    wanted = {f"{model_id}/{name}" for name in FILES}
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    return sum(file.size or 0 for file in info.siblings if file.rfilename in wanted)


def download(model_id: str, on_progress=None) -> str:
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm
    from .paths import data_dir
    if model_id not in {entry["id"] for entry in CATALOG}:
        raise ValueError(f"Whisper model '{model_id}' is not available in the ARM64 build.")
    total = total_bytes(model_id)
    done = 0
    lock = threading.Lock()

    class Progress(tqdm):
        def __init__(self, *args, **kwargs):
            self.counts_bytes = kwargs.get("unit") == "B" and "reconstruct" not in str(kwargs.get("desc", "")).lower()
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            nonlocal done
            result = super().update(n)
            if on_progress and n and self.counts_bytes:
                with lock:
                    done = min(total, done + int(n))
                    on_progress(done, total)
            return result

    snapshot_download(REPO, revision=REVISION,
                      allow_patterns=[f"{model_id}/{name}" for name in FILES],
                      local_dir=str(data_dir() / "onnx-models"), tqdm_class=Progress)
    path = model_path(model_id)
    if on_progress:
        on_progress(total, total)
    return str(path)
