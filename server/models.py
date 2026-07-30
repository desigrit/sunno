"""Model catalog and first-run download.

Weights are deliberately not shipped inside the app package: they are large, they are data
rather than code, and keeping them in the user's cache means an MSIX install stays small and
the download can be resumed. huggingface_hub handles resume and integrity, so this module
only adds a catalog, a local-availability check, and progress reporting.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

# Offered at first run. Ordered best-first; the UI marks the first as recommended.
CATALOG: list[dict] = [
    {
        "id": "large-v3",
        "name": "Whisper large-v3",
        "detail": "Best accuracy across accents. Recommended.",
        "approx_mb": 3090,
        "languages": "multilingual",
    },
    {
        "id": "distil-large-v3",
        "name": "Distil-Whisper large-v3",
        "detail": "About half the size and faster, English only.",
        "approx_mb": 1520,
        "languages": "English",
    },
    {
        "id": "medium",
        "name": "Whisper medium",
        "detail": "Noticeably less accurate on accented speech.",
        "approx_mb": 1530,
        "languages": "multilingual",
    },
    {
        "id": "small",
        "name": "Whisper small",
        "detail": "Fastest and smallest. Struggles with accents and distance.",
        "approx_mb": 490,
        "languages": "multilingual",
    },
]

_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]

ProgressFn = Callable[[int, int], None]


@dataclass
class ModelStatus:
    model_id: str
    available: bool
    path: str | None = None


def is_available(model_id: str) -> ModelStatus:
    """True when the model is already in the local cache, so no download is needed."""
    from faster_whisper.utils import download_model

    try:
        path = download_model(model_id, local_files_only=True)
        return ModelStatus(model_id, True, path)
    except Exception:
        return ModelStatus(model_id, False)


def catalog_with_status() -> list[dict]:
    return [dict(entry, available=is_available(entry["id"]).available) for entry in CATALOG]


def _repo_id(model_id: str) -> str:
    from faster_whisper.utils import _MODELS

    return model_id if "/" in model_id else _MODELS.get(model_id, model_id)


def total_size_bytes(model_id: str) -> int:
    """Exact download size from the Hub, so progress is real rather than estimated."""
    import fnmatch

    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(_repo_id(model_id), files_metadata=True)
    except Exception:
        entry = next((e for e in CATALOG if e["id"] == model_id), None)
        return int(entry["approx_mb"] * 1024 * 1024) if entry else 0

    total = 0
    for sibling in info.siblings or []:
        if any(fnmatch.fnmatch(sibling.rfilename, p) for p in _ALLOW_PATTERNS):
            total += sibling.size or 0
    return total


def download(model_id: str, on_progress: ProgressFn | None = None) -> str:
    """Download a model, reporting cumulative bytes. Resumes an interrupted download."""
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    total = total_size_bytes(model_id)
    state = {"done": 0}
    lock = threading.Lock()

    class _ReportingTqdm(tqdm):
        """Aggregates huggingface_hub's progress bars into a single byte count.

        The hub creates several bars per snapshot: one for the network download, one for
        local Xet reassembly ("Reconstructing"), and one counting files. Summing all of them
        double-counts and reports ~200%, so only true byte-transfer bars are counted.
        """

        def __init__(self, *args, **kwargs):
            desc = str(kwargs.get("desc") or "")
            unit = kwargs.get("unit")
            # Count only byte-denominated transfer bars, excluding post-processing phases.
            self._counts = unit == "B" and "reconstruct" not in desc.lower()
            kwargs["disable"] = True  # keep the console clean; we report over the socket
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            result = super().update(n)
            if on_progress and n and getattr(self, "_counts", False):
                with lock:
                    # Clamp: a future hub version adding another byte bar must not
                    # produce >100%.
                    state["done"] = min(state["done"] + int(n), total) if total else state["done"] + int(n)
                    done = state["done"]
                on_progress(done, total)
            return result

    path = snapshot_download(
        _repo_id(model_id),
        allow_patterns=_ALLOW_PATTERNS,
        tqdm_class=_ReportingTqdm,
    )
    if on_progress:
        on_progress(total, total)
    return path
