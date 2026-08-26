"""Model catalog and first-run download.

Weights are deliberately not shipped inside the app package: they are large, they are data
rather than code, and keeping them in the user's cache means an MSIX install stays small and
the download can be resumed. huggingface_hub handles resume and integrity, so this module
only adds a catalog, a local-availability check, and progress reporting.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Every model id the engine can load, and where it lives on the Hub.
#
# Mirrors faster_whisper.utils._MODELS, which cannot be read without importing faster_whisper
# and therefore ctranslate2 - a dependency this module needs for nothing else, and one with no
# wheel on Windows on ARM.
#
# Deliberately the whole table rather than the four ids the picker offers. --model takes any
# string (server/app.py) and so does the websocket download command, so an id missing from here
# would not fail cleanly: it would be passed to the Hub as a bare repo name, come back 401, and
# present as a download failure for a model already sitting in the cache.
_REPOS: dict[str, str] = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

# Offered at first run. Ordered best-accuracy-first.
#
# Descriptions say what a model is, never whether to pick it. Which one to pick depends on
# the machine, and the app measures that: hardware.estimated_lag_ms decides what keeps up
# here, and the UI groups and preselects on the answer. A fixed word cannot know. large-v3
# carried "Recommended." in its description, which was true on a GPU and wrong on every
# laptop, where the same screen had already measured it at five seconds behind speech and
# filed it under the models that cannot keep up.
CATALOG: list[dict] = [
    {
        "id": "large-v3",
        "name": "Whisper large-v3",
        "detail": "Best accuracy across accents.",
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
        "detail": "Much faster. Less reliable on accented speech.",
        "approx_mb": 490,
        "languages": "multilingual",
    },
    {
        "id": "base",
        "name": "Whisper base",
        "detail": "Fast enough for a laptop with no graphics card. Makes more mistakes.",
        "approx_mb": 145,
        "languages": "multilingual",
    },
    {
        "id": "stream-en",
        "name": "Zipformer",
        "detail": "Built for PCs with no graphics card. English only, lower case with no "
                  "punctuation.",
        "approx_mb": 69,
        "languages": "English",
    },
    # Placed after stream-en deliberately, and the order is load-bearing rather than
    # cosmetic. This list is otherwise best-accuracy-first, and Kroko is the more accurate
    # of the two, so its natural slot is above. It sits below because default_model scans
    # this list in order, and THIRD-PARTY-NOTICES.md tells the reader that neither model
    # without a declared licence is a default. Ordering alone would still hand it the
    # bottom-of-the-list fallback on a very slow machine, so the guarantee is enforced by
    # AUTO_SELECT_EXCLUDED below rather than by this comment.
    {
        "id": "stream-en-kroko",
        "name": "Kroko",
        "detail": "Built for PCs with no graphics card. English only, with capitals and "
                  "punctuation.",
        "approx_mb": 68,
        "languages": "English",
    },
]

# Models the app will never choose on someone's behalf.
#
# Not a quality judgement. These are the models whose publisher has declared no licence, and
# THIRD-PARTY-NOTICES.md states plainly that none of them is a default. A user who reads that
# and picks one anyway has chosen it; the app arriving at one on its own would make the
# disclosure false. Enforced in hardware.default_model and pinned by a test, because catalogue
# order alone does not survive the fallback branch when nothing meets the latency budget.
AUTO_SELECT_EXCLUDED: frozenset[str] = frozenset({"stream-en-kroko"})


def auto_selectable(model_id: str) -> bool:
    """Whether the app may choose this model when the user has not."""
    return model_id not in AUTO_SELECT_EXCLUDED

# Streaming transducer models, which are a different shape from a Whisper checkpoint: an
# encoder, a decoder and a joiner rather than one file. Kept apart from _REPOS so that a
# Whisper id can never resolve to half of one of these.
#
# The two differ in licence, and it is the sharpest difference in the catalogue. `stream-en`
# is Apache-2.0, declared in the repo's own metadata. `stream-en-kroko` has no declared
# licence at all, and the chain was followed rather than assumed: the repo that serves the
# ONNX files declares nothing (`cardData: null`) and its README says only "See license at
# .../Banafo/Kroko-ASR"; that repo declares `license: other` with `license_name: "test"` and
# `license_link: LICENSE`; that LICENSE file is zero bytes. CC-BY-SA appears only as prose in
# a README that splits models into community and commercial tiers without saying which tier
# this checkpoint is in. So the honest description is "no grant", not "ShareAlike".
#
# There is a second, separate gap. Banafo publishes `.data` files; the repo above serves
# `encoder.onnx` / `decoder.onnx` / `joiner.onnx`. Somebody converted them, unattributed, so
# the app fetches a third party's conversion rather than an artifact from the author. Even a
# clear licence from Banafo would not by itself cover that.
#
# It ships anyway, as a deliberate and recorded decision by the project owner, who was shown
# the above and accepted it. What that costs is not hidden: the model is never auto-selected
# (see AUTO_SELECT_EXCLUDED) and THIRD-PARTY-NOTICES.md carries a section stating the licence
# position plainly. The catalogue description used to say it too and no longer does, removed
# on the owner's instruction because it made that row read as a warning beside every other
# model; the notices are the place it is recorded. What it buys is readability rather than
# speed: it writes its own capitals and punctuation, and it revises text already on screen
# far less often, 44 caption refreshes in 251 against 147. On latency the two are near-tied,
# 120 ms against 135 on the same clip and thread count, so speed is not the argument for it.
# An earlier version of this comment claimed 80 against 130, which did not reproduce; see
# the note in hardware.py.
#
# No punctuation model yet. sherpa-onnx publishes a 7 MB streaming punctuation and casing
# model that would give `stream-en` sentence case, measured at 4.6 ms, but it also has no
# licence and only reaches this app as a GitHub release tarball, so it brings a download path
# with no resume or integrity check. Kroko already punctuates, which is most of the reason to
# want it. The candidate is `sherpa-onnx-online-punct-en-2024-08-06`, in the k2-fsa/sherpa-onnx
# `punctuation-models` release. No request for a licence has been sent from here, so nobody is
# waiting on a reply.
_STREAM_REPOS: dict[str, dict] = {
    "stream-en": {
        "repo": "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26",
        # Emits unbroken upper case with no punctuation, so the engine lower-cases it.
        "cased": False,
        "files": {
            "encoder": "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
            "decoder": "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
            "joiner": "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
            "tokens": "tokens.txt",
        },
    },
    "stream-en-kroko": {
        "repo": "csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
        # Writes its own capitals and punctuation. Lower-casing it would throw away the
        # main reason to prefer it, so _readable leaves this one alone.
        "cased": True,
        "files": {
            "encoder": "encoder.onnx",
            "decoder": "decoder.onnx",
            "joiner": "joiner.onnx",
            "tokens": "tokens.txt",
        },
    },
}


def is_stream_model(model_id: str) -> bool:
    return model_id in _STREAM_REPOS


def stream_model_is_cased(model_id: str) -> bool:
    """Whether a streaming model writes its own capitals and punctuation.

    Asked by the engine before it lower-cases. Defaults to False for an unknown id, which is
    the safe direction: lower-casing already-readable text is untidy, whereas leaving a wall
    of upper case on screen is the thing this is here to avoid.
    """
    spec = _STREAM_REPOS.get(model_id)
    return bool(spec and spec.get("cased"))


def _stream_root() -> Path:
    """Where streaming models are kept, beside the other caches."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / "Sunno" / "stream-models"


def stream_model_paths(model_id: str) -> dict:
    """Local paths for a streaming model, raising if it has not been downloaded.

    Raises rather than fetching, for the same reason ``onnx_model_path`` does: downloading
    is the download path's job, which reports progress, and doing it from inside engine
    construction would look like a very slow start.
    """
    spec = _STREAM_REPOS.get(model_id)
    if spec is None:
        raise KeyError(f"'{model_id}' is not a streaming model")

    root = _stream_root() / model_id
    paths: dict = {}
    for role, filename in spec["files"].items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"No streaming model at {root}. It has to be downloaded before the engine "
                "can start."
            )
        paths[role] = path

    return paths

# Deliberately not offered: whisper-tiny.
#
# It is the smallest model and it is not the fastest one here. On two-second clips it measured
# 870 ms against base's 420 - twice as slow - because it hallucinates on short audio and then
# spends real time decoding words nobody said. One run returned "Attention!" over speech that
# contained no such word.
#
# Short utterances are this app's entire workload, and a caption that invents a word is worse
# for someone relying on it than a caption that arrives a moment later. base is smaller than
# anything else offered and is genuinely quick, so tiny buys nothing and costs trust.

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
    """True when the model is already in the local cache, so no download is needed.

    Asks huggingface_hub directly rather than faster_whisper.utils.download_model. The two
    answer the same question over the same cache, but importing faster_whisper executes its
    __init__, which imports ctranslate2 - so a plain "is this downloaded?" check could take the
    whole backend down on a machine where CTranslate2 will not load, before anything had a
    chance to explain why.
    """
    from huggingface_hub import snapshot_download

    if is_stream_model(model_id):
        try:
            paths = stream_model_paths(model_id)
            return ModelStatus(model_id=model_id, available=True,
                               path=str(paths["encoder"].parent))
        except (FileNotFoundError, KeyError):
            return ModelStatus(model_id=model_id, available=False)

    try:
        path = snapshot_download(
            _repo_id(model_id),
            allow_patterns=_ALLOW_PATTERNS,
            local_files_only=True,
        )
        return ModelStatus(model_id, True, path)
    except Exception:
        return ModelStatus(model_id, False)


def catalog_with_status(device: str | None = None) -> list[dict]:
    """Catalog entries plus local availability and expected decode lag.

    The lag matters as much as the accuracy: on a CPU-only machine large-v3 runs about
    4.5 s behind, which is fine for captioning a recorded video and useless for following a
    conversation. Surfacing it here means the user learns that before downloading 3 GB
    rather than after.

    ``auto_select`` rides along because the frontend, not this process, is what actually
    chooses a model for a first-time user. hardware.default_model only runs when nobody
    passed --model, and both launchers always do, so a guard living only there guards a path
    the product never takes. Sending the flag with the catalogue puts the rule next to the
    data it constrains, where the screen that preselects can honour it.

    ``uses_gpu`` says whether an entry runs on the CUDA engine at all. The streaming models
    are pinned to CPU providers, so choosing one never triggers a 455 MB download. It is
    derived from is_stream_model rather than from an id prefix, because nothing else in the
    codebase treats "stream-" as meaningful and a second, parallel rule would drift.
    """
    from . import hardware

    device = device or hardware.resolve_device()
    entries = []
    for entry in CATALOG:
        lag_ms = hardware.estimated_lag_ms(entry["id"], device)
        entries.append(dict(
            entry,
            available=is_available(entry["id"]).available,
            lag_ms=lag_ms,
            lag_text=hardware.describe_lag(lag_ms),
            responsive=lag_ms <= hardware.RESPONSIVE_LAG_MS,
            auto_select=auto_selectable(entry["id"]),
            uses_gpu=not is_stream_model(entry["id"]),
        ))
    return entries


def _repo_id(model_id: str) -> str:
    """The Hub repo holding a model id.

    A path or an explicit "org/name" passes through, so a user-supplied model still works. An
    id that is neither known nor qualified is returned unchanged and will fail at the Hub -
    which is the same behaviour as before and is at least honest about not knowing it.
    """
    if "/" in model_id:
        return model_id
    return _REPOS.get(model_id, model_id)


def onnx_model_path(model_id: str) -> Path:
    """Where the ONNX build of a model lives locally.

    Separate from the CTranslate2 cache because they are different artifacts: a genai model is
    a directory of genai_config.json, the tokenizer, and an encoder/decoder pair, and nothing
    about it is interchangeable with a CT2 conversion of the same weights.

    Raises rather than downloading. Fetching several hundred megabytes is the download path's
    job, which reports progress; doing it silently from inside engine construction would look
    like a very slow startup.
    """
    root = _onnx_root() / model_id
    if (root / "genai_config.json").is_file():
        return root
    # snapshot_download keeps the repo's own layout, so the config may sit one level down.
    for nested in root.glob("*/genai_config.json"):
        return nested.parent
    raise FileNotFoundError(
        f"No ONNX build of '{model_id}' at {root}. It has to be downloaded before the engine "
        "can start."
    )


def _onnx_root() -> Path:
    """Where ONNX models are kept. Beside the CT2 cache, under the user's local app data."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / "Sunno" / "onnx-models"


def total_size_bytes(model_id: str) -> int:
    """Exact download size from the Hub, so progress is real rather than estimated."""
    import fnmatch

    from huggingface_hub import HfApi

    if is_stream_model(model_id):
        return _stream_total_size_bytes(model_id)

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


def _stream_total_size_bytes(model_id: str) -> int:
    """Size of just the files a streaming model needs.

    A whole-repo figure would be wrong here: these repos carry several quantisations and
    chunk configurations side by side, and only one set is fetched, so quoting the repo
    would tell someone to expect several times what actually downloads.
    """
    from huggingface_hub import HfApi

    spec = _STREAM_REPOS[model_id]
    api = HfApi()
    wanted = set(spec["files"].values())

    try:
        info = api.model_info(spec["repo"], files_metadata=True)
    except Exception:
        entry = next((e for e in CATALOG if e["id"] == model_id), None)
        return int(entry["approx_mb"] * 1024 * 1024) if entry else 0

    total = 0
    for sibling in info.siblings or []:
        if sibling.rfilename in wanted:
            total += sibling.size or 0

    return total


def _download_stream(model_id: str, on_progress: ProgressFn | None = None) -> str:
    """Fetch the handful of files a streaming model needs.

    Named files rather than a snapshot, because these repos hold several quantisations and
    chunk configurations together and a snapshot would pull hundreds of megabytes that are
    never loaded.
    """
    from huggingface_hub import hf_hub_download

    spec = _STREAM_REPOS[model_id]
    root = _stream_root() / model_id
    root.mkdir(parents=True, exist_ok=True)

    total = total_size_bytes(model_id)
    done = 0

    jobs = [(spec["repo"], name, root) for name in spec["files"].values()]

    for repo, filename, target_dir in jobs:
        cached = hf_hub_download(repo_id=repo, filename=filename)
        target = target_dir / filename
        if not target.is_file():
            import shutil

            # Copied out of the hub cache rather than linked, because the cache is prunable
            # and a model the engine cannot open is a worse failure than a duplicated file.
            #
            # Written beside the target and renamed, because a copy interrupted partway
            # leaves a truncated file that nothing repairs: stream_model_paths only asks
            # whether the file exists, so the next launch would report the model ready and
            # then fail inside ONNX with an error about the file format. A rename either
            # happened or did not.
            temporary = target.with_suffix(target.suffix + ".part")
            shutil.copyfile(cached, temporary)
            temporary.replace(target)
        done += target.stat().st_size
        if on_progress:
            on_progress(min(done, total), total)

    if on_progress:
        on_progress(total, total)
    return str(root)


def download(model_id: str, on_progress: ProgressFn | None = None) -> str:
    """Download a model, reporting cumulative bytes. Resumes an interrupted download."""
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    if is_stream_model(model_id):
        return _download_stream(model_id, on_progress)

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
