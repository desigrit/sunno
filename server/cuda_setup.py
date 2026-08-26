"""Locate and register the NVIDIA CUDA DLLs with Windows.

Must be imported before ctranslate2 / faster_whisper. Python 3.8+ on Windows no longer
searches PATH for dependent DLLs, so the cuBLAS directories have to be registered
explicitly with ``os.add_dll_directory``.

The payload is no longer shipped inside the package. It used to be 828 MB of the MSIX, 61%
of the whole thing, carried by every machine including the AMD and Intel ones that can never
load it. It is now downloaded on demand into LocalAppData, so this module looks in two
places: the interpreter's own site-packages, which is where pip puts it for a source
checkout, and the downloaded location. Whichever is complete wins.

Completeness matters more than it used to. While the payload shipped in the package,
``stage-backend.ps1`` failed the build if any allow-listed DLL was missing, so the payload
was all-or-nothing by construction. A download can be interrupted, so that guarantee is gone
and this module has to re-establish it: ``payload_present`` checks every file in the manifest
by name and size, and the downloader writes its sentinel only once they have all landed.

A directory-level check would not be enough. The version this replaces passed as soon as a
single ``<pkg>/bin`` folder existed, which a half-finished download produces routinely, and
the cost of getting that wrong is not a clear error at startup: CTranslate2 loads cuBLAS
lazily, so a partial payload gets as far as the first real decode before it fails.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

# Written by the downloader only after every file in the manifest has landed and verified.
SENTINEL = ".payload-complete"

MANIFEST = Path(__file__).resolve().parent / "cuda_manifest.json"

_registered_root: Path | None = None


def manifest() -> list[dict]:
    """The allow-listed DLLs with their expected size and digest.

    Generated at build time from packaging/cuda_allowlist.txt, and written into server/
    because the staging script copies only server/ and ui/ — the allow-list itself is not on
    disk at runtime.
    """
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError):
        return []


def download_root() -> Path:
    """Where an on-demand payload is installed. Mirrors the site-packages layout so the
    registration below does not care which root it ended up using."""
    from .paths import data_dir

    return data_dir() / "cuda" / "nvidia"


def candidate_roots() -> list[Path]:
    """Every place a payload could live, most-preferred first.

    site-packages first, so a source checkout that pip-installed the wheels keeps using them
    and never downloads something it already has.
    """
    return [
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia",
        download_root(),
    ]


def payload_present(root: Path | None = None) -> bool:
    """True when a root holds every allow-listed DLL at its expected size.

    Size rather than a full hash: the digests are verified once, at download time, and
    re-hashing 828 MB on every launch would add real seconds to a startup the UI already
    advertises as about half a minute, on files an antivirus scanner is also reading. Size
    catches the failure that actually occurs here, which is a truncated or missing file.
    """
    files = manifest()
    if not files:
        return False
    roots = [root] if root is not None else candidate_roots()
    dl_root = download_root()
    for candidate in roots:
        if not candidate.is_dir():
            continue
        # pip leaves no sentinel, so the marker is only demanded of a root we wrote.
        if candidate == dl_root and not (candidate / SENTINEL).exists():
            continue
        ok = True
        for f in files:
            p = candidate / f["path"]
            if not p.is_file() or p.stat().st_size != f["size"]:
                ok = False
                break
        if ok:
            return True
    return False


def nvidia_root() -> Path | None:
    """The first complete payload, or None when there is not one."""
    for candidate in candidate_roots():
        if payload_present(candidate):
            return candidate
    return None


def register_cuda_dlls(required: bool = False) -> list[Path]:
    """Add the NVIDIA DLL directories to the Windows DLL search path.

    Args:
        required: when True, a missing payload raises instead of warning. The server passes
            this for CUDA runs so a broken payload is caught at startup rather than at the
            first decode.

    Registration is keyed on which root was registered, not on a bare "have we run" flag.
    The version this replaces set that flag inside its failure branch, so once the
    import-time call found nothing, a later successful call returned early and never reached
    ``os.add_dll_directory``: the DLLs were never registered and the process went on to die
    at the first decode. That mattered little when a payload could only ever be present at
    startup. It matters now that one can arrive while the process is running.

    A payload that lands mid-process still does not enable GPU for that process — the engine
    is already built against the device chosen at startup. It takes effect on the next
    launch. Nothing here tries to hot-swap a running engine.
    """
    global _registered_root

    root = nvidia_root()
    if root is None:
        searched = ", ".join(str(p) for p in candidate_roots())
        message = (
            "NVIDIA CUDA libraries not found or incomplete. Searched: "
            f"{searched}. Sunno will use the processor instead."
        )
        # Deliberately not memoised on the failure path, so a later call can still succeed.
        if required:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return []

    if _registered_root == root:
        return []

    bin_dirs: list[Path] = []
    for f in manifest():
        d = (root / f["path"]).parent
        if d not in bin_dirs:
            bin_dirs.append(d)

    added: list[Path] = []
    for bin_dir in bin_dirs:
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        added.append(bin_dir)

    _registered_root = root
    return added


if sys.platform == "win32":
    register_cuda_dlls()
