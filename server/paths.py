"""Filesystem locations, split by whether they're read-only or writable.

An MSIX package's install directory is read-only, so anything the app writes at runtime
must live under LocalAppData. Keeping that split explicit here means the same code runs
unpackaged from a source checkout and packaged as MSIX with no changes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "Sunno"

# Read-only: ships with the app (models, UI assets).
INSTALL_ROOT = Path(__file__).resolve().parent.parent


def is_packaged() -> bool:
    """True when running from an MSIX package (which has package identity)."""
    return bool(os.environ.get("MSIX_PACKAGE_ROOT")) or "WindowsApps" in str(INSTALL_ROOT)


def data_dir() -> Path:
    """Writable per-user directory for profiles, caches and logs."""
    override = os.environ.get("Sunno_DATA_DIR")
    if override:
        path = Path(override)
    elif sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        path = Path(base) / APP_NAME
    else:
        path = Path.home() / f".{APP_NAME.lower()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def bundled_model(name: str) -> Path:
    """A model shipped with the app (read-only)."""
    return INSTALL_ROOT / "models" / name


def speaker_profiles_path() -> Path:
    """Where named speaker profiles are persisted (writable)."""
    return data_dir() / "speakers.json"


def ui_dir() -> Path:
    return INSTALL_ROOT / "ui"
