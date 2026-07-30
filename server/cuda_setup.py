"""Register the pip-installed NVIDIA CUDA DLLs with Windows.

Must be imported before ctranslate2 / faster_whisper. Python 3.8+ on Windows no longer
searches PATH for dependent DLLs, so the cuBLAS directories shipped alongside the
interpreter have to be registered explicitly.

This is layout-sensitive: it expects ``<prefix>/Lib/site-packages/nvidia/<pkg>/bin``. The
MSIX staging script deliberately preserves that shape. If it ever stops matching, the
failure surfaces here as a clear warning rather than as an opaque ctranslate2 DLL load
error several seconds later.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

_NVIDIA_DLL_SUBDIRS = ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime")

_registered = False


def nvidia_root() -> Path:
    return Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"


def register_cuda_dlls(required: bool = False) -> list[Path]:
    """Add bundled NVIDIA DLL directories to the Windows DLL search path.

    Args:
        required: when True, a missing or empty NVIDIA payload raises instead of warning.
            The server passes this for CUDA runs so a mis-staged package fails loudly at
            startup rather than mid-conversation.
    """
    global _registered
    added: list[Path] = []
    if _registered or sys.platform != "win32":
        return added

    root = nvidia_root()
    if not root.is_dir():
        message = (
            f"NVIDIA CUDA libraries not found at {root}. GPU inference will fail. "
            "If this is a packaged build, the staging step did not preserve "
            "Lib/site-packages/nvidia/<pkg>/bin."
        )
        _registered = True
        if required:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return added

    for subdir in _NVIDIA_DLL_SUBDIRS:
        bin_dir = root / subdir / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            added.append(bin_dir)

    if not added:
        message = (
            f"NVIDIA directory {root} exists but contains no <pkg>/bin folders; "
            "GPU inference will fail."
        )
        _registered = True
        if required:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return added

    _registered = True
    return added


register_cuda_dlls()
