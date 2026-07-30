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

    The ``required=True`` check is deliberately NOT short-circuited by the memo: this module
    is imported (and therefore registers with ``required=False``) before the server decides
    whether it needs CUDA, so an early-return on ``_registered`` would make the strict check
    unreachable — which is exactly the bug this guard exists to prevent.
    """
    global _registered

    root = nvidia_root()
    dll_dirs = [root / sub / "bin" for sub in _NVIDIA_DLL_SUBDIRS]
    present = [d for d in dll_dirs if d.is_dir()]

    if not root.is_dir() or not present:
        message = (
            f"NVIDIA CUDA libraries not found at {root}. GPU inference will fail. "
            "If this is a packaged build, the staging step did not preserve "
            "Lib/site-packages/nvidia/<pkg>/bin."
            if not root.is_dir() else
            f"NVIDIA directory {root} exists but contains no <pkg>/bin folders; "
            "GPU inference will fail."
        )
        _registered = True
        if required:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return []

    if _registered:
        # Already on the search path; the presence check above still ran, so a strict
        # caller has been given a real answer rather than a memoised one.
        return []

    added: list[Path] = []
    for bin_dir in present:
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        added.append(bin_dir)

    _registered = True
    return added


if sys.platform == "win32":
    register_cuda_dlls()
