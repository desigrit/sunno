"""Register the pip-installed NVIDIA CUDA DLLs with Windows.

Must be imported before ctranslate2 / faster_whisper. Python 3.8+ on Windows no longer
searches PATH for dependent DLLs, so the cuBLAS and cuDNN directories shipped inside the
venv have to be registered explicitly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_NVIDIA_DLL_SUBDIRS = ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime")

_registered = False


def register_cuda_dlls() -> list[Path]:
    """Add bundled NVIDIA DLL directories to the Windows DLL search path."""
    global _registered
    added: list[Path] = []
    if _registered or sys.platform != "win32":
        return added

    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia_root.is_dir():
        _registered = True
        return added

    for subdir in _NVIDIA_DLL_SUBDIRS:
        bin_dir = nvidia_root / subdir / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            added.append(bin_dir)

    _registered = True
    return added


register_cuda_dlls()
