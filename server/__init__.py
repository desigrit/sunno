"""Offline live-captioning server package."""

import os

# Disposable audio children never load recognition or GPU libraries.
if os.environ.get("SUNNO_AUDIO_CHILD") != "1":
    from . import cuda_setup  # noqa: F401  (registers CUDA DLL dirs on import)
