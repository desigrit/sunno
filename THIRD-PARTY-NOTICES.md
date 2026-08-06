# Third-party notices

Sunno redistributes the components below. Each remains under its own licence.

## Python packages

| Component | Version | Licence |
|---|---|---|
| av | 18.0.0 | BSD-3-Clause |
| cffi | 2.1.1 | MIT |
| ctranslate2 | 4.8.1 | MIT |
| faster-whisper | 1.2.1 | MIT |
| huggingface-hub | 1.25.1 | Apache-2.0 |
| numpy | 2.5.1 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| onnxruntime | 1.28.0 | MIT License |
| pycparser | 3.0 | BSD-3-Clause |
| sherpa-onnx | 1.13.4 | Apache licensed, as found in the LICENSE file |
| SoundCard | 0.4.6 | BSD-3-Clause |
| sounddevice | 0.5.5 | MIT |
| tokenizers | 0.23.1 | Apache Software License |
| tqdm **(copyleft — see notes)** | 4.70.0 | MPL-2.0 AND MIT |
| websockets | 17.0 | BSD-3-Clause |

## Native and model components

| Component | Licence | Notes |
|---|---|---|
| NVIDIA CUDA runtime (cuBLAS, NVRTC) | NVIDIA CUDA Toolkit EULA | Redistributable components only, per the EULA's distribution list. https://docs.nvidia.com/cuda/eula/ |
| CPython 3.12 | Python Software Foundation License 2.0 | https://docs.python.org/3/license.html |
| Whisper model weights | MIT (OpenAI) | Downloaded at first run, not redistributed in the package. |
| Silero VAD | MIT | Vendored at `server/assets/silero_vad_v6.onnx` and redistributed in the package. From [snakers4/silero-vad](https://github.com/snakers4/silero-vad). |
| WeSpeaker CAM++ speaker embedding | Apache-2.0 | Downloaded at first run, not redistributed in the package. |

## MPL note: tqdm

`tqdm` is dual-licensed **MPL-2.0 AND MIT**. MPL-2.0 is file-level copyleft: it
requires that modifications to MPL-covered files be published, and that the licence
travel with them. Sunno uses tqdm unmodified, and ships as source, so nothing further
is required — but it is listed here rather than lumped in with the permissive
dependencies, because it is not one.

- tqdm source: https://github.com/tqdm/tqdm
- MPL-2.0 text: https://mozilla.org/MPL/2.0/
