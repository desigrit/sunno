# Third-party notices

Sunno redistributes the components below. Each remains under its own licence.

## Python packages

| Component | Version | Licence |
|---|---|---|
| PyAudioWPatch | 0.2.12.8 | Apache-2.0 license |
| av | 18.0.0 | BSD-3-Clause |
| ctranslate2 | 4.8.1 | MIT |
| faster-whisper | 1.2.1 | MIT |
| huggingface-hub | 1.25.1 | Apache-2.0 |
| numpy | 2.5.1 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| onnxruntime | 1.28.0 | MIT License |
| sherpa-onnx | 1.13.4 | Apache licensed, as found in the LICENSE file |
| sounddevice | 0.5.5 | MIT |
| soxr **(copyleft — see notes)** | 1.1.0 | LGPL-2.1-or-later |
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

## LGPL note: soxr

`soxr` (the SoX Resampler binding) is **LGPL-2.1-or-later**, and is the only
component here under a copyleft licence. Sunno uses it to convert audio from the
sample rate a capture device runs at to the 16 kHz the speech model requires.

The prebuilt wheel links libsoxr **statically** into `soxr_ext.pyd`, so LGPL-2.1
section 6 applies: a recipient must be able to modify the library and relink.
That is satisfied here because Sunno is distributed as complete source, including
the packaging scripts, so anyone can substitute a modified soxr and rebuild.

- soxr source: https://github.com/dofuuz/python-soxr
- libsoxr source: https://sourceforge.net/projects/soxr/
- LGPL-2.1 text: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html

Linking an LGPL library does not place Sunno itself under the LGPL; that is the
difference between the LGPL and the GPL, and is what the LGPL exists for.

## MPL note: tqdm

`tqdm` is dual-licensed **MPL-2.0 AND MIT**. MPL-2.0 is file-level copyleft: it
requires that modifications to MPL-covered files be published, and that the licence
travel with them. Sunno uses tqdm unmodified, and ships as source, so nothing further
is required — but it is listed here rather than lumped in with the permissive
dependencies, because it is not one.

- tqdm source: https://github.com/tqdm/tqdm
- MPL-2.0 text: https://mozilla.org/MPL/2.0/
