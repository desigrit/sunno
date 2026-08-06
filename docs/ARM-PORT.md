# Sunno on ARM64 — engineering notes

Status of the native ARM64 port, for anyone picking it up. Everything below is measured or
verified unless it says otherwise.

---

## The goal

A native ARM64 build of Sunno for Snapdragon X laptops (Surface Laptop), same UI and controls,
different engine and models underneath.

## Decisions already taken

| Decision | Rationale |
|---|---|
| **Latency beats accuracy on ARM** | A caption that arrives late is worse than a caption with a wrong word in it. |
| **No Prism emulation as a strategy** | Emulation would be tuned to whichever laptop it was tested on, and most ARM machines are weaker. |
| **Clarity % can go on ARM** | The confidence badge is worth less than the latency it would cost to compute. |

So the ARM tier is **base (297 ms)** with **tiny (132 ms)** below it. `small` (987 ms)
and the NPU path (1132 ms) are too slow to be defaults. Design for a mid-range Snapdragon on
battery in Balanced mode, not a fast one plugged in.

**Still undecided:** per-word uncertainty (`WordInlines.cs:135-144` greys words below
`UncertainBelow = 0.55` on *all* lines — a different feature from Clarity, which shows only on
your own lines, and arguably more valuable to a hard-of-hearing user).

Speaker labelling is off in the native ARM build because sherpa-onnx has no importable Windows
ARM64 artifact. The owner chose not to show a warning for an optional feature. System-audio
loopback remains available through SoundCard's CFFI WASAPI implementation.

---

## Measured numbers (Snapdragon X Elite, ARM64 native)

Mean decode, ms, against a 1000 ms responsiveness budget:

| model | CPU (Balanced) | CPU (Best Perf) | i9-14900K |
|---|---|---|---|
| tiny | 203 | **132** | 173 |
| base | 519 | **297** | 298 |
| small | 1580 | 987 | 846 |
| medium | 5032 | 3331 | — |

Power mode is worth ~1.6x. On Best Performance the Snapdragon **matches an i9-14900K** at
tiny/base. Raw data in `bench/results/`.

**NPU (Hexagon), whisper-small w8a16:** encoder 713.5 ms (fixed, once per utterance) + decode
step 17.46 ms × 24 = 1132 ms total. The decoder is excellent; the encoder dominates. Note the
probe round-trips ~27 MB of cross-attention cache to the host, which a real implementation would
keep on-device with IOBinding, so the encoder figure is pessimistic. Off-the-shelf *float* models
fall back to CPU entirely (4-11% slower, load up to 14x higher, identical transcripts).

---

## What is done

**`e2ad71e` — voice detection and the model list work without CTranslate2.**
`vad.py` and `models.py` no longer import anything under `faster_whisper` (importing it executes
`__init__` → `ctranslate2`). Silero weights vendored to `server/assets/`. `_REPOS` in `models.py`
carries all 19 model ids. `tests/test_model_repos.py` guards the copy against drift.

**`524f77e` — the engine seam.**
`server/engine.py` has the `SpeechEngine` protocol (four members: `settings`, `partial`, `final`,
`warmup`), the shared `Transcript`/`Word` types, and `create_engine()`. `server/asr.py` holds
`CTranslate2Engine`; `server/asr_onnx.py` holds `OnnxEngine`. Consumers are `app.py:427` and
`pipeline.py:22,84,208`.

**`bench/convert_whisper_genai.py` — produces the ARM models.**
Converts `openai/whisper-*` (MIT) into genai format. **Verified working**: base at int8 decodes
real speech in 363 ms on x64. This removes the supply-chain problem — see below.

**The ONNX engine, ARM catalog, native audio dependencies, backend staging and dual-architecture
package have been exercised.**
`create_engine(settings, "onnx")` decodes real speech through the published `desigrit/Sunno`
base model. The ARM picker offers base and tiny with measured lag and genai-shaped downloads.
PyAV's stateful `AudioResampler` measured 81.2 dB at both 44.1 and 48 kHz on the original
two-speaker corpus, against soxr HQ's 81.4-81.5 dB, and produced identical large-v3 transcripts.
SoundCard replaces pyaudiowpatch for WASAPI loopback; its complete dependency chain resolves to
CPython 3.12 `win_arm64` or pure-Python wheels. Native ARM silently omits speaker labelling.
`stage-backend.ps1 -Architecture arm64` produces a 162 MB tree with the ARM64 embeddable
interpreter, the redistributable ARM64 C++ runtime required by ONNX, and 140 ARM64 PE files
with zero x64 binaries. Executable inputs and every ARM wheel are version- and hash-locked.
`build-msix.ps1` publishes and validates both complete layouts, signs standalone x64 and ARM64
packages, and produces a signed 928 MB `Sunno.msixbundle` whose deterministic `1.0.60.0`
manifest contains exactly those two architectures.

---

## What is left

1. **Unwind the ARM refusal.** `c04e3fe` makes the app say *"Sunno's speech engine needs a
   64-bit Intel or AMD processor"* — correct for x64-only, wrong once an ARM build exists. See
   `hardware.py engine_importable()` and `BackendHost._engineUnloadableOnArm`.

---

## Landmines — these cost real time to find

**`build-msix.ps1:33`'s `\x64\` filter must NOT be templated.** It selects the **host**
`makeappx`/`signtool` on the dev box. Changing it breaks the build.

**MakeAppx cannot bundle already-signed child packages.** Pack both unsigned MSIX files, create
the bundle with an explicit `/bv` matching the package version, then sign the bundle and the two
standalone packages. Without `/bv`, MakeAppx silently invents a timestamp-derived bundle version.

**Cross-platform pip emits launchers for the build host.** `pip install --platform win_arm64
--target` creates x64 console executables under `site-packages/bin`; they are build-time entry
points, not runtime dependencies, and must be removed. Python 3.12.10's official ARM64
embeddable archive also carries an unused x64 `vcruntime140_1.dll`; no ARM64 PE imports it.

**Build proof venvs *from* `requirements.txt`, never by hand.** A hand-built venv is structurally
incapable of catching a broken requirements file — that is exactly how a deleted
`onnxruntime>=1.17` got through a review round.

**The QNN incantation is three steps and all are load-bearing:**
```python
ort.register_execution_provider_library(name, onnxruntime_qnn.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == name]
so.add_provider_for_devices(devices, {"backend_path": onnxruntime_qnn.get_qnn_htp_path()})
session = ort.InferenceSession(path, sess_options=so)   # NOT providers=[...]
```
`providers=["QNNExecutionProvider"]` only reaches EPs compiled into the build and fails with
*"not compatible with any execution provider added to the session"* even when the name appears in
`get_available_providers()`. Selecting by device without `backend_path` fails differently:
*"Could not determine default backend path for device"*.

**genai API shape:** `set_inputs` is on `Generator`, not `GeneratorParams`. The processor takes
`prompt=` singular. The Whisper prompt is
`<|startoftranscript|><|en|><|transcribe|><|notimestamps|>`.

**genai will not load onnx-community models** — they are Optimum/Transformers.js exports with no
`genai_config.json`. Use `bench/convert_whisper_genai.py`.

**The converter needs two undocumented workarounds** (both already handled in the script, but
know why): transformers 5.x asks the Hub with `token=True` even for public weights, so fetch
first with `huggingface_hub`; and the builder deletes its cache dir when empty while the Whisper
path saves twice, so keep a file in it.

**`cuda_setup.py:79`** raises a RuntimeWarning about missing NVIDIA libraries on every import in
a non-CUDA environment. Harmless, but it will be the first thing an ARM user sees and it is
misleading there.

**SoundCard and WASAPI are both COM-thread-affine.** Device enumeration runs on an HTTP worker
and capture runs on the pump thread, so each thread must call `CoInitializeEx` before touching
SoundCard. Persist the stable WASAPI endpoint id, not SoundCard's enumeration slot or the friendly
name; duplicate monitors and docks can have identical names.

---

## Working practices

- **Every change goes through a review pass before commit.** It has caught a shipping-severity
  bug in most rounds of this work — including one that would have failed on exactly the ARM model
  tier. Budget for two or three rounds.
- Commit messages are prose explaining *why*. See `git log`.
- Tests are directly-runnable scripts in `tests/`, not pytest.
- `packaging/stage-backend.ps1` uses an explicit include-list, deliberately, so a stray model
  cannot inflate the package.
