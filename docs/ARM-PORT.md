# Sunno on Windows ARM64

Release 1.0.92.0 includes a native ARM64 package with the same app identity as the x64 and
Microsoft Store editions. The previous ARM preview used a separate package identity.

## Runtime and features

The WinUI shell, .NET runtime, CPython 3.12.10 interpreter and native Python extensions are
ARM64. Packaging checks every PE binary and rejects foreign executable code.

Whisper base and tiny use ONNX Runtime GenAI with int8 weights from
[desigrit/Sunno](https://huggingface.co/desigrit/Sunno). Downloads are pinned to revision
`f57a056cc4f7044e7528af439f69e8deb8c86c03`. The cache is separate from CTranslate2 models;
a model is available only when all required files are present. Old selections that cannot
run natively migrate to base. The picker also offers Zipformer and Kroko through sherpa-onnx.

The release includes microphone and system-audio capture, device switching and recovery,
speaker labels, recording, appearance settings, caption scroll following and GitHub updates.
Capture uses the current Windows Core Audio worker, including its process isolation and
stable endpoint identifiers. PyAV's stateful 256-tap resampler replaces soxr on ARM64.

CTranslate2, Moonshine and soxr have no Windows ARM64 wheels in this release. Native Whisper
captions have no per-word confidence or clarity score. The x64 package remains available
for its larger Whisper models and Moonshine through Windows emulation.

## Build and publish

```powershell
.\packaging\build-msix.ps1 -Architecture arm64 -Store
```

The build cross-compiles from x64, installs hash-locked ARM64 wheels from
`requirements-arm64.txt`, and stages into `packaging/staging/release-arm64`. This leaves the
x64 staging and versioned release asset intact. Embedded Python explicitly includes the
backend root in its isolated module search path, which is required by the app and capture workers.

The Store package is `packaging/out/store/Sunno_1.0.92.0_arm64_Store.msix`. GitHub uses the
same bytes named `Sunno_1.0.92.0_arm64.msix`, matching the in-app updater's asset selector.
Both architecture packages can be submitted in the same Partner Center submission.
The Store profile excludes the experimental Windows AI Speech helper and `systemAIModels`.

## Validation

The release is cross-built on x64. Checks cover the ARM catalog and cache, migration of old
model selections, engine-specific latency measurements, resampling quality, capture recovery,
updater architecture selection, package identity, native binary architecture and signature.
Both GenAI Whisper models decode the speech fixture using the equivalent x64 runtime.
These checks do not execute ARM64 code on the build host.

On a Snapdragon, verify first launch, base and tiny downloads, microphone captions,
system-audio captions through a pause and resume, device switching, speaker labels, recording,
scroll following and Settings > Check for updates. This release has not yet completed that
hardware pass.

## Existing measurements

The earlier native Snapdragon X Elite measurements in `bench/results/` were:

| Model | Balanced CPU | Best Performance CPU |
|---|---:|---:|
| Whisper tiny | 203 ms | 132 ms |
| Whisper base | 519 ms | 297 ms |
| Whisper small | 1580 ms | 987 ms |
| Whisper medium | 5032 ms | 3331 ms |

These are historical decode measurements, not a new benchmark of 1.0.92.0. Base and tiny
use the Balanced figures as initial estimates, then learn from actual use. Native ONNX
measurements have separate cache keys so old emulated CTranslate2 timings do not override them.
