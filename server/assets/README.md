# Vendored assets

## `silero_vad_v6.onnx`

Silero VAD v6, the voice-activity model `server/vad.py` drives frame by frame.

**Why it lives here rather than being imported.** It ships inside `faster_whisper/assets/`, and
reaching it through `faster_whisper.utils.get_assets_path()` executes that package's `__init__`,
which imports `ctranslate2`. CTranslate2 publishes no `win_arm64` wheel and will not build there,
so on Windows on ARM `pip install faster-whisper` cannot succeed at all, meaning no
`faster_whisper/assets/` tree exists to find. Voice detection needs none of CTranslate2 and
should not be unable to start without it.

`stage-backend.ps1` copies `server/` recursively, so a copy here travels into the MSIX by itself.

**Provenance.** Taken from `faster-whisper` (MIT), which redistributes it from
[snakers4/silero-vad](https://github.com/snakers4/silero-vad) (MIT). Both licences permit
redistribution. 1,216 KB, sha256 `4cbf549b8326f60f80f2536d9eefeb45…`.
