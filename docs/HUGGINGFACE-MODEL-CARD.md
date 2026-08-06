---
license: mit
library_name: onnxruntime-genai
pipeline_tag: automatic-speech-recognition
tags:
  - whisper
  - onnx
  - onnxruntime-genai
  - windows-arm64
base_model:
  - openai/whisper-base
  - openai/whisper-tiny
---

# Sunno Whisper models

These are reproducible ONNX Runtime GenAI conversions used by
[Sunno](https://github.com/desigrit/sunno), an offline live-captioning app.

| Directory | Upstream model | Precision | Intended runtime |
|---|---|---|---|
| `base/` | `openai/whisper-base` | int8 | ONNX Runtime GenAI on CPU |
| `tiny/` | `openai/whisper-tiny` | int8 | ONNX Runtime GenAI on CPU |

The files were generated directly from OpenAI's MIT-licensed Whisper weights with
`bench/convert_whisper_genai.py` in the Sunno repository. They are not copies of a
third-party ONNX conversion.

Each directory is a complete model and contains `genai_config.json`, the audio
processor and tokenizer configuration, and the encoder and decoder ONNX graphs with
their external data files.

## Reproduce

```powershell
python -m venv .venv-convert
.venv-convert\Scripts\python.exe -m pip install onnxruntime-genai onnx onnx_ir transformers torch
.venv-convert\Scripts\python.exe bench\convert_whisper_genai.py --model base tiny --out dist\onnx
```

The conversion targets the CPU execution provider because that is the native path on
Windows ARM64. Sunno performs all inference locally after the initial model download.
