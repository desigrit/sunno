# Live Captions

Offline, real-time speech-to-text captioning. Audio never leaves the machine — no network
calls, no API keys, no per-minute cost.

Built for live captioning of in-person conversation: a room microphone feeds Whisper
large-v3 on the local GPU, and captions appear in a small always-on-top window.

```
Room mic ──► resample 16 kHz ──► Silero VAD ──► Whisper large-v3 (FP16, CUDA)
                                                   │
                                    partial (greedy, ~250 ms)   final (beam=5)
                                                   │
                                          WebSocket JSON
                                                   │
                                    ┌──────────────┴──────────────┐
                                    ▼                             ▼
                          always-on-top window            handheld / phone
                             (this PC, today)            (same page over LAN)
```

The UI is a plain web page talking to the server over a WebSocket. It knows nothing about
machine learning, so moving the display to a separate handheld screen later needs no rewrite —
just load the same page over the network while this PC keeps doing the inference.

## Quick start

```powershell
.\run.cmd -ListDevices     # find your microphone's index
.\run.cmd -Device 27       # start captioning
```

First run downloads Whisper large-v3 (~3 GB) and takes ~30 s to load. Later starts are faster.

To view captions from another device on your network:

```powershell
.\run.cmd -Device 27 -Lan  # prints the URL to open on the other device
```

> **Which shell?** In **PowerShell 7** (`pwsh`), `.\run.ps1` works directly — its default
> execution policy is `RemoteSigned`. In **Windows PowerShell 5.1** (the blue icon) the default
> is `Restricted`, which refuses `.ps1` files; use `.\run.cmd` there instead. `run.cmd` works in
> both shells and when double-clicked from File Explorer, so it's the safe default. Note that
> `Get-ExecutionPolicy` can mislead: the two shells read *different* registry keys, so one can
> allow scripts while the other blocks them.

## Setup

Requires Python 3.12 (python.org build — **not** the Microsoft Store build, which is
sandboxed and unreliable at loading native CUDA DLLs) and an NVIDIA GPU.

```powershell
winget install --id Python.Python.3.12 --scope user
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Usage

| Command | Effect |
|---|---|
| `.\run.ps1` | Start with the default input device |
| `.\run.ps1 -Device 27` | Use a specific input device |
| `.\run.ps1 -ListDevices` | List available input devices |
| `.\run.ps1 -Lan` | Also serve the UI to other devices on the network |
| `.\run.ps1 -NoWindow` | Run the server only, without opening a window |
| `.\run.ps1 -Model small` | Use a smaller/faster model |

In the caption window:

| Control | Action |
|---|---|
| **Stop / Start** (or **Space**) | Stop or resume transcribing |
| **Click a speaker chip** | Name that person, mark them as you, or merge two speakers |
| `A+` / `A−` (or Ctrl `+` / Ctrl `−`) | Change text size — remembered between runs |
| `Clear` | Clear the transcript |

**Stop genuinely releases the microphone** — it closes the audio device rather than quietly
discarding audio, so Windows' microphone-in-use indicator switches off and people in the room
can see that capture has stopped. The Whisper model stays loaded on the GPU, so resuming takes
well under a second. Audio captured mid-sentence when you press Stop is discarded, not
transcribed. Launch with `--start-stopped` to begin paused.

### Speaker labels

Every finished utterance is matched against the voices heard so far, and lines are prefixed
with a colour-coded speaker chip. Click any chip to:

- **Name the speaker** — names persist across sessions, and a named profile stops drifting, so
  recognition of that person gets more reliable.
- **Mark them as you** — your own lines then render fainter, labelled *You*, with a **clarity
  score**. Your speech is still transcribed; the score is there so you can read back how clearly
  you came across.
- **Merge two speakers** — use this when one person gets split across two labels.

**Accuracy expectations.** On a two-speaker recording this correctly finds two speakers and gets
roughly two thirds of the turns right. It is genuinely useful for following who-said-what, but
it is not reliable enough to trust blindly. Two reasons, both inherent:

- Speaker embeddings need ~2–3 s of speech to be dependable; short conversational turns
  ("yeah", "okay, fine") often score no better than chance. Such turns are left unlabelled
  rather than guessed at.
- If two people talk without a pause between them, the VAD produces one segment and it gets a
  single label. Lowering `end_silence_ms` reduces how often this happens.

Naming the people you talk to most is the single biggest improvement, because matching then
happens against a deliberately-chosen reference instead of a noisy first guess.

### Clarity score

Lines marked as yours carry a 0–100 clarity score derived from Whisper's average token
log-probability. It is **not** a calibrated measure of pronunciation — it is a monotonic proxy
for how confidently the model decoded the audio. Read it comparatively ("that came through more
clearly than last time"), not absolutely. Mic distance and background noise move it too.

### Benchmarking a recording

Replay a WAV file through the exact live pipeline — useful for comparing microphones or
placements without re-recording:

```powershell
.\.venv\Scripts\python.exe -m server.app --wav testdata\room.wav        # real-time
.\.venv\Scripts\python.exe -m server.app --wav testdata\room.wav --fast # as fast as possible
```

## Why these choices

**Whisper large-v3, not `large-v3-turbo` or a faster streaming model.**
Turbo prunes the decoder from 32 layers to 4. Published deltas show that costs +4.5% relative
WER on clean speech but **+8.4% on harder, more varied audio** — the degradation is 2–4x worse
exactly where accents live. The English-only streaming models (Parakeet, Canary, Zipformer,
Moonshine) are faster still, but are trained overwhelmingly on US English and publish **no
multi-accent benchmarks**. Whisper large-v3 scores 7.2% WER on SVARAH (Indian-accented English)
versus 20.7% for Google Cloud's `en-IN` model, and its 680k-hour multilingual training covers
European-accented English well.

**One model, two decoding passes.** Provisional text uses greedy decoding (~250 ms) so words
appear almost immediately; the final pass re-decodes the completed utterance with beam search
for accuracy. Only one model sits in VRAM. Both passes are Whisper, so accent robustness holds
end to end — a fast non-Whisper model for provisional text would break down on the accents it
was never trained on.

**Caption display follows BBC/DCMP live-subtitle convention.** Provisional text is dimmed and
italic; it is replaced exactly once, at the utterance boundary, by full-contrast final text.
Corrections are batched at sentence boundaries rather than churning word by word, which is what
accessibility guidance recommends for deaf and hard-of-hearing readers. Target latency is under
the BBC's 1.5 s tolerance for live subtitles.

**No software noise suppression.** Do not route audio through NVIDIA Broadcast or similar
neural denoisers before this pipeline. Noise suppression measurably *lowers* transcription
accuracy and disproportionately clips accented speakers and short utterances. Hardware
beamforming inside a microphone array is fine — it's post-capture neural denoising that hurts.
What the pipeline *does* apply is deliberately conservative: DC removal, an 80 Hz high-pass to
strip rumble below the speech band, and gentle level normalisation for quiet far-field audio.

**`float16`, not `int8`.** Turing has INT8 tensor cores, so `int8_float16` looked promising.
Measured on this machine it is ~15% *slower* — the bottleneck isn't the matmuls, and
quantisation just adds overhead:

| compute type | beam=1 | beam=5 |
|---|---|---|
| **float16** | **322 ms** | **397 ms** |
| int8_float16 | 374 ms | 431 ms |
| int8 | 370 ms | 429 ms |

Beam search also showed no text improvement on clean audio (beam=8 was actively worse), but
beam=5 is kept for finals as headroom on hard far-field accented speech, where it does help.

**Context, not `condition_on_previous_text`.** Recent finalised text is fed back as
`initial_prompt`, which improves proper nouns and continuity. It's capped and expires after 30 s
of silence, because unbounded feedback of the model's own output seeds hallucination loops —
the same reason `condition_on_previous_text` stays off.

## Microphone matters more than the model

This is the single biggest factor in transcription quality. Whisper large-v3 on the AMI meeting
corpus:

| Microphone condition | WER |
|---|---|
| Close-talking / headset | **15.95%** |
| Single distant mic at 1–2 m | ~33–42% |
| Mic array with beamforming | ~22–30% |

Moving from a distant tabletop mic to a close-talking mic is worth roughly **2x the accuracy** —
more than any model change available. If several people are sharing one microphone, a
beamforming array (Seeed ReSpeaker XVF3800, Anker PowerConf S3) substantially outperforms a
plain omnidirectional mic, which captures every room reflection equally.

## Layout

```
server/          Python backend
  app.py         entry point: WebSocket + static UI server, model gating, CLI
  pipeline.py    VAD state machine, session control, two-pass ASR worker
  asr.py         faster-whisper wrapper (provisional/final passes, context, clarity)
  speaker.py     online speaker labelling with naming and persistence
  models.py      model catalog + first-run download with progress
  preprocess.py  conservative audio conditioning (high-pass biquad, level, DC)
  vad.py         streaming Silero VAD with state carried across frames
  audio.py       microphone capture, format negotiation, resampling; WAV replay
  paths.py       read-only install vs writable LocalAppData split
  config.py      tunable settings
  cuda_setup.py  registers the bundled NVIDIA DLLs; fails loudly if mis-staged
app/             WinUI 3 (C#) frontend
  MainWindow.*   captions, speakers pane, command bar, first-run setup
  Services/      BackendHost, CaptionClient, ChildProcessJob, AppSettings
  Assets/        generated icon set (see packaging/make_icon.py)
ui/              browser client, served over HTTP for the phone/handheld route
packaging/
  make_icon.py       generates the .ico and the full MSIX asset set
  cuda_allowlist.txt CUDA DLLs proven reachable by import analysis
  stage-backend.ps1  stages a self-contained Python runtime + backend
  build-msix.ps1     publishes, stages, packs and signs the MSIX
  Package.appxmanifest
```

## Packaging

```powershell
cd packaging
.\build-msix.ps1
```

Produces a signed `out\LiveCaptions.msix` (~794 MB). The script deliberately stops before
installing: installing a self-signed package requires adding its certificate to
LocalMachine Trusted People, which is a machine-wide trust change, so it prints the two
elevated commands rather than running them.

**What is and isn't in the package.** The app, a self-contained Python 3.12 runtime, the
backend, the browser UI and the 28 MB speaker-embedding model ship inside. The 2.9 GB
Whisper model does not — it is downloaded on first run into the user's cache, which keeps
the package small and the install directory read-only-safe.

**CUDA payload.** The pip NVIDIA wheels total 1,984 MB; only 828 MB ships. Parsing the PE
import tables (static, delay-load and string-literal `LoadLibrary` targets) of ctranslate2,
onnxruntime and sherpa-onnx shows the entire dependency chain is
`ctranslate2 → cublas64_12 → cublasLt64_12 → nvrtc`. Nothing reaches cuDNN; CTranslate2's
own bundled `cudnn64_9.dll` is vestigial. Confirmed empirically too — transcripts are
byte-identical with the cuDNN tree removed. Regenerate the allow-list with
`packaging/cuda_decide.py` if dependencies change.

## Tuning

Endpointing lives in `server/config.py`:

- `end_silence_ms` (520) — silence before an utterance is committed. **This dominates
  perceived latency far more than GPU speed does**: final text lands at roughly
  `end_silence_ms` + ~350 ms of inference. Lower feels snappier and reduces mixed-speaker
  segments, but splits sentences at natural pauses.
- `partial_interval_ms` (450) — how often provisional text refreshes.
- `vad_start_threshold` / `vad_end_threshold` — raise in a noisy room to avoid false triggers.
- `speaker_threshold` (0.50) — lower merges speakers, higher splits them. Measured stable
  between 0.40 and 0.55.
- `vocabulary` — names and places to bias transcription. Also `--vocabulary "Priya,Hyderabad"`.
- `max_utterance_s` (20) — long monologues are force-committed to bound latency.

## Known limitations

- **Speaker labels are best-effort.** ~2/3 of turns correct on two-speaker audio; short turns
  are left unlabelled rather than guessed. Naming people improves this markedly. See above.
- **Overlapping speech.** When two people talk at once, single-channel ASR degrades badly
  (50–80% WER is typical for any model). Separation models fast enough for real time aren't
  practical yet on this hardware.
- Utterances longer than `max_utterance_s` are split mid-sentence.
- Provisional lines carry no speaker chip — identification happens when the utterance ends —
  so the chip appears at commit time.

## Requirements

NVIDIA GPU with CUDA support. Developed against a Quadro RTX 8000 (Turing, sm_75, 48 GB).
Note that Turing supports FP16 and INT8 but **not** bfloat16 or FlashAttention-2, so
`compute_type` must stay `float16` or `int8_float16`. Whisper large-v3 needs ~3 GB of VRAM.
