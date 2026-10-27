# Sunno engineering notes

This was the original README, kept because the reasoning and the measurements behind each
decision are worth more than the summary of them. It explains *why* the pipeline looks the way it
does: why Whisper large-v3 rather than something faster, why two decoding passes, why no noise
suppression, why float16, and what the numbers were.

For what Sunno is and how to install it, start at the [README](../README.md).

Two things to know while reading:

* The sections on `run.ps1`, `run.cmd` and the browser UI describe the original interface, which
  still works and is still how the phone or handheld route is served. The shipping product is now
  a WinUI 3 desktop app that launches the same backend for you.
* "Requires an NVIDIA GPU" is no longer true. The app falls back to CPU, which is slower but
  works. The accuracy and latency measurements below were all taken on a GPU.

---

Offline, real-time speech-to-text captioning. Recognition makes no network calls and needs
no API keys or per-minute payments. Model downloads and user-requested app update checks
use the network; audio never leaves the machine.

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
machine learning, so moving the display to a separate handheld screen later needs no rewrite, 
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

> **Which shell?** In **PowerShell 7** (`pwsh`), `.\run.ps1` works directly: its default
> execution policy is `RemoteSigned`. In **Windows PowerShell 5.1** (the blue icon) the default
> is `Restricted`, which refuses `.ps1` files; use `.\run.cmd` there instead. `run.cmd` works in
> both shells and when double-clicked from File Explorer, so it's the safe default. Note that
> `Get-ExecutionPolicy` can mislead: the two shells read *different* registry keys, so one can
> allow scripts while the other blocks them.

## Setup

Requires Python 3.12 (python.org build, **not** the Microsoft Store build, which is
sandboxed and unreliable at loading native CUDA DLLs) and an NVIDIA GPU.

```powershell
winget install --id Python.Python.3.12 --scope user
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For development, running the tests or rebuilding the icon and CUDA allow-list, also
install `requirements-dev.txt`. Those packages (scipy, pefile, Pillow) are deliberately kept
out of the shipped payload; `stage-backend.ps1` fails the build if they appear in it.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe tests\test_biquad.py     # standalone script, not pytest
```

`test_biquad.py` guards the scipy removal: it compares the hand-rolled high-pass against
`scipy.signal.sosfilt` across impulse, noise, DC, sines, clipping and a synthetic
speech-plus-rumble signal, and checks that filter state does not leak between utterances. It
reports how many signals it ran; the count is higher when `testdata/verify.wav` is present.

## Input switching and recovery

On Windows, capture runs in a disposable child process using native WASAPI shared mode.
Microphones and system audio are identified by persistent endpoint IDs, not enumeration
indices. A legacy saved name is resolved before its index; an ambiguous name requires a new
selection. Each group offers a separate option to follow the Windows default.

The backend keeps one speech engine, ASR worker, speaker roster and recorder alive across input
changes. A healthy input remains active until its replacement produces its first frame. A
failed manual switch keeps the previous healthy input and leaves the saved selection intact.
Rapid changes cancel superseded candidates, and reconnects replay the same request without
opening another stream. Paused selections only inspect metadata, never open capture.

An opening worker has a six-second deadline and an active worker a three-second heartbeat.
Transient failures retry after 0.25, 0.5, 1, 2 and then 5 seconds, with hardware notifications
bypassing the delay. Access denial and unsupported formats wait for user action. Idle system
audio produces healthy silence while checking for endpoint invalidation. Pinned inputs wait
for that input to return instead of silently capturing a different microphone. Default mode
checks Windows every two seconds and reacts sooner to native endpoint notifications.

Workers use the endpoint's native mix format, then downmix and resample to 16 kHz. This avoids
repeated format probes. Shared-mode buffer capacity is not the capture delay; samples are
consumed as they arrive. See Microsoft's [shared-mode initialization documentation](https://learn.microsoft.com/en-us/windows/win32/api/audioclient/nf-audioclient-iaudioclient-initialize)
and [loopback recording documentation](https://learn.microsoft.com/en-us/windows/win32/coreaudio/loopback-recording).
Worker audio stays in bounded memory queues. Device names and IDs appear only in local settings
and UI frames, never in capture diagnostics.

Verification:

```powershell
python tests/test_capture_recovery.py
dotnet run --project tests/windows/InputRecoveryTests.csproj
node tests/test_browser_recovery.js
# Optional Windows smoke test: enumeration plus loopback audio in memory, no microphone capture.
$env:SUNNO_TEST_NATIVE_AUDIO = "1"
python tests/test_capture_recovery.py
```

The automated checks cover capture hangs, malformed IPC, retry limits, rollback, rapid changes,
pause safety, default following, Unicode WebSocket frames, reconnect replay and recording
continuity. Physical USB and Bluetooth disconnects, sample-rate changes, permission revocation
and sleep/wake still need testing with the affected hardware.

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
| `A+` / `A−` (or Ctrl `+` / Ctrl `−`) | Change text size, remembered between runs |
| `Clear` | Clear the transcript |

In the Windows app, upward reader navigation pauses automatic scrolling. Scrolling down again
within 80 DIPs of the bottom resumes it. **Skip to current** appears above the bottom edge
while reading history; selecting it immediately jumps to the latest captions and resumes
following. Clearing the transcript also resets live following.

`CaptionScrollFollower` stores following as reader intent. `CaptionScrollViewer` wraps the
native WinUI ScrollViewer and reports scrollbar and accessibility navigation; the follower
also handles wheel, keyboard and direct manipulation. Offset changes alone never pause
following: final word rendering rebuilds text blocks, and trimming the 200-line history can
move WinUI's anchor even when the content height is unchanged. Treating either adjustment
as a gesture caused following to stop unexpectedly after finalization or a speech pause.
Follow requests are coalesced after layout, recheck reader intent, and settle asynchronous
anchor adjustments without animated scrolling. While paused, native scroll anchors preserve
the visible caption through reflow and history trimming. Real WinUI regression checks in
`tests/windows-ui` cover these transitions and the floating button.

**Stop genuinely releases the microphone**: it closes the audio device rather than quietly
discarding audio, so Windows' microphone-in-use indicator switches off and people in the room
can see that capture has stopped. The Whisper model stays loaded on the GPU, so resuming takes
well under a second. Audio captured mid-sentence when you press Stop is discarded, not
transcribed. Launch with `--start-stopped` to begin paused.

### Speaker labels

Settings > Captions > Speaker labels offers Off and Automatic (Experimental). Off skips new
speaker inference and hides speaker metadata without restarting capture or the speech model.
An in-flight native embedding can finish, but its result is discarded.
Automatic assigns provisional labels alongside the first words and corrects them in place.
Click a speaker to:

- **Name the speaker**: optionally persist an identified voice across sessions. This is not
  required for unfamiliar presenters in videos. Naming pins the current reference; it does
  not guarantee that a poor initial voice sample becomes more accurate.
- **Mark them as you**: your own lines then render fainter, labelled *You*, and on a Whisper
  model they also carry a **clarity score**. Your speech is still transcribed; the score is
  there so you can read back how clearly you came across. The streaming models do not produce
  one, which is why the setting and both speaker dialogs name Whisper explicitly.
- **Merge two speakers**: use this when one person gets split across two labels.

The speaker worker is separate from ASR. It uses VAD-selected speech, not silence padding,
and receives at most eight seconds per snapshot with an eight-job backlog limit. Partial
snapshots are sampled at roughly 1.3-second voiced intervals; completed turns can update a
small in-memory reference bank. Short or ambiguous samples keep a provisional assignment.
There is no user confirmation or wait-for-confirmation stage before showing a label.

Recent completed assignments can be revised after a stronger reference arrives. Strongly
matching, established anonymous profiles can be consolidated; named profiles are not
automatically merged. Corrections carry an utterance ID and speech timestamp, and caption
events are annotated with the latest assignment when emitted, so late ASR jobs cannot undo
a correction. A session token prevents ID reuse after an engine restart from renaming old
captions. Recording journals retain amendments for export and crash recovery.

**Accuracy expectations.** This is experimental, not a verified diarization accuracy claim.
The regression tests cover state, timing, persistence and corrections using controlled
embeddings. Similar voices, very short turns, overlap and rapid changes without a pause can
still be wrong. Each caption utterance currently receives one label, even if it contains
multiple speakers. A representative real-audio benchmark is still needed.

### Clarity score

Lines marked as yours carry a 0-100 clarity score derived from Whisper's average token
log-probability, **on a Whisper model only**. It comes from `asr.py`; the streaming engines
in `asr_stream.py` and `asr_onnx.py` both return `clarity=None`, because a transducer exposes
no comparable score and inventing one would be worse than leaving it out. That is why the
Settings toggle, both speaker dialogs and the section above all name Whisper explicitly, and
why `tests/test_stream_engine.py` pins those strings against the engines that justify them.

It is **not** a calibrated measure of pronunciation: it is a monotonic proxy
for how confidently the model decoded the audio. Read it comparatively ("that came through more
clearly than last time"), not absolutely. Mic distance and background noise move it too.

### Benchmarking a recording

Replay a WAV file through the exact live pipeline, useful for comparing microphones or
placements without re-recording:

```powershell
.\.venv\Scripts\python.exe -m server.app --wav testdata\room.wav        # real-time
.\.venv\Scripts\python.exe -m server.app --wav testdata\room.wav --fast # as fast as possible
```

## Why these choices

**Whisper large-v3, not `large-v3-turbo` or a faster streaming model.**
Turbo prunes the decoder from 32 layers to 4. Published deltas show that costs +4.5% relative
WER on clean speech but **+8.4% on harder, more varied audio**: the degradation is 2–4x worse
exactly where accents live. The English-only streaming models (Parakeet, Canary, Zipformer,
Moonshine) are faster still, but are trained overwhelmingly on US English and publish **no
multi-accent benchmarks**. Whisper large-v3 scores 7.2% WER on SVARAH (Indian-accented English)
versus 20.7% for Google Cloud's `en-IN` model, and its 680k-hour multilingual training covers
European-accented English well.

**One model, two decoding passes.** Provisional text uses greedy decoding (~250 ms) so words
appear almost immediately; the final pass re-decodes the completed utterance with beam search
for accuracy. Only one model sits in VRAM. Both passes are Whisper, so accent robustness holds
end to end, a fast non-Whisper model for provisional text would break down on the accents it
was never trained on.

**Caption display follows BBC/DCMP live-subtitle convention.** Provisional text is dimmed and
italic; it is replaced exactly once, at the utterance boundary, by full-contrast final text.
Corrections are batched at sentence boundaries rather than churning word by word, which is what
accessibility guidance recommends for deaf and hard-of-hearing readers. Target latency is under
the BBC's 1.5 s tolerance for live subtitles.

**No software noise suppression.** Do not route audio through NVIDIA Broadcast or similar
neural denoisers before this pipeline. Noise suppression measurably *lowers* transcription
accuracy and disproportionately clips accented speakers and short utterances. Hardware
beamforming inside a microphone array is fine, it's post-capture neural denoising that hurts.
What the pipeline *does* apply is deliberately conservative: DC removal, an 80 Hz high-pass to
strip rumble below the speech band, and gentle level normalisation for quiet far-field audio.

**`float16`, not `int8`.** Turing has INT8 tensor cores, so `int8_float16` looked promising.
Measured on this machine it is ~15% *slower*, the bottleneck isn't the matmuls, and
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
of silence, because unbounded feedback of the model's own output seeds hallucination loops, 
the same reason `condition_on_previous_text` stays off.

## Microphone matters more than the model

This is the single biggest factor in transcription quality. Whisper large-v3 on the AMI meeting
corpus:

| Microphone condition | WER |
|---|---|
| Close-talking / headset | **15.95%** |
| Single distant mic at 1–2 m | ~33–42% |
| Mic array with beamforming | ~22–30% |

Moving from a distant tabletop mic to a close-talking mic is worth roughly **2x the accuracy**: 
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

Produces a signed `out\Sunno.msix` (~794 MB). The script deliberately stops before
installing: installing a self-signed package requires adding its certificate to
LocalMachine Trusted People, which is a machine-wide trust change, so it prints the two
elevated commands rather than running them.

**What is and isn't in the package.** The app, a self-contained Python 3.12 runtime, the
backend, the browser UI and the 28 MB speaker-embedding model ship inside. The 2.9 GB
Whisper model does not, it is downloaded on first run into the user's cache, which keeps
the package small and the install directory read-only-safe.

**CUDA payload.** The pip NVIDIA wheels total 1,984 MB; only 828 MB ships. Parsing the PE
import tables (static, delay-load and string-literal `LoadLibrary` targets) of ctranslate2,
onnxruntime and sherpa-onnx shows the entire dependency chain is
`ctranslate2 → cublas64_12 → cublasLt64_12 → nvrtc`. Nothing reaches cuDNN; CTranslate2's
own bundled `cudnn64_9.dll` is vestigial. Confirmed empirically too, transcripts are
byte-identical with the cuDNN tree removed. Regenerate the allow-list with
`packaging/cuda_decide.py` if dependencies change.

### Publishing Windows app updates

Settings checks `desigrit/sunno` releases only when the user requests it. The updater looks
through the latest 100 releases, excludes drafts and prereleases, and compares numeric
package versions. CUDA releases and releases without compatible installer assets are ignored.
It does not check local folders or GitHub Actions artifacts.

The packaging script emits both `Sunno.msix` and `Sunno_<version>_<architecture>.msix`.
For example, a stable release tagged `v1.0.90.0` must attach `Sunno_1.0.90.0_x64.msix`.
The tag, filename and package manifest version must agree. Publish the release against the
commit used to build it. Upload only the signed MSIX, never the development PFX or its password.
GitHub supplies the asset's SHA-256 digest, which Sunno requires and verifies after download
and again before installation. The package identity, publisher and architecture must match
the app; only a strictly newer version can be offered. On ARM Windows the current shell is
x64, so it still selects the x64 installer.

Windows App Installer owns installation, signature trust and any app-in-use prompts. Sunno
does not import certificates or claim success just because the installer opened. A recording
must be stopped before handoff. The installed version shown on the next launch is the result.
Self-signed development builds require the publisher certificate to be trusted on that PC;
the updater does not bypass that requirement. The existing model and settings locations are
unchanged. A newer source commit or unpublished build is not a downloadable release.

Run `dotnet run --project tests/updates/UpdateTests.csproj -c Release` for release selection,
download verification, cancellation, redirect and package identity checks. Pass `-- --live`
to also read the public GitHub release feed. The native UI harness supports
`SUNNO_UPDATE_TEST_ONLY=1` for the production update panel's light/dark and action states.

## Tuning

Endpointing lives in `server/config.py`:

- `end_silence_ms` (520), silence before an utterance is committed. **This dominates
  perceived latency far more than GPU speed does**: final text lands at roughly
  `end_silence_ms` + ~350 ms of inference. Lower feels snappier and reduces mixed-speaker
  segments, but splits sentences at natural pauses.
- `partial_interval_ms` (450), how often provisional text refreshes.
- `vad_start_threshold` / `vad_end_threshold`, raise in a noisy room to avoid false triggers.
- `speaker_threshold` (0.50), lower merges speakers, higher splits them. Measured stable
  between 0.40 and 0.55.
- `vocabulary`, names and places to bias transcription. Also `--vocabulary "Priya,Hyderabad"`.
- `max_utterance_s` (20), long monologues are force-committed to bound latency.

## Known limitations

- **Speaker labels are experimental.** Immediate assignments can be corrected later. Off
  disables inference; saved voice profiles are optional. See above for known limitations.
- **Overlapping speech.** When two people talk at once, single-channel ASR degrades badly
  (50–80% WER is typical for any model). Separation models fast enough for real time aren't
  practical yet on this hardware.
- Utterances longer than `max_utterance_s` are split mid-sentence.
- Provisional lines carry a provisional speaker chip in Automatic mode. The label can change
  independently of the words as more evidence arrives.

## Requirements

**A 64-bit Intel or AMD processor.** The engine is CTranslate2, which publishes no `win_arm64`
wheel, so an ARM PC runs this build under Prism emulation, slowly at best, and the backend may
not load at all. `server/hardware.py` reports the native machine via `IsWow64Process2` so that
case is identifiable in a bug report rather than surfacing as an unexplained exit.

A GPU is **optional**. An NVIDIA card with CUDA support is much faster: the measured decode lag
for large-v3 is 650 ms on CUDA against 4,400–4,540 ms on CPU depending on thread count
(`server/hardware.py:31-55`; Quadro RTX 8000 float16 / i9-14900K int8, greedy decode, best of
three), but the CPU path works and is the default fallback when no usable GPU is found.

Developed against a Quadro RTX 8000 (Turing, sm_75, 48 GB). Note that Turing supports FP16 and
INT8 but **not** bfloat16 or FlashAttention-2, so `compute_type` must stay `float16` or
`int8_float16`. Whisper large-v3 needs ~3 GB of VRAM.
