# Measured decode latency

Raw output from `bench/bench_arm.py`. These files are the evidence behind the ARM model tiers;
`server/hardware.py`'s lag tables are source constants, so numbers move from here into that file
by hand.

## Machines

| file | machine | notes |
|---|---|---|
| `x64-i9-14900k.json` | i9-14900K, x64 | desktop baseline, ONNX via genai |
| `x64-i9-14900k-ct2.json` | same machine | CTranslate2, all six models, 2/4/8 s. **The 4-thread pass is contended**: `distil-large-v3` reads 8.36 s against 3.79/3.71, and `base` comes out faster at 4 threads than at 16, which cannot be right. The 16-thread and CUDA passes are sound. This is the provenance for `base` in `server/hardware.py`. |
| `snapdragon-balanced.json` | Snapdragon X, ARM64 native | Windows power mode **Balanced** |
| `snapdragon-best-performance.json` | same machine | Windows power mode **Best Performance** |
| `snapdragon-best-performance-qnn.json` | same machine | `--qnn` asked for, **provider never registered** - see below |
| `snapdragon-best-performance-qnn-registered.json` | same machine | `--qnn` with the provider actually registered |
| `snapdragon-qnn-small-w8a16.json` | same machine | Qualcomm's precompiled Whisper on the Hexagon NPU |

## Whisper tiny is slower than base on short speech

From `x64-i9-14900k-ct2.json`, 16 threads:

| clip | tiny | base |
|---|---|---|
| 2 s | **870 ms** | 420 ms |
| 4 s | 230 ms | 410 ms |
| 8 s | 290 ms | 490 ms |
| mean | **463 ms** | 440 ms |

The smallest model is the slower one, and the 2 s row is why: tiny hallucinates on short audio and then spends real time decoding words nobody said. One run returned *"Attention!"* over speech containing no such word. Since short utterances are the whole workload here, tiny is not offered. See the note in `server/models.py`.

Reproduced across three separate runs. All three used the same clip, so treat the *magnitude* as one data point; the direction is consistent.

## Mean decode, ms (budget: 1000 ms)

| model | i9-14900K | Snapdragon balanced | Snapdragon best perf |
|---|---|---|---|
| tiny | 173 | 203 | **132** |
| base | 298 | 519 | **297** |
| small | 846 | 1580 | **987** |
| medium | n/a | 5032 | 3331 |

## The NPU does not take the work

Measured, not assumed. With `QNNExecutionProvider` registered and the models loading through it:

| model | CPU | QNN | decode | load |
|---|---|---|---|---|
| tiny | 132.2 | 146.3 | **+10.7%** | 14.3x |
| base | 302.1 | 317.4 | **+5.1%** | 9.1x |
| small | 1037.0 | 1097.8 | **+5.9%** | 3.7x |
| medium | 3582.5 | 3741.5 | **+4.4%** | 1.9x |

Three signatures of a complete fall back to CPU: decode is consistently *slower* rather than
faster, load time rises by up to 14x, and the transcripts are byte-identical. QNN partitions the
graph, finds it cannot place the operators on the Hexagon backend, compiles that discovery
expensively, and hands everything back to the CPU - which then pays a few percent for the
partitioning layer it did not need.

This is the predicted outcome rather than a surprise: HTP wants static shapes and quantised
graphs, and these are float models whose decoder carries a KV cache that grows one step at a
time.

**What it does not prove:** that the NPU is useless for this. A QDQ-quantised, static-shape
Whisper built specifically for HTP is a different experiment - and a much larger one, requiring
quantisation, calibration data, and a fixed decode window. What it does establish is that no
off-the-shelf ONNX Whisper benefits, so the ARM tiers should be chosen on CPU numbers.

**The earlier `-qnn` file is not evidence.** It failed with *"QNN execution provider is not
supported in this build"*, which was true of the build and silent about the machine: QNN was
never installed (it ships in `onnxruntime-qnn`, separate from `onnxruntime-genai`) and the
provider was being requested under the wrong name. Kept because a run that looks like a
measurement and is not is worth being able to point at.

## What these show

**Power mode is worth ~1.6x.** Balanced to Best Performance takes small from 1580 ms to 987 ms -
across the responsiveness line rather than merely nearer it. A laptop on battery in Balanced is
the honest default to design for, so `small` cannot be treated as reliably live-capable even
though one configuration clears the budget. The re-run measured it at 1037 ms, on the other side
of the line again, which settles the point: `small` sits on the boundary and lands either way
depending on the day.

**On Best Performance the Snapdragon matches an i9-14900K** at tiny and base (132 vs 173, 297 vs
298). That is a much stronger result than the emulated-x64 path this replaces.

**No accuracy signal.** Every model returned near-identical text because the clip is synthesised
speech, which is what makes it a fair latency test and a useless accuracy one. Whether a smaller
model handles accented speech well enough is a separate measurement.

## 2026-10-05 engine preview: first observational pass

[Raw timing and resource measurements](2026-10-05-engine-preview.json) are from
`bench/bench_engines.py`, using Sunno's production engine calls. Windows ran on an
i9-14900K with 64 GB RAM and a Quadro RTX 8000. The Mac ran the preview engine code
on an M1 Max with 32 GB RAM and macOS 26.6.2. No microphone was opened.

These are observational measurements on an in-use desktop, not an idle laboratory
calibration. Background applications were not paused. Windows Moonshine Small was
particularly variable: the first real-time run reported compute RTF near 0.7 but
failed to save its detailed results because Python's optional WMI platform query
timed out. The saved repeat exceeded 1.7. The harness now avoids that metadata query.
Do not use this pass to assign a stable device-independent delay or change defaults.

### Live-style streaming replay

Audio was delivered in real time. Refresh scheduling followed Sunno's adaptive
cadence and coalesced audio after expensive calls. Each entry is a range across
the medians of three warmed runs on identical 2, 4 and 8 second clips.

| Model | Windows final available after clip end | M1 Max final available after clip end |
|---|---|---|
| Zipformer | 30-50 ms | Not measured |
| Moonshine Tiny | 47-373 ms | 51-86 ms |
| Moonshine Small | 1701-8581 ms, highly variable | 51-124 ms |
| Moonshine Medium | 840-2333 ms | 50-156 ms |

This is **not end-to-end caption delay**. It excludes capture, VAD endpointing
(currently about 520 ms), speaker identification, UI rendering and hardware input
buffers. First-word latency also includes the model's lookahead and caption cadence.
The raw file retains per-refresh peaks and processing RTF, not just final flush cost.
Low final flush cost alone can conceal expensive earlier refreshes or backlog.

### Whisper's actual final decoding

Sunno's final pass uses beam 5, word timestamps and temperature fallback. Its existing
picker reference table measures greedy decoding, so these figures are not a direct
replacement for that table. These ranges again cover median 2, 4 and 8 second decodes.

| Model | Windows CPU final decode | RTX 8000 final decode |
|---|---|---|
| Whisper base | 609-679 ms | Not measured |
| Whisper small | 1694-1987 ms | Not measured |
| Distil-Whisper large-v3 | 6828-6943 ms | 606-931 ms |
| Whisper large-v3 | 8443-9318 ms | 1131-2458 ms |

The requested thread count was 4. This controls Whisper and Zipformer, but is not
proof that every native Moonshine session uses exactly four threads. GPU inference
was float16; Whisper CPU inference was int8.

### Accuracy and memory sanity checks

The public fixture is the first eight ID-sorted clips from
[Hugging Face's LibriSpeech test subset](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy),
pinned by SHA256 in `bench/prepare_public_corpus.py`. It contains 86.3 seconds,
200 reference words and **one speaker reading clean audio**. Case and most punctuation
are ignored; internal apostrophes are retained and abbreviations are not expanded. This is not a representative accuracy
or accented-English evaluation; a one-word difference is 0.5 percentage points.

| Model | Windows word error rate | M1 Max word error rate | Windows peak process RAM |
|---|---|---|---|
| Zipformer | 5.0% | Not measured | 225 MiB |
| Kroko | 9.5% | Not measured | 248 MiB |
| Moonshine Tiny | 12.5% | 11.5% | 255 MiB |
| Moonshine Small | 6.5% | 6.5% | 431 MiB |
| Moonshine Medium | 5.5% | 5.0% | 595 MiB |
| Whisper base | 10.5% | Not measured | 356 MiB |
| Whisper small | 9.5% | Not measured | 745 MiB |
| Distil-Whisper large-v3 | 7.5% | Not measured | 1828 MiB |
| Whisper large-v3 | 5.0% | Not measured | 3332 MiB |

GPU Whisper produced the same error rates. RAM measures the standalone benchmark
process, excluding Sunno's UI, VAD, speaker model, GPU memory and helper processes.
M1 Max peak RAM during the accuracy pass was 784/1606/2246 MiB for Moonshine
Tiny/Small/Medium. Neither a 16 GB machine nor the older Intel and Snapdragon laptops
was tested in this pass.

### Native engine limits

Windows reported the model as supported but not installed. Provisioning failed with
`0x8000FFFF`, including inside Sunno's MSIX identity and with the complete experimental
SDK instead of the smaller AI package. CPU support is documented by
[Microsoft](https://learn.microsoft.com/en-us/windows/ai/apis/speech-recognition).
The setup failure is unresolved; it is not evidence that Intel CPUs are categorically
unsupported. Sunno now preserves readiness status and actual setup errors.

Apple SpeechAnalyzer reported support, but its model assets were not installed.
Neither native engine has a successful performance or accuracy measurement here.

### Practical interpretation

Zipformer had the most consistent low CPU cost on this Windows desktop. Moonshine
Tiny was lighter than the larger Moonshine variants, but made more errors on the tiny
sanity sample. Windows Moonshine refresh spikes need investigation before trusting a
latency estimate. Moonshine Small and Medium were promising on M1 Max. Large Whisper
remains substantially more practical with the measured GPU than with CPU alone.

Repeat on the actual older Intel and Snapdragon laptops, with representative accented
speech and a controlled background workload, before changing recommendations or defaults.

## 2026-10-05: Moonshine caption-processing delay

This follow-up measures caption emission through the actual speech pipeline,
rather than isolated engine calls. Audio was replayed in real time through
Sunno's VAD, streaming audio conditioner, adaptive refresh scheduling and serial
ASR worker. Production speech code was not changed.

Three clean English clips (4.82, 5.86 and 12.49 seconds) were each replayed three
times per model, after one discarded warmup. All 54 measured utterances returned
nonempty partial and final captions, with no ASR errors and no EOF-forced
endpoints. Two seconds of silence were appended so the normal endpoint detector
could finish each utterance.

Times below are seconds. First caption is measured from the VAD-estimated speech
onset. Final caption is measured from the VAD-estimated end of speech, not from
the end of the WAV or from model startup.

| Machine | Model | First caption, median | Final after speech ends, median | Final, observed range |
|---|---|---|---|---|
| Windows i9 | Moonshine Tiny | 0.64 | 1.64 | 0.80-1.91 |
| Windows i9 | Moonshine Small | 1.10 | 3.94 | 2.33-4.96 |
| Windows i9 | Moonshine Medium | 1.53 | 6.07 | 5.07-9.94 |
| M1 Max | Moonshine Tiny | 0.51 | 0.55 | 0.53-0.66 |
| M1 Max | Moonshine Small | 0.54 | 0.57 | 0.55-0.77 |
| M1 Max | Moonshine Medium | 0.56 | 0.62 | 0.55-0.80 |

The Windows host was the i9-14900K desktop with 64 GB RAM. The Mac was the M1 Max
with 32 GB RAM. All Moonshine inference used the adapter's CPU provider, not the
desktop GPU or the Mac GPU. Windows used the development checkout; the Mac used
the preview source and its bundled Python runtime. The VAD, conditioner, caption
pipeline and Moonshine adapter source hashes matched across hosts. The WAV hashes
and detected speech-boundary frames matched across all six measurements.

### What the numbers mean

The configured 520 ms silence wait is rounded to 16 frames, or 512 ms, by the
current pipeline. This is most of the Mac's final-caption delay. Windows Small
and Medium instead show substantial decode cost and queueing. Medium's median
final decode cost was 3.48 seconds and its median final queue wait was 1.92
seconds; the longest observed queue wait was 7.41 seconds. Component medians
must not be added to reconstruct the median total.

These are observational runs on in-use machines, not an idle calibration or an
isolated OS comparison. Background applications were not paused, and runtime
versions differ. Input-feed lateness is preserved in the JSON; the worst frame
was 762 ms late on Windows and 26 ms late on the Mac. The cause of the Windows
slowdowns has not been determined by this measurement.

This is backend caption-processing delay, not full microphone-to-screen latency.
It excludes physical input buffers, speaker labeling, WebSocket transport,
screen painting and cold model loading. Speech boundaries are estimates from
the production VAD with 32 ms frame resolution, not human phonetic annotations.
One clean reader is not representative of accented speech, noisy conversations,
long overlapping turns, or the older Intel and Snapdragon laptops.

[Sanitized timings, repeat-level results, runtime versions and source hashes](2026-10-05-moonshine-caption-delay.json)
retain the measurement evidence without transcript text or private profile paths.
The reusable harness is `bench/bench_caption_delay.py`; it reads existing model
weights and does not download models or open a microphone. Four focused tests
cover timestamp arithmetic, final-only/empty results, utterance matching and
observation of the real segmentation transitions.
