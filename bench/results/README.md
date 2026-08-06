# Measured decode latency

Raw output from `bench/bench_arm.py`. These files are the evidence behind the ARM model tiers;
`server/hardware.py`'s lag tables are source constants, so numbers move from here into that file
by hand.

## Machines

| file | machine | notes |
|---|---|---|
| `x64-i9-14900k.json` | i9-14900K, x64 | desktop baseline, ONNX via genai |
| `x64-i9-14900k-ct2.json` | same machine | CTranslate2, all six models, 2/4/8 s. **The 4-thread pass is contended** — `distil-large-v3` reads 8.36 s against 3.79/3.71, and `base` comes out faster at 4 threads than at 16, which cannot be right. The 16-thread and CUDA passes are sound. This is the provenance for `base` in `server/hardware.py`. |
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

The smallest model is the slower one, and the 2 s row is why: tiny hallucinates on short audio and then spends real time decoding words nobody said. One run returned *"Attention!"* over speech containing no such word. Since short utterances are the whole workload here, tiny is not offered — see the note in `server/models.py`.

Reproduced across three separate runs. All three used the same clip, so treat the *magnitude* as one data point; the direction is consistent.

## Mean decode, ms (budget: 1000 ms)

| model | i9-14900K | Snapdragon balanced | Snapdragon best perf |
|---|---|---|---|
| tiny | 173 | 203 | **132** |
| base | 298 | 519 | **297** |
| small | 846 | 1580 | **987** |
| medium | — | 5032 | 3331 |

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
