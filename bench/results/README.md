# Measured decode latency

Raw output from `bench/bench_arm.py`. These files are the evidence behind the ARM model tiers;
`server/hardware.py`'s lag tables are source constants, so numbers move from here into that file
by hand.

## Machines

| file | machine | notes |
|---|---|---|
| `x64-i9-14900k.json` | i9-14900K, x64 | desktop baseline |
| `snapdragon-balanced.json` | Snapdragon X, ARM64 native | Windows power mode **Balanced** |
| `snapdragon-best-performance.json` | same machine | Windows power mode **Best Performance** |
| `snapdragon-best-performance-qnn.json` | same machine | Best Performance, `--qnn` attempted |

## Mean decode, ms (budget: 1000 ms)

| model | i9-14900K | Snapdragon balanced | Snapdragon best perf |
|---|---|---|---|
| tiny | 173 | 203 | **132** |
| base | 298 | 519 | **297** |
| small | 846 | 1580 | **987** |
| medium | — | 5032 | 3331 |

## What these show

**Power mode is worth ~1.6x.** Balanced to Best Performance takes small from 1580 ms to 987 ms -
across the responsiveness line rather than merely nearer it. A laptop on battery in Balanced is
the honest default to design for, so `small` cannot be treated as reliably live-capable even
though one configuration clears the budget.

**On Best Performance the Snapdragon matches an i9-14900K** at tiny and base (132 vs 173, 297 vs
298). That is a much stronger result than the emulated-x64 path this replaces.

**The NPU is still unmeasured.** Every `:qnn` entry failed with *"QNN execution provider is not
supported in this build"*, which is a fact about the wheel rather than about the hardware: QNN
ships in the separate `onnxruntime-qnn` distribution, and the plain `onnxruntime-genai` wheel
contains only `onnxruntime-genai.dll`. `--setup --qnn` now installs it. Until that runs, nothing
here says whether the Hexagon NPU helps.

**No accuracy signal.** Every model returned near-identical text because the clip is synthesised
speech, which is what makes it a fair latency test and a useless accuracy one. Whether a smaller
model handles accented speech well enough is a separate measurement.
