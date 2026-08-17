"""The streaming engine's contract, without needing the model downloaded.

Two things here are worth a test rather than a reading.

The engine is used statelessly on purpose, because the pipeline hands over the whole
utterance every time and ``AudioConditioner`` renormalises gain over whatever it is given.
Measured, one loud word changes already-decoded samples by 67 percent, so an engine that
kept state across partials would be splicing two different gains together with nothing to
catch it. The test below pins that property of the conditioner, so that if someone later
makes the conditioner streaming-stable, this fails and points at the engine that could
then be made stateful.

The other is that asking for a streaming model has to select the streaming engine. Model
and engine are not independent here: a transducer cannot be loaded by CTranslate2, so a
wrong answer is a startup crash rather than a slow decode.

Run:  .venv\\Scripts\\python.exe tests\\test_stream_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.config import SAMPLE_RATE, Settings  # noqa: E402
from server.engine import available_engines, resolve_engine  # noqa: E402
from server.models import (  # noqa: E402
    CATALOG,
    _STREAM_REPOS,
    is_stream_model,
    stream_model_paths,
)
from server.preprocess import AudioConditioner  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if not ok and detail else ''}")
    if not ok:
        failures.append(f"{name}: {detail}")


# --- selection -------------------------------------------------------------

have = available_engines()
check("sherpa-onnx is importable", have.get("stream", False),
      "the app already ships it for speaker embeddings")

if have.get("stream"):
    check("a streaming model selects the streaming engine",
          resolve_engine("auto", "stream-en") == "stream",
          f"got {resolve_engine('auto', 'stream-en')}")

if have.get("ct2"):
    check("a Whisper model still selects CTranslate2",
          resolve_engine("auto", "base") == "ct2",
          f"got {resolve_engine('auto', 'base')}")

# An unknown id must not be quietly treated as streaming, or a typo would send someone to
# an engine that cannot load a Whisper checkpoint.
check("an unknown id is not treated as streaming", not is_stream_model("no-such-model"))


# --- the reason the engine is stateless ------------------------------------

# Cumulative audio through the conditioner, the way pipeline._maybe_partial does it, with
# one loud word arriving partway through.
rng = np.random.default_rng(0)
audio = (rng.standard_normal(SAMPLE_RATE * 12) * 0.02).astype(np.float32)
audio[SAMPLE_RATE * 8 : SAMPLE_RATE * 8 + SAMPLE_RATE // 2] *= 25.0

conditioner = AudioConditioner(Settings())
worst = 0.0
previous = None
for end in range(int(SAMPLE_RATE * 0.7), len(audio), int(SAMPLE_RATE * 0.45)):
    current = conditioner(audio[:end])
    if previous is not None:
        overlap = len(previous)
        drift = float(np.abs(current[:overlap] - previous).max())
        worst = max(worst, drift / max(float(np.abs(previous).max()), 1e-9))
    previous = current

check("the conditioned prefix is not stable, so the engine must stay stateless",
      worst > 0.1, f"worst drift only {worst:.1%}, the engine could now hold state")


# --- catalogue -------------------------------------------------------------

entry = next((e for e in CATALOG if e["id"] == "stream-en"), None)
check("the streaming model is offered in the picker", entry is not None)
if entry:
    check("its size is quoted", entry.get("approx_mb", 0) > 0)
    check("it says which languages it covers", bool(entry.get("languages")))

spec = _STREAM_REPOS.get("stream-en")
check("it names a repo and four files", bool(spec) and bool(spec.get("repo"))
      and {"encoder", "decoder", "joiner", "tokens"} <= set(spec.get("files", {})))

# Every model downloaded at first run has to appear in the notices, and a streaming model
# is two clicks from a Store submission rather than a research script.
#
# Walks the whole spec rather than reading spec["repo"], because the first version of this
# check read only the top-level repo and would have passed the very thing it exists to
# catch: an earlier draft attached a second, unlicensed model under spec["punct"]["repo"],
# and a guard that only looks at the top level would have called that clean.
def _repo_ids(value) -> list[str]:
    """Every Hub repo id anywhere in a model spec, however it is nested."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        found = []
        for key, item in value.items():
            if key == "repo" and isinstance(item, str):
                found.append(item)
            elif isinstance(item, (dict, list)):
                found.extend(_repo_ids(item))
        return found
    if isinstance(value, list):
        return [r for item in value for r in _repo_ids(item)]
    return []


notices = (Path(__file__).resolve().parents[1] / "THIRD-PARTY-NOTICES.md").read_text(
    encoding="utf-8")
for model_id, model_spec in _STREAM_REPOS.items():
    repos = _repo_ids(model_spec)
    check(f"{model_id} names at least one repo", bool(repos))
    for repo in repos:
        check(f"{repo} is in THIRD-PARTY-NOTICES", repo in notices,
              f"downloaded at first run but not in the notices")

# The rewrite bug this engine was blocked on came from text that changed as later words
# arrived. Whatever the engine does to a transcript must not depend on anything but the
# transcript, or already-displayed captions start moving under the reader again.
try:
    from server.asr_stream import StreamingEngine  # noqa: E402

    readable = StreamingEngine._readable
    sample = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
    stable = all(
        readable(None, sample[:n]) == readable(None, sample)[:n]
        for n in range(5, len(sample))
    )
    check("a growing transcript is never rewritten", stable,
          "a prefix rendered differently once more words arrived")
except Exception as exc:  # pragma: no cover
    check("the engine module imports", False, str(exc))


# --- local paths -----------------------------------------------------------

# Only meaningful once downloaded. Skipped rather than failed, because a fresh clone has
# no models and that is not a defect.
try:
    paths = stream_model_paths("stream-en")
    check("all four parts are present locally",
          all(paths[k].is_file() for k in ("encoder", "decoder", "joiner", "tokens")))
except FileNotFoundError:
    print("SKIP  streaming model not downloaded, local path checks skipped")

print()
if failures:
    print(f"{len(failures)} FAILED")
    for f in failures:
        print(f"  {f}")
    raise SystemExit(1)

print("ALL PASS")
