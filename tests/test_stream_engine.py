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
from server.hardware import (  # noqa: E402
    _LAG_MS_CPU_4,
    _LAG_MS_CPU_16,
    _LAG_MS_CUDA,
)
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

# A streaming model missing from any lag table falls through to the unknown-model default,
# and the picker then tells the user a model that decodes in under 200 ms is about five
# seconds behind.
#
# All three tables must carry the SAME figure, and that is the assertion rather than a
# plausibility range. The engine pins provider="cpu" (asr_stream.py) and caps itself at
# four threads, so neither a graphics card nor a bigger machine changes the number: a
# streaming row that differs across tables is wrong by construction, whatever it says.
#
# A range was tried first and was close to useless. Against the tables as they stand, 20 to
# 600 ms admits five of the fifteen Whisper figures, including four of the five CUDA rows,
# and CUDA is the table most likely to be filled in by copying a neighbour. Equality
# catches every one of those, because no two Whisper tables agree.
figures = {name: table.get("stream-en")
           for name, table in (("CPU 4 threads", _LAG_MS_CPU_4),
                               ("CPU 16 threads", _LAG_MS_CPU_16),
                               ("CUDA", _LAG_MS_CUDA))}
for name, value in figures.items():
    check(f"stream-en has a lag figure for {name}", value is not None,
          "the picker would quote the unknown-model default instead")

present = [v for v in figures.values() if v is not None]
# Equality and a range catch different things rather than one superseding the other: the
# same wrong number written into all three rows passes equality, and _UNKNOWN_MODEL_LAG_MS
# is 5000, so the exact figure this guard exists to stop the picker quoting would be
# invisible to equality alone if somebody typed it in rather than defaulted to it.
check("the three lag figures agree", len(set(present)) == 1,
      f"{figures}, but this engine runs the same work on every machine, so a row that "
      "differs was copied from a Whisper model or measured wrongly")
check("and they are in the range a transducer decodes in",
      bool(present) and 20 <= present[0] <= 600,
      f"{present[:1]} ms is not a streaming figure")

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
#
# On its own this check has no teeth: `str.lower` maps one character at a time, so it
# passes by construction and would pass just as happily if `_readable` were deleted. It is
# a canary for a change nobody has made yet, most obviously a truecaser or a punctuation
# model, which is exactly the shape that fails it. So it is paired with a negative control
# below that proves the check can still fail, because a guard that cannot fail is worse
# than no guard: it reads like evidence.
try:
    from server.asr_stream import StreamingEngine  # noqa: E402

    def _prefix_stable(fn) -> bool:
        sample = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
        return all(fn(sample[:n]) == fn(sample)[:n] for n in range(5, len(sample)))

    check("a growing transcript is never rewritten",
          _prefix_stable(lambda t: StreamingEngine._readable(None, t)),
          "a prefix rendered differently once more words arrived")

    # A stand-in for the punctuation model that was dropped. The point of it is lookahead:
    # it decides how to render a word only once the NEXT word has arrived, so text already
    # on screen changes underneath the reader. That is the real failure shape ("lazy dog.
    # We" / "lazy dog we"), and it is what a truecaser presents.
    #
    # Paired with a causal version of the same transform, identical but for the lookahead.
    # An earlier attempt at this control was a left-to-right state machine, which is
    # prefix-stable by construction; the only thing making it fail was an appended full
    # stop, so it proved the check could notice a length change and nothing more. The pair
    # below pins the failure to the lookahead itself: the causal one must pass and the
    # lookahead one must fail, or this is measuring the wrong thing again.
    def _recase(text: str, lookahead: bool) -> str:
        # split(" ") rather than split(), because split() drops a trailing space and the
        # rejoined text would then differ from the original for reasons having nothing to
        # do with lookahead. The pin below caught exactly that on the first attempt.
        words = text.lower().split(" ")
        out = []
        for i, word in enumerate(words):
            following = words[i + 1] if i + 1 < len(words) else ""
            cue = following if lookahead else word
            out.append(word.capitalize() if cue.startswith(("j", "o")) else word)
        return " ".join(out)

    check("that check can actually fail",
          not _prefix_stable(lambda t: _recase(t, lookahead=True)),
          "a transform that rewrites an earlier word once a later one arrives was reported "
          "as stable, so the check above is proving nothing")
    check("and it fails because of the lookahead, not the reshaping",
          _prefix_stable(lambda t: _recase(t, lookahead=False)),
          "the same transform without lookahead also failed, so the control is firing on "
          "something incidental rather than on prefix revision")
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
