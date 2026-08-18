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

# Before any server import, because several modules resolve paths at import time. This
# points the app's data directory at a temp folder so the suite cannot write to the real
# profile; see tests/_isolate.py for what went wrong without it. The cost, stated there: the
# "already downloaded" checks below now always skip.
import tests._isolate  # noqa: E402,F401

from server.config import SAMPLE_RATE, Settings  # noqa: E402
from server.engine import available_engines, resolve_engine  # noqa: E402
from server.hardware import (  # noqa: E402
    _LAG_MS_CPU_4,
    _LAG_MS_CPU_16,
    _LAG_MS_CUDA,
)
from server.models import (  # noqa: E402
    AUTO_SELECT_EXCLUDED,
    CATALOG,
    _STREAM_REPOS,
    auto_selectable,
    is_stream_model,
    stream_model_is_cased,
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
    for stream_id in _STREAM_REPOS:
        check(f"{stream_id} selects the streaming engine",
              resolve_engine("auto", stream_id) == "stream",
              f"got {resolve_engine('auto', stream_id)}")

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

for stream_id, spec in _STREAM_REPOS.items():
    entry = next((e for e in CATALOG if e["id"] == stream_id), None)
    check(f"{stream_id} is offered in the picker", entry is not None)
    if entry:
        check(f"{stream_id} quotes its size", entry.get("approx_mb", 0) > 0)
        check(f"{stream_id} says which languages it covers", bool(entry.get("languages")))
        # There WAS a check here that an unlicensed model named its licence in this string.
        # It was removed with the string, on the owner's instruction, and deliberately not
        # replaced with a weaker version of itself. THIRD-PARTY-NOTICES.md is where the
        # licence position lives and it no longer claims the picker restates it.
    check(f"{stream_id} names a repo and four files",
          bool(spec.get("repo"))
          and {"encoder", "decoder", "joiner", "tokens"} <= set(spec.get("files", {})))

# Descriptions are one or two short sentences. On the real screen a four-sentence pair of
# streaming rows stood about twice the height of every Whisper row and read as a warning
# rather than a description, which is what prompted this.
#
# Measured in characters, not sentences. A first version counted full stops, which guards
# the shape that regression happened to take rather than the thing that went wrong: the same
# over-long text rewritten with commas is ONE sentence and longer still, and it sails
# through, while a legitimate short line like "Needs about 1.5 GB of memory." fails on the
# decimal point.
#
# The ceiling caps the catalogue string against the longest Whisper row, which is 68. It is
# NOT the rendered width: the picker draws ModelChoice.DetailWithSpeed, which prefixes a
# delay in brackets, thirteen or fourteen characters depending on magnitude and omitted
# entirely when the lag is unknown. A relative bound against the rows it sits beside is the
# property worth having here, and no row carries a prefix long enough to change the ordering.
#
# Applied to every entry rather than only the streaming pair, because the Whisper rows are
# what define the house style being matched, and a guard that exempts the examples it is
# imitating would not notice the style moving.
_DETAIL_MAX = 90   # longest today is 82; base, the longest Whisper row, is 68
for entry in CATALOG:
    detail = entry.get("detail", "")
    check(f"{entry['id']} keeps its catalogue description short",
          0 < len(detail) <= _DETAIL_MAX,
          f"{len(detail)} characters against a {_DETAIL_MAX} ceiling, so this row will "
          "stand well above the ones beside it")

# A streaming model missing from any lag table falls through to the unknown-model default,
# and the picker then tells the user a model that decodes in under 200 ms is about five
# seconds behind.
#
# All three tables must carry the SAME figure per model, and that is the assertion rather
# than a plausibility range. The engine pins provider="cpu" and caps itself at four threads,
# so neither a graphics card nor a bigger machine changes the number.
#
# A range was tried first and was close to useless. Against the tables as they stand, 20 to
# 600 ms admits five of the fifteen Whisper figures, including four of the five CUDA rows,
# and CUDA is the table most likely to be filled in by copying a neighbour. Equality
# catches every one of those, because no two Whisper tables agree. Both are kept: the same
# wrong number written into all three rows passes equality, and _UNKNOWN_MODEL_LAG_MS is
# 5000, so the exact figure this guard exists to stop the picker quoting would be invisible
# to equality alone if somebody typed it in rather than defaulted to it.
for stream_id in _STREAM_REPOS:
    figures = {name: table.get(stream_id)
               for name, table in (("CPU 4 threads", _LAG_MS_CPU_4),
                                   ("CPU 16 threads", _LAG_MS_CPU_16),
                                   ("CUDA", _LAG_MS_CUDA))}
    for name, value in figures.items():
        check(f"{stream_id} has a lag figure for {name}", value is not None,
              "the picker would quote the unknown-model default instead")

    present = [v for v in figures.values() if v is not None]
    check(f"{stream_id}: the three lag figures agree", len(set(present)) == 1,
          f"{figures}, but this engine runs the same work on every machine, so a row that "
          "differs was copied from a Whisper model or measured wrongly")
    check(f"{stream_id}: and they are in the range a transducer decodes in",
          bool(present) and 20 <= present[0] <= 600,
          f"{present[:1]} ms is not a streaming figure")

# The tables above file ONE figure for both thread counts, and the justification is not that
# the two measured the same. They did not; more threads are measurably worse for these
# models. The justification is that asr_stream._threads() never asks for more than four, so
# the sixteen-thread column describes something the app cannot do. That is a coupling between
# two files, so it is asserted rather than left in a comment: raise the cap and this fails,
# which is the moment those rows need remeasuring.
from server.asr_stream import _threads as _stream_threads  # noqa: E402

# Asserted on the source as well as the return value. `_threads() <= 4` is the property that
# matters, but on a four-core machine it stays true even if somebody raises the cap to
# sixteen, so on its own it would pass on exactly the modest hardware these models exist for.
_stream_src = (Path(__file__).resolve().parents[1] / "server" / "asr_stream.py").read_text(
    encoding="utf-8")
check("the thread cap the lag tables assume is still in place",
      _stream_threads() <= 4 and "min(4," in _stream_src,
      f"_threads() returns {_stream_threads()} and the cap literal is "
      f"{'present' if 'min(4,' in _stream_src else 'GONE'}, so the 16-thread lag rows no "
      "longer describe what the engine does and must be remeasured")

# Models with no declared licence must never be chosen for someone. THIRD-PARTY-NOTICES.md
# says so in as many words, and a disclosure enforced by nothing is worth nothing.
#
# Swept across the whole plausible hardware range rather than spot-checked, because the two
# branches of default_model fail differently: the scan returns the first model inside the
# latency budget, which ordering protects, while the fallback returns the QUICKEST model of
# all when nothing is inside it, and the quickest streaming model is the unlicensed one.
# Only a slow-machine score reaches that second branch.
import server.hardware as _hw  # noqa: E402

catalog_ids = [e["id"] for e in CATALOG]
picked = set()
branches = {"scan": 0, "fallback": 0}
_real_score = _hw.cpu_score
try:
    for score in (5, 10, 20, 40, 60, 80, 100, 150, 200, 400):
        _hw.cpu_score = lambda _s=score: _s
        _hw.estimated_lag_ms.cache_clear() if hasattr(
            _hw.estimated_lag_ms, "cache_clear") else None
        for device in ("cpu", "cuda"):
            picked.add(_hw.default_model(catalog_ids, device=device))
            # Which branch answered. Recomputed rather than instrumented, so this stays a
            # statement about the scores swept and not about default_model's internals.
            responsive = any(
                _hw.estimated_lag_ms(m, device) <= _hw.RESPONSIVE_LAG_MS
                for m in catalog_ids if auto_selectable(m))
            branches["scan" if responsive else "fallback"] += 1
finally:
    _hw.cpu_score = _real_score
    if hasattr(_hw.estimated_lag_ms, "cache_clear"):
        _hw.estimated_lag_ms.cache_clear()

check("the picker never lands on an unlicensed model by itself",
      not (picked & AUTO_SELECT_EXCLUDED),
      f"default_model chose {sorted(picked & AUTO_SELECT_EXCLUDED)} on some machine, but "
      "THIRD-PARTY-NOTICES.md tells the reader that never happens")
check("and it still returns something on every machine tried",
      all(auto_selectable(m) for m in picked) and len(picked) > 1,
      f"got {sorted(picked)}")
# Without this the sweep quietly degrades into a test of the scan branch only, which
# catalogue order already protects, and still prints PASS. Exactly one of the ten scores
# reaches the fallback today, so a single edit to that tuple would do it.
check("and the sweep actually exercised both branches", min(branches.values()) > 0,
      f"{branches}: the scores swept no longer reach one of the two branches, so the check "
      "above is only testing the other one")

# The rule above is only half the enforcement, and it is the half the product does not run.
# hardware.default_model fires only when nobody passes --model, and both launchers always
# do (BackendHost.cs builds "--model" unconditionally, run.ps1 defaults it), so the screen
# that actually preselects for a first-time user is OnModelRequired in the frontend. That
# code cannot be reached from here, but the data it decides on can: it filters on the
# auto_select flag this payload carries, so if the flag stops being sent the C# silently
# falls back to selecting everything, including the model the notices promise it will not.
from server.models import catalog_with_status  # noqa: E402

payload = catalog_with_status(device="cpu")
check("every catalogue entry tells the frontend whether it may be auto-selected",
      all("auto_select" in e for e in payload),
      "OnModelRequired defaults a missing flag to true, so an unlicensed model would be "
      "preselectable again with nothing failing here")
flagged = {e["id"] for e in payload if not e.get("auto_select", True)}
check("and the flag marks exactly the models the notices say are never chosen",
      flagged == set(AUTO_SELECT_EXCLUDED),
      f"payload marks {sorted(flagged)}, AUTO_SELECT_EXCLUDED is {sorted(AUTO_SELECT_EXCLUDED)}")

# The fastest-first branch is the one that bites: it ignores catalogue order entirely, so an
# excluded model that is also the quickest would win it outright. Replays that branch over
# the payload the frontend receives.
selectable = [e for e in payload if e.get("auto_select", True)]
fastest = min(selectable, key=lambda e: e["lag_ms"] if e["lag_ms"] > 0 else 10**9)
check("the frontend's fastest-first fallback cannot land on an excluded model",
      fastest["id"] not in AUTO_SELECT_EXCLUDED, f"would preselect {fastest['id']}")

# The C# half of this contract fails open by design: CaptionClient reads the flag with
# `?? true` so an older backend still works. That is right for compatibility and wrong for
# safety, because deleting the filter or misspelling the field leaves every Python test
# passing and the build clean, while THIRD-PARTY-NOTICES.md still promises the screen
# honours the mark. There is no C# test project, so these lines are asserted by reading the
# sources, the same way the notices are grepped above.
#
# It pins the CONSUMERS, not the flag, and that distinction was earned. A first version
# looked only for `m.AutoSelect`, which survives the most likely future revert: keeping the
# `selectable` declaration but pointing the three preselect branches back at `choices`. That
# is the exact shape this code had before, it reintroduces the whole defect, and the needle
# was still present. Each pattern below is a line that decides.
_app = Path(__file__).resolve().parents[1] / "app"
_CSHARP_CONTRACT = [
    (_app / "Services" / "CaptionClient.cs", ['"auto_select"'],
     "the frontend would stop reading the flag and default every model to selectable"),
    (_app / "MainWindow.xaml.cs",
     ["choices.Where(m => m.AutoSelect)",
      "var preferred = selectable",
      "?? selectable.FirstOrDefault(m => m.Available)",
      "?? selectable.OrderBy("],
     "a preselect branch would draw from every model again, including excluded ones"),
]
for path, needles, why in _CSHARP_CONTRACT:
    source = path.read_text(encoding="utf-8") if path.is_file() else ""
    missing = [n for n in needles if n not in source]
    check(f"{path.name} still honours the auto-select contract", not missing,
          f"missing {missing}: {why}")

# Every model downloaded at first run has to appear in the notices, and a streaming model
# is two clicks from a Store submission rather than a research script.
#
# Walks the whole spec rather than reading spec["repo"], because the first version of this
# check read only the top-level repo and would have passed the very thing it exists to
# catch: an earlier draft attached a second, unlicensed model under spec["punct"]["repo"],
# and a guard that only looks at the top level would have called that clean.
#
# It also collects bare http(s) strings under any key, not just values under "repo". A
# review of a plan to fetch the punctuation model from a GitHub release tarball found that
# a spec shaped `{"punct": {"url": "https://github.com/..."}}` returned nothing here, so the
# loop below ran zero times and passed vacuously. Choosing a more trustworthy source would
# have silently switched this guard off, which is the worst way for a guard to fail.
def _repo_ids(value) -> list[str]:
    """Every model source anywhere in a spec: Hub repo ids and any http(s) URL."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        found = []
        for key, item in value.items():
            if isinstance(item, str):
                if key == "repo" or item.startswith(("http://", "https://")):
                    found.append(item)
            elif isinstance(item, (dict, list)):
                found.extend(_repo_ids(item))
        return found
    if isinstance(value, list):
        return [r for item in value for r in _repo_ids(item)]
    return []


# _repo_ids is the guard's guard, so it gets its own fixtures. Each shape below has been a
# real or proposed spec at some point; the url one was found passing vacuously in review.
_REPO_ID_CASES = [
    ({"repo": "org/a"}, {"org/a"}),
    ({"punct": {"repo": "org/b"}}, {"org/b"}),
    ({"punct": {"url": "https://github.com/k2-fsa/x.tar.bz2"}},
     {"https://github.com/k2-fsa/x.tar.bz2"}),
    ({"repos": ["org/c", "org/d"]}, {"org/c", "org/d"}),
    ({"repo": "org/e", "extra": [{"repo": "org/f"}]}, {"org/e", "org/f"}),
    ({"files": {"encoder": "encoder.onnx"}}, set()),
]
for spec_in, expected in _REPO_ID_CASES:
    got = set(_repo_ids(spec_in))
    check(f"_repo_ids finds every source in {spec_in}", got == expected, f"got {got}")


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

    # Bound to a stand-in rather than called as StreamingEngine._readable(None, text). That
    # unbound form worked while _readable ignored self, and broke the moment it started
    # reading self._cased: the AttributeError surfaced through the except below as "the
    # engine module imports: FAIL", a failure naming the wrong thing entirely.
    class _Fake:
        def __init__(self, cased: bool) -> None:
            self._cased = cased

        _readable = StreamingEngine._readable

    def _prefix_stable(fn) -> bool:
        sample = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
        return all(fn(sample[:n]) == fn(sample)[:n] for n in range(5, len(sample)))

    for cased in (False, True):
        check(f"a growing transcript is never rewritten (self-cased={cased})",
              _prefix_stable(_Fake(cased)._readable),
              "a prefix rendered differently once more words arrived")

    # The whole reason to offer Kroko is that it punctuates and capitalises itself, so the
    # engine must leave it alone. Lower-casing it would silently produce two models that
    # look identical on screen, with nothing failing.
    check("a self-cased model keeps its capitals and punctuation",
          _Fake(True)._readable("Hello there. How are you?") == "Hello there. How are you?")
    check("a model that emits upper case is lower-cased",
          _Fake(False)._readable("HELLO THERE") == "hello there")

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


# The casing flag has to be wired to the right models, read from the spec rather than
# restated here so it cannot drift from what ships. What is pinned is that the two DISAGREE:
# if both were marked the same way one of them would be rendering wrongly, and every other
# check in this file would still pass.
check("stream-en is not self-cased", not stream_model_is_cased("stream-en"))
check("stream-en-kroko is self-cased", stream_model_is_cased("stream-en-kroko"))
check("an unknown id defaults to lower-casing", not stream_model_is_cased("no-such-model"))
check("the two streaming models disagree about casing",
      len({stream_model_is_cased(m) for m in _STREAM_REPOS}) == 2,
      "both are marked the same, so one of them renders wrongly")


# --- local paths -----------------------------------------------------------

# Only meaningful once downloaded. Skipped rather than failed, because a fresh clone has
# no models and that is not a defect.
for stream_id in _STREAM_REPOS:
    try:
        paths = stream_model_paths(stream_id)
        check(f"{stream_id}: all four parts are present locally",
              all(paths[k].is_file() for k in ("encoder", "decoder", "joiner", "tokens")))
    except FileNotFoundError:
        print(f"SKIP  {stream_id} not downloaded, local path checks skipped")

print()
if failures:
    print(f"{len(failures)} FAILED")
    for f in failures:
        print(f"  {f}")
    raise SystemExit(1)

print("ALL PASS")
