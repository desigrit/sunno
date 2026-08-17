"""The model repo table must stay in step with faster-whisper's.

server/models.py carries its own id -> Hub repo map instead of reading
faster_whisper.utils._MODELS, because importing anything under faster_whisper executes its
__init__, which imports ctranslate2 - a package with no wheel on Windows on ARM. Answering
"is this model already downloaded?" should not require the inference engine to load.

The copy is the point and also the risk: nothing makes the two agree, so when faster-whisper
adds or repoints a model, ours silently keeps the old answer. A wrong repo id does not fail
loudly either - it goes to the Hub as a bare name, returns 401, and surfaces as a download
failure for a model that may already be sitting in the cache.

Runs on x64, where faster_whisper is importable. On a machine where it is not, there is nothing
to compare against and the test skips - which is exactly the situation the copy exists for.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.models import CATALOG, _REPOS, _STREAM_REPOS, is_stream_model  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"PASS  {name}")
    else:
        failures.append(f"{name}: {detail}")
        print(f"FAIL  {name}  {detail}")


try:
    from faster_whisper.utils import _MODELS
except Exception as exc:  # noqa: BLE001
    print(f"SKIP  faster_whisper is not importable here ({type(exc).__name__})")
    print("      Nothing to compare against; this is the case the local table exists for.")
    raise SystemExit(0)

missing = sorted(set(_MODELS) - set(_REPOS))
check("every faster-whisper model id is known here", not missing, f"missing {missing}")

extra = sorted(set(_REPOS) - set(_MODELS))
check("no invented model ids", not extra, f"extra {extra}")

wrong = {k: (_REPOS[k], _MODELS[k]) for k in set(_REPOS) & set(_MODELS) if _REPOS[k] != _MODELS[k]}
check("every repo matches", not wrong, f"mismatched {wrong}")

# The picker's ids are the ones a user actually reaches, so a gap there is worse than a gap
# elsewhere in the table. A catalog id has to resolve through exactly one of the two tables:
# _REPOS for Whisper checkpoints, _STREAM_REPOS for streaming transducers, which are a
# different artifact rather than another Whisper size.
unknown = [
    e["id"] for e in CATALOG
    if e["id"] not in _REPOS and not is_stream_model(e["id"])
]
check("every catalog id resolves", not unknown, f"unresolvable {unknown}")

# An id in both tables would resolve differently depending on which check ran first, which
# is the kind of thing that works until someone reorders two lines.
both = [e["id"] for e in CATALOG if e["id"] in _REPOS and is_stream_model(e["id"])]
check("no id claimed by both tables", not both, f"ambiguous {both}")

# Every streaming model needs its four parts plus somewhere to fetch them, or the engine
# raises a missing-file error at construction that reads like a failed download.
incomplete = []
for model_id, spec in _STREAM_REPOS.items():
    missing = {"encoder", "decoder", "joiner", "tokens"} - set(spec.get("files", {}))
    if missing or not spec.get("repo"):
        incomplete.append((model_id, sorted(missing)))
check("every streaming model is complete", not incomplete, f"incomplete {incomplete}")

# The picker quotes approx_mb before anything is downloaded, so a streaming model with no
# catalog entry would be offered with no size at all.
uncatalogued = [m for m in _STREAM_REPOS if m not in {e["id"] for e in CATALOG}]
check("every streaming model is offered", not uncatalogued, f"missing {uncatalogued}")

print()
if failures:
    print(f"{len(failures)} FAILED")
    for f in failures:
        print(f"  {f}")
    raise SystemExit(1)

print("ALL PASS")
print(f"({len(_REPOS)} model ids, {len(CATALOG)} offered in the picker)")
