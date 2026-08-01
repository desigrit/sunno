"""Speaker ids must survive a merge.

Ids were list positions once. Merging one speaker into another deleted from the middle of
that list and renumbered everybody above, while the UI relabels existing captions by
matching on id - so folding two speakers together silently re-attributed other people's
already-scrolled lines to the wrong person. For a deaf user the transcript is the only
record of who said what, so that is the worst class of bug this app can have.

These exercise the real SpeakerIdentifier. The constructor loads a sherpa-onnx model, which
none of this needs, so instances are built with object.__new__ and the fields the tested
methods actually touch. identify() is covered separately with a stubbed embed().
"""
import json
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from server.speaker import SpeakerIdentifier

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}   {detail}")
        failures.append(name)


def make(names, profile_path=None, threshold=0.5, max_speakers=8):
    """A SpeakerIdentifier with `names` enrolled, no model loaded."""
    s = object.__new__(SpeakerIdentifier)
    s._lock = threading.Lock()
    s.profile_path = Path(profile_path) if profile_path else None
    s._profiles = {}
    s._next_id = 0
    s.threshold = threshold
    s.max_speakers = max_speakers
    s.min_identify_samples = 0
    s.min_new_speaker_samples = 0
    for i, n in enumerate(names):
        v = np.zeros(8, dtype=np.float32)
        v[i] = 1.0                       # orthogonal, so identify() picks deterministically
        p = s._mint(v, name=n, pinned=n is not None)
        p.count = 1
    return s


def ids_labels(store):
    return {r["id"]: r["label"] for r in store.roster()}


# --- the regression itself -------------------------------------------------
store = make(["Priya", "Marco", "Sarah"])
before = ids_labels(store)
priya, marco, sarah = 0, 1, 2

store.merge(priya, marco)
after = ids_labels(store)

check("merge: survivors keep their ids",
      marco in after and sarah in after,
      f"roster is {after}")
check("merge: the survivor's label is unchanged",
      after.get(marco) == before[marco],
      f"Marco was {before[marco]!r}, now {after.get(marco)!r}")
check("merge: the uninvolved speaker's label is unchanged",
      after.get(sarah) == before[sarah],
      f"Sarah was {before[sarah]!r}, now {after.get(sarah)!r}")
check("merge: the absorbed id is gone from the roster",
      priya not in after,
      f"roster is {after}")

# --- lookups must follow the id, not a position ----------------------------
store = make(["Priya", "Marco", "Sarah"])
store.merge(0, 1)                        # Priya folded into Marco; Sarah keeps id 2

store.rename(sarah, "Sarah Chen")
check("rename after a merge names the right person",
      ids_labels(store).get(sarah) == "Sarah Chen",
      f"roster is {ids_labels(store)}")

# Clearing the name is a real edit, not a no-op: the UI used to skip the call entirely when
# the box was empty, so a name typed by mistake could not be taken back.
store.rename(sarah, "")
check("an empty name unnames the speaker",
      ids_labels(store).get(sarah) == "Speaker 3",
      f"roster is {ids_labels(store)}")
check("an empty name unpins the profile, so it stops being saved",
      store._profiles[sarah].pinned is False)
check("an unnamed speaker reports named=False",
      any(r["id"] == sarah and r["named"] is False for r in store.roster()))
store.rename(sarah, "Sarah Chen")   # restore for the checks that follow

store.set_self(sarah, True)
check("set_self after a merge marks the right person",
      store.is_self_speaker(sarah) and not store.is_self_speaker(marco))
check("set_self is exclusive",
      sum(1 for r in store.roster() if r["is_self"]) == 1)
check("label() resolves by id after a merge",
      store.label(sarah) == "Sarah Chen",
      f"got {store.label(sarah)!r}")
check("label() of a retired id is None, not another person",
      store.label(priya) is None,
      f"got {store.label(priya)!r}")
check("is_self_speaker of a retired id is False",
      store.is_self_speaker(priya) is False)

# --- unnamed speakers keep their number ------------------------------------
store = make([None, None, None])
before = ids_labels(store)
store.merge(0, 1)
after = ids_labels(store)
check("unnamed speakers are not renumbered by a merge",
      after.get(2) == before[2],
      f"speaker 2 was {before[2]!r}, now {after.get(2)!r}")

# --- ids are never reused --------------------------------------------------
store = make(["Priya", "Marco"])
store.merge(0, 1)
fresh = np.zeros(8, dtype=np.float32)
fresh[7] = 1.0
with store._lock:
    minted = store._mint(fresh)
check("a retired id is never handed to a new speaker",
      minted.id not in (0,) and minted.id == 2,
      f"new speaker got id {minted.id}")

# --- guards ----------------------------------------------------------------
store = make(["Priya", "Marco"])
check("merge with an unknown source is refused", store.merge(99, 1) is False)
check("merge with an unknown target is refused", store.merge(0, 99) is False)
check("merge into self is refused", store.merge(0, 0) is False)
check("rename of an unknown id is refused", store.rename(99, "X") is False)
check("set_self of an unknown id is refused", store.set_self(99, True) is False)
check("label(None) is None", store.label(None) is None)
check("is_self_speaker(None) is False", store.is_self_speaker(None) is False)

# --- reset keeps named people ----------------------------------------------
store = make(["Priya", None, "Sarah"])
store.reset()
labels = ids_labels(store)
check("reset keeps pinned speakers and drops discovered ones",
      sorted(labels.values()) == ["Priya", "Sarah"],
      f"roster is {labels}")

# --- merge must not discard what the user asserted ------------------------
# save() writes only pinned profiles, so a named-but-unpinned survivor is reported as named
# and then silently forgotten on the next launch.
store = make(["Priya", None])           # a named speaker and a discovered one
store.merge(0, 1)                       # the ordinary direction: "this stranger is Priya"
survivor = store._profiles[1]
check("merging a named speaker into a discovered one keeps the name",
      survivor.name == "Priya", f"name is {survivor.name!r}")
check("...and pins it, so save() does not drop it",
      survivor.pinned is True,
      "survivor is named but unpinned - the name is gone on the next launch")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "speakers.json"
    store = make(["Priya", None], profile_path=path)
    store.merge(0, 1)
    store.save()
    written = json.loads(path.read_text(encoding="utf-8"))
    check("the merged survivor's name actually reaches disk",
          [e["name"] for e in written] == ["Priya"], f"disk holds {written}")

# is_self drives the dimming and the clarity score, so losing it silently changes how the
# user's own lines render.
store = make([None, None])
store.set_self(0, True)
store.merge(0, 1)
check("merging your own profile into another keeps 'this is me'",
      store.is_self_speaker(1) is True)
check("...and pins the survivor",
      store._profiles[1].pinned is True)
check("still exactly one self after the merge",
      sum(1 for r in store.roster() if r["is_self"]) == 1)

# The reverse direction was already correct; guard it so it stays that way.
store = make([None, "Marco"])
store.merge(0, 1)
check("merging a discovered speaker into a named one keeps the name",
      store._profiles[1].name == "Marco" and store._profiles[1].pinned is True)

# --- delete ----------------------------------------------------------------
store = make(["Priya", "Marco", "Sarah"])
fallback = store.delete(priya)
after = ids_labels(store)

check("delete returns a generic label, never the name",
      fallback == "Speaker 1", f"got {fallback!r}")
check("delete removes only that speaker",
      sorted(after.values()) == ["Marco", "Sarah"], f"roster is {after}")
check("delete leaves the other ids alone",
      marco in after and sarah in after, f"roster is {after}")
check("delete of an unknown id returns None",
      store.delete(99) is None)
check("deleting twice returns None the second time",
      store.delete(priya) is None)

# The fallback label must not collide with anyone still present, now or later. Ids are
# never reused, so the retired id's label can never be minted for a live speaker.
store = make([None, None, None])
live_labels_before = set(ids_labels(store).values())
freed = store.delete(1)
check("the fallback label is not in use by a living speaker",
      freed not in set(ids_labels(store).values()),
      f"fallback {freed!r} collides with {ids_labels(store)}")
check("the fallback matches what the speaker was called before deletion",
      freed == "Speaker 2" and "Speaker 2" in live_labels_before,
      f"got {freed!r}, roster had {live_labels_before}")

with store._lock:
    after_delete = store._mint(np.zeros(4, dtype=np.float32))
check("a speaker discovered after a delete cannot take the retired label",
      store.label(after_delete.id) != freed,
      f"new speaker is {store.label(after_delete.id)!r}, retired label was {freed!r}")

# Deleting a named speaker must drop them from disk too, not just from the roster.
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "speakers.json"
    store = make(["Priya", "Marco"], profile_path=path)
    store.save()
    check("both named speakers are on disk before the delete",
          len(json.loads(path.read_text(encoding="utf-8"))) == 2)
    store.delete(0)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    check("a deleted speaker is removed from disk",
          [e["name"] for e in on_disk] == ["Marco"], f"disk holds {on_disk}")

# --- persistence: no ids on disk, no migration needed ----------------------
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "speakers.json"

    # A file written before ids existed: no "id" key anywhere.
    legacy = [
        {"name": "Priya", "count": 4, "is_self": False, "centroid": [1.0, 0.0, 0.0, 0.0]},
        {"name": "Marco", "count": 7, "is_self": True, "centroid": [0.0, 1.0, 0.0, 0.0]},
    ]
    path.write_text(json.dumps(legacy), encoding="utf-8")

    store = make([], profile_path=path)
    store.load()
    loaded = ids_labels(store)
    check("a legacy id-less speakers.json loads without migration",
          sorted(loaded.values()) == ["Marco", "Priya"],
          f"roster is {loaded}")
    check("loaded speakers get compact ids from zero",
          sorted(loaded.keys()) == [0, 1],
          f"ids are {sorted(loaded.keys())}")
    check("is_self survives the load",
          any(r["is_self"] and r["label"] == "Marco" for r in store.roster()))
    check("count survives the load",
          sorted(p.count for p in store._profiles.values()) == [4, 7])

    with store._lock:
        nxt = store._mint(np.zeros(4, dtype=np.float32))
    check("a speaker discovered after a load does not collide with a loaded id",
          nxt.id == 2, f"got id {nxt.id}")

    # Save must not write ids: they mean nothing outside the session that minted them.
    store.merge(0, 1)
    store.save()
    written = json.loads(path.read_text(encoding="utf-8"))
    check("save() writes no ids", all("id" not in e for e in written),
          f"wrote {written}")
    check("save() persists only pinned speakers", len(written) == 1,
          f"wrote {len(written)} entries")

# --- identify() returns durable ids ----------------------------------------
store = make(["Priya", "Marco", "Sarah"])
store._extractor = None


def fake_embed(vec):
    def _embed(_audio):
        return vec
    return _embed


audio = np.ones(16000, dtype=np.float32)

sarah_vec = np.zeros(8, dtype=np.float32)
sarah_vec[2] = 1.0
store.embed = fake_embed(sarah_vec)
sid_before, _ = store.identify(audio)
check("identify() finds the right speaker", sid_before == sarah,
      f"got {sid_before}")

store.merge(0, 1)                        # unrelated merge, below Sarah
sid_after, _ = store.identify(audio)
check("identify() returns the same id after an unrelated merge",
      sid_after == sid_before,
      f"was {sid_before}, now {sid_after}")

# A genuinely new voice gets a fresh id, not a recycled one.
new_vec = np.zeros(8, dtype=np.float32)
new_vec[6] = 1.0
store.embed = fake_embed(new_vec)
new_id, _ = store.identify(audio)
check("a new voice gets an unused id",
      new_id not in (sarah, 0) and new_id == 3,
      f"got {new_id}")
check("the new voice did not steal an existing label",
      store.label(sarah) == "Sarah",
      f"Sarah is now {store.label(sarah)!r}")

# --- max_speakers still counts people, not ids -----------------------------
store = make([None, None], max_speakers=2)
store.embed = fake_embed(np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
capped, _ = store.identify(audio)
check("max_speakers refuses a third speaker", capped is None,
      f"got {capped}")

store.merge(0, 1)                        # now one person, ids up to 1 retired
store.embed = fake_embed(np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
freed, _ = store.identify(audio)
check("max_speakers frees up after a merge, and uses a fresh id",
      freed == 2, f"got {freed}")

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("ALL PASS")
print("(merge regression, delete, id-based lookups, reset, persistence/migration,"
      " identify, capacity)")
