"""Only one output endpoint is the default, and only if we actually know which.

`is_default_output` decides which entry the picker offers when someone chooses to caption
system audio. Marking the wrong one is not cosmetic: it captures a device the user is not
listening to, and a transcript of the wrong audio is indistinguishable from a broken app.

The check used to be a two-way substring test against a name that defaulted to the empty
string, which is wrong in both directions. Every string contains "", so a machine where the
default lookup failed marked its whole output list as the default; and any two devices
sharing a fragment matched each other. Neither shows up on a machine whose device names
happen not to collide, which is why the cases below are all synthetic.

No pytest, matching the other tests here: run it directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import loopback  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"PASS  {name}")
    else:
        failures.append(f"{name}: {detail}")
        print(f"FAIL  {name}  {detail}")


SUFFIX = loopback._LOOPBACK_SUFFIX


class FakePyAudio:
    """The slice of pyaudiowpatch that list_loopback_devices touches."""

    paWASAPI = 2

    def __init__(self, names, default_name=None, raise_on_host_api=False):
        self._names = names
        self._default_name = default_name
        self._raise = raise_on_host_api
        self.terminated = False

    # module-level factory
    def PyAudio(self):  # noqa: N802
        return self

    def get_host_api_info_by_type(self, _kind):
        if self._raise:
            raise OSError("no WASAPI host api")
        return {"defaultOutputDevice": 99}

    def get_device_info_by_index(self, _idx):
        if self._default_name is None:
            raise OSError("no such device")
        return {"name": self._default_name}

    def get_loopback_device_info_generator(self):
        for i, n in enumerate(self._names):
            yield {
                "index": 100 + i,
                "name": n,
                "maxInputChannels": 2,
                "defaultSampleRate": 48000.0,
            }

    def terminate(self):
        self.terminated = True


def run(names, default_name=None, raise_on_host_api=False):
    fake = FakePyAudio(names, default_name, raise_on_host_api)
    saved = loopback._pyaudio
    loopback._pyaudio = lambda: fake
    try:
        return loopback.list_loopback_devices(), fake
    finally:
        loopback._pyaudio = saved


def defaults(devices):
    return [d["name"] for d in devices if d["is_default_output"]]


# The ordinary case. The capture-side twin carries a suffix the render endpoint does not,
# so the two names only match once both have been through the same stripping.
devices, fake = run(
    [f"Speakers (Realtek Audio){SUFFIX}", f"Headphones (Phonak){SUFFIX}"],
    default_name="Headphones (Phonak)",
)
check(
    "the default output is found across the loopback suffix",
    defaults(devices) == ["Headphones (Phonak)"],
    f"got {defaults(devices)}",
)
check("the suffix is stripped from what the user sees",
      all(SUFFIX not in d["name"] for d in devices),
      f"got {[d['name'] for d in devices]}")
check("PyAudio is always terminated", fake.terminated, "an instance was left holding PortAudio")

# The bug that mattered most: no default could be determined. Every name contains the empty
# string, so the old comparison marked all of them.
devices, _ = run(
    [f"Speakers (Realtek Audio){SUFFIX}", f"Headphones (Phonak){SUFFIX}"],
    raise_on_host_api=True,
)
check(
    "an unknown default marks nothing, not everything",
    defaults(devices) == [],
    f"got {defaults(devices)}",
)

devices, _ = run([f"Speakers{SUFFIX}"], default_name=None)
check(
    "a default that cannot be read marks nothing",
    defaults(devices) == [],
    f"got {defaults(devices)}",
)

# Names that share a fragment. "Realtek" sits inside "Speakers (Realtek Audio)", so the old
# two-way test called both of them the default.
devices, _ = run(
    [f"Realtek{SUFFIX}", f"Speakers (Realtek Audio){SUFFIX}"],
    default_name="Speakers (Realtek Audio)",
)
check(
    "a shared fragment does not make two defaults",
    defaults(devices) == ["Speakers (Realtek Audio)"],
    f"got {defaults(devices)}",
)

# The reverse direction of the same mistake: the default's name is a substring of another
# device's, which the old test also matched.
devices, _ = run(
    [f"Speakers{SUFFIX}", f"Speakers (Rear){SUFFIX}"],
    default_name="Speakers",
)
check(
    "a longer name is not mistaken for the default",
    defaults(devices) == ["Speakers"],
    f"got {defaults(devices)}",
)

# Whatever happens, the picker must never be offered two defaults to choose between.
for label, kwargs in [
    ("normal", dict(names=[f"A{SUFFIX}", f"B{SUFFIX}"], default_name="B")),
    ("unknown default", dict(names=[f"A{SUFFIX}", f"B{SUFFIX}"], raise_on_host_api=True)),
    ("collisions", dict(names=[f"A{SUFFIX}", f"A (Rear){SUFFIX}"], default_name="A")),
    ("default absent", dict(names=[f"A{SUFFIX}"], default_name="Something else")),
    ("no devices", dict(names=[], default_name="A")),
]:
    devices, _ = run(**kwargs)
    check(
        f"at most one default ({label})",
        len(defaults(devices)) <= 1,
        f"got {defaults(devices)}",
    )

check("no devices is not an error", run([], default_name="A")[0] == [], "expected an empty list")

print()
if failures:
    print(f"{len(failures)} FAILED")
    for f in failures:
        print(f"  {f}")
    raise SystemExit(1)

print("ALL PASS")
