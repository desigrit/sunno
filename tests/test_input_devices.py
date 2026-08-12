"""The microphone picker should offer devices that exist, not devices a driver remembers.

PortAudio exposes the same machine through four Windows host APIs. Only WASAPI enumerates
with DEVICE_STATE_ACTIVE, so only WASAPI can tell a plugged-in microphone from an empty
jack. On the desktop this was written against, that was the difference between 22 entries
and 4.

The fake enumerations below mirror the real shape of sounddevice's output: a flat device
list whose "hostapi" is an integer index into a separate host API list, with output-only
devices interleaved among the inputs.

No pytest here, matching the other tests in this directory: run it directly.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import audio  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"PASS  {name}")
    else:
        failures.append(f"{name}: {detail}")
        print(f"FAIL  {name}  {detail}")


MME, DSOUND, WASAPI, WDMKS = 0, 1, 2, 3

HOST_APIS = [
    {"name": "MME", "default_input_device": 1},
    {"name": "Windows DirectSound", "default_input_device": 4},
    {"name": "Windows WASAPI", "default_input_device": 7},
    {"name": "Windows WDM-KS", "default_input_device": 9},
]


def dev(name: str, hostapi: int, inputs: int = 2) -> dict:
    return {
        "name": name,
        "hostapi": hostapi,
        "max_input_channels": inputs,
        "default_samplerate": 48000.0,
    }


@contextlib.contextmanager
def portaudio(devices, host_apis=HOST_APIS, default_device=(1, 5)):
    """Swap in a fake PortAudio for the duration of a block.

    sd.query_hostapis() is called both with an index, to name one device's API, and with no
    argument, to walk every API, so the fake has to honour both signatures.
    """
    saved = (audio.sd.query_devices, audio.sd.query_hostapis, audio.sd.default)
    audio.sd.query_devices = lambda: list(devices)
    audio.sd.query_hostapis = lambda idx=None: host_apis if idx is None else host_apis[idx]
    audio.sd.default = type("Default", (), {"device": default_device})()
    try:
        yield
    finally:
        audio.sd.query_devices, audio.sd.query_hostapis, audio.sd.default = saved


def names(devices) -> list[str]:
    return [d["name"] for d in devices]


def listed(devices, **kw):
    """The devices offered, with stdout swallowed so the count line does not clutter output."""
    with portaudio(devices, **kw), contextlib.redirect_stdout(io.StringIO()):
        return audio.list_input_devices()


# A machine much like the one that prompted this: three real microphones, plus an unplugged
# jack, a legacy mixer, a camera that has not been connected in months, and duplicate views
# of the same hardware through older APIs.
MIXED = [
    dev("Microsoft Sound Mapper - Input", MME),          # 0  virtual
    dev("Microphone (NVIDIA Broadcast)", MME),           # 1  duplicate of 7
    dev("Speakers (Realtek)", MME, inputs=0),            # 2  output only
    dev("Line In (Realtek USB Audio)", DSOUND),          # 3  unplugged jack
    dev("Microphone (NVIDIA Broadcast)", DSOUND),        # 4  duplicate of 7
    dev("Stereo Mix (Realtek USB Audio)", DSOUND),       # 5  legacy loopback
    dev("Microphone (Logitech BRIO)", WDMKS),            # 6  camera, not connected
    dev("Microphone (NVIDIA Broadcast)", WASAPI),        # 7  real
    dev("Headset (R-Phonak hearing aid)", WASAPI),       # 8  real
    dev("Microphone (Umik-1)", WASAPI),                  # 9  real
]

REAL = [
    "Microphone (NVIDIA Broadcast)",
    "Headset (R-Phonak hearing aid)",
    "Microphone (Umik-1)",
]

offered = listed(MIXED)
check("only connected devices are offered", names(offered) == REAL, f"got {names(offered)}")

# The app persists these numbers and the backend opens streams by them. Renumbering the
# filtered list 0,1,2 would still look correct in a picker while silently moving every saved
# microphone, which is the failure this app can least afford.
check(
    "PortAudio indices survive the filter",
    [d["index"] for d in offered] == [7, 8, 9],
    f"got {[d['index'] for d in offered]}",
)

out_only = listed([dev("Speakers", WASAPI, inputs=0), dev("Mic", WASAPI)])
check("output-only devices are never offered", names(out_only) == ["Mic"], f"got {names(out_only)}")

# Documents the one way this filter can hurt someone. An interface that publishes no WASAPI
# endpoint disappears from the picker even though it may be the microphone the user wants.
# Accepted deliberately: it did not occur on any machine measured, and the alternative is
# showing every stale entry to everyone. The C# side already refuses to substitute a
# different device silently, so the failure is visible rather than quiet. If this is ever
# reported from the wild, this is the test to change.
legacy_only = listed([dev("Fancy Interface", WDMKS), dev("Microphone (Umik-1)", WASAPI)])
check(
    "a device seen only through a legacy API is dropped",
    names(legacy_only) == ["Microphone (Umik-1)"],
    f"got {names(legacy_only)}",
)

# Linux and macOS have no WASAPI, and neither does a Windows box whose backend failed to
# start. A short list of real devices is the goal, but an empty picker is worse than a
# cluttered one: it leaves a user with no way to start captioning at all.
no_wasapi = listed([dev("Built-in Mic", MME), dev("USB Mic", DSOUND)])
check(
    "everything is offered when WASAPI is absent",
    names(no_wasapi) == ["Built-in Mic", "USB Mic"],
    f"got {names(no_wasapi)}",
)

check("no devices at all is not an error", listed([]) == [], "expected an empty list")

# The count is the only trace a dropped device leaves, and names must not be in it:
# backend.log is the file users are asked to send when something breaks, and
# "Headset (R-Phonak hearing aid)" in it would disclose that someone wears a hearing aid.
buf = io.StringIO()
with portaudio(MIXED), contextlib.redirect_stdout(buf):
    audio.list_input_devices()
logged = buf.getvalue()
check(
    "the hidden device count is logged",
    "3 connected" in logged and "6 hidden" in logged,
    f"got {logged!r}",
)
leaked = [d["name"] for d in MIXED if d["name"] in logged]
check("no device name reaches the log", not leaked, f"leaked {leaked}")

buf = io.StringIO()
with portaudio([dev("Microphone (Umik-1)", WASAPI)]), contextlib.redirect_stdout(buf):
    audio.list_input_devices()
check("nothing is logged when nothing is hidden", buf.getvalue() == "", f"got {buf.getvalue()!r}")

# PortAudio's global default is an MME index, which never survives the filter. Taking it
# verbatim would print a listing with no default marked at all.
with portaudio(MIXED), contextlib.redirect_stdout(io.StringIO()):
    marked = audio._default_input_index(audio.list_input_devices())
check("the default marker uses the WASAPI default", marked == 7, f"got {marked}")

with portaudio([dev("Built-in Mic", MME), dev("USB Mic", DSOUND)]), contextlib.redirect_stdout(
    io.StringIO()
):
    marked = audio._default_input_index(audio.list_input_devices())
check("the default marker falls back when the list was not filtered", marked == 1, f"got {marked}")

# Windows can genuinely have no default capture device. Saying so beats promoting an
# arbitrary microphone to "default", and beats returning an index from a host API that was
# filtered out, which would silently select nothing in any caller that pre-selects by it.
apis = [dict(a) for a in HOST_APIS]
apis[WASAPI]["default_input_device"] = -1
with portaudio(MIXED, host_apis=apis), contextlib.redirect_stdout(io.StringIO()):
    marked = audio._default_input_index(audio.list_input_devices())
check("no default is reported as none, not as a filtered-out device", marked == -1, f"got {marked}")

apis[WASAPI]["default_input_device"] = None
with portaudio(MIXED, host_apis=apis), contextlib.redirect_stdout(io.StringIO()):
    marked = audio._default_input_index(audio.list_input_devices())
check("a null host API default is handled", marked == -1, f"got {marked}")

# The invariant the callers rely on, asserted over every fixture in this file rather than
# argued once. A returned index that is not on offer would pre-select nothing.
for label, fixture, host_apis in [
    ("mixed", MIXED, HOST_APIS),
    ("no WASAPI", [dev("Built-in Mic", MME), dev("USB Mic", DSOUND)], HOST_APIS),
    ("empty", [], HOST_APIS),
    ("WASAPI has no default", MIXED, apis),
    ("single device", [dev("Microphone (Umik-1)", WASAPI)], HOST_APIS),
]:
    with portaudio(fixture, host_apis=host_apis), contextlib.redirect_stdout(io.StringIO()):
        shown = audio.list_input_devices()
        marked = audio._default_input_index(shown)
    check(
        f"the marked default is on offer or absent ({label})",
        marked == -1 or marked in {d["index"] for d in shown},
        f"marked {marked}, offered {[d['index'] for d in shown]}",
    )

print()
if failures:
    print(f"{len(failures)} FAILED")
    for f in failures:
        print(f"  {f}")
    raise SystemExit(1)

print("ALL PASS")
print(f"({len(MIXED)} device enumeration narrowed to {len(REAL)} connected)")
