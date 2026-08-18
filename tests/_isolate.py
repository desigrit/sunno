"""Keep the tests out of the real profile. Import this before anything under ``server``.

The suite was caught writing to live app data. ``hardware.cpu_score()`` caches its
measurement, so running the tests baked a score measured under test load into the state a
real launch then reads to quote model latencies on the first-run screen. A test that changes
what the product shows the user is not a test.

This lives in its own module rather than in one test file because the isolation has to
outlive whoever wrote it. It was originally set in the preamble of ``test_stream_engine.py``,
which worked only because that happened to be the single script importing
``server.hardware``; the next test to touch hardware state would have written to the real
profile with nothing failing. Importing this module is the one line a new test needs.

Usage, before any ``server`` import::

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tests._isolate  # noqa: F401

Two things it does NOT do. It does not redirect the frontend, which builds its
LocalApplicationData paths in C# and knows nothing about this variable. And it does not give
a test access to models the user has downloaded, since the temp directory is empty; checks
that depend on a real download should skip rather than fail.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

DATA_DIR = tempfile.mkdtemp(prefix="sunno-tests-")
os.environ["Sunno_DATA_DIR"] = DATA_DIR


@atexit.register
def _cleanup() -> None:
    """Remove the directory on exit.

    Registered rather than left to the operating system because the first version used a
    bare ``mkdtemp`` and leaked one directory per run, several of them holding a cached
    hardware.json. Trading a write to the user's profile for an unbounded pile of temp
    directories is not a fix. ``ignore_errors`` because a failure to tidy up must never turn
    a passing suite red.
    """
    shutil.rmtree(DATA_DIR, ignore_errors=True)
