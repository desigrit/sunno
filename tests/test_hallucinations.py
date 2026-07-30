"""Guards the Whisper hallucination filter.

Standalone script, like tests/test_biquad.py - run it directly:

    python tests\\test_hallucinations.py

Needs no model and no GPU: it exercises the pattern matching only, which is the part that
can silently over- or under-match as phrases are added.

The failure this protects against is asymmetric. Letting boilerplate through puts sentences
nobody said into a transcript someone is relying on to follow a conversation; matching too
broadly deletes real speech. Both cases are covered below.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.asr import _HALLUCINATION_PATTERNS  # noqa: E402
from server.config import Settings  # noqa: E402

# Caption credits Whisper reproduces over near-silence, having been trained on subtitled
# video. The first two were observed in this app's own transcript during a live session.
SHOULD_BLOCK = [
    "Subtitling by SUBS Hamburg",
    "Subtitles by the Amara.org community",
    "subtitles by the amara.org community",
    "Transcription by CastingWords",
    "Transcript by ESO, translated by —",
    "Translated by Releaser",
    "Subs by www.zeoranger.co.uk",
    "www.mooji.org",
    "Subscribe to my channel",
    "Please subscribe to our channel",
]

# Ordinary speech that happens to mention the same words. Deleting any of these would be a
# worse bug than the one the filter exists to fix.
SHOULD_PASS = [
    "Can you turn the subtitles on for this film?",
    "I was transcribing the meeting notes yesterday",
    "She subscribed to three newsletters",
    "The translation was done well",
    "I'll transcribe it later",
    "Did you subscribe yet?",
    "The quick brown fox jumps over the lazy dog",
    "We are validating that the trimmed CUDA payload produces identical transcripts",
    "Let's put subtitles on the video before we ship it",
]


def main() -> int:
    failures = []

    for text in SHOULD_BLOCK:
        if not _HALLUCINATION_PATTERNS.search(text.lower()):
            failures.append(f"should have been blocked but was not: {text!r}")

    for text in SHOULD_PASS:
        if _HALLUCINATION_PATTERNS.search(text.lower()):
            failures.append(f"real speech was wrongly blocked: {text!r}")

    # The exact-match list is applied separately, lower-cased and stripped, so entries must
    # already be in that form or they can never match.
    settings = Settings()
    for phrase in settings.hallucinations:
        if phrase != phrase.lower().strip():
            failures.append(f"hallucinations entry is not normalised: {phrase!r}")

    # Thresholds must stay in the range where they mean anything.
    if not 0.0 < settings.drop_no_speech_above <= 1.0:
        failures.append(f"drop_no_speech_above out of range: {settings.drop_no_speech_above}")
    if not 0.0 < settings.low_confidence_below < 1.0:
        failures.append(f"low_confidence_below out of range: {settings.low_confidence_below}")

    checked = len(SHOULD_BLOCK) + len(SHOULD_PASS) + len(settings.hallucinations) + 2
    if failures:
        print(f"\n{len(failures)} FAILURE(S) out of {checked} checks\n")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nALL PASS")
    print(f"({len(SHOULD_BLOCK)} blocked, {len(SHOULD_PASS)} passed through, "
          f"{len(settings.hallucinations)} list entries, 2 thresholds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
