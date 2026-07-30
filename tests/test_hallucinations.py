"""Guards the Whisper hallucination filter.

Standalone script, like tests/test_biquad.py - run it directly:

    python tests\\test_hallucinations.py

Needs no model and no GPU: it exercises the text classification only, which is the part that
can silently over- or under-match as phrases are added.

The failure this protects against is asymmetric, and the more dangerous direction is the one
that is easy to miss. Letting boilerplate through puts sentences nobody said into a transcript
someone is relying on to follow a conversation. Deleting real speech is worse: the user cannot
tell it happened. An earlier version of this filter matched "translated by" anywhere in a
segment and dropped the whole segment, so "The book was translated by Tolkien himself"
vanished silently. The SHOULD_PASS list below therefore deliberately includes the "<word> by"
constructions that an earlier, narrower test avoided.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.asr import _looks_like_caption_credit  # noqa: E402
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
    "  Subtitles by the Amara.org community  ",
    "\u201cSubtitling by SUBS Hamburg\u201d",
    # Credits that slipped through an earlier, stricter-looking version of this filter.
    "Please subscribe to our channel and hit the bell now.",
    "\u266a Subtitles by NanoStudio \u266a",
    "Thanks for watching! Subtitles by NanoStudio",
    "Subtitles by Steamteam",
]

# Ordinary speech. Deleting any of these would be a worse bug than the one the filter exists
# to fix, because it is invisible to the person reading the transcript.
SHOULD_PASS = [
    # The class that a naive "contains" filter destroys.
    "The poem was translated by my grandmother",
    "The book was translated by Tolkien himself",
    "The subtitles by that studio were really good",
    "My cousin does transcription by hand for the courts",
    "This edition was translated by someone who really understood the original",
    "I think the transcript by the court reporter had errors in it",
    "Translated by my grandmother, the poem finally made sense to me",
    # Short sentences that OPEN with a credit phrase but attribute to ordinary words rather
    # than a name — the collocations a start-anchored filter alone would eat.
    "Subtitles by default are off.",
    "Translated by my sister.",
    "Transcription by hand takes forever.",
    "Subtitles by themselves don't help much.",
    "Subscribe to that podcast, it's really good.",
    "Subscribe to whichever plan works for you.",
    "Translated by the court reporter, apparently.",
    # Plain speech mentioning the same subject matter.
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
        if not _looks_like_caption_credit(text):
            failures.append(f"boilerplate was not blocked: {text!r}")

    for text in SHOULD_PASS:
        if _looks_like_caption_credit(text):
            failures.append(f"REAL SPEECH WAS DELETED: {text!r}")

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
    print(f"({len(SHOULD_BLOCK)} blocked, {len(SHOULD_PASS)} real sentences kept, "
          f"{len(settings.hallucinations)} list entries, 2 thresholds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
