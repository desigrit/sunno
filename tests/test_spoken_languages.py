import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.spoken_language import PhraseLanguages, SpokenLanguage, resolve


class LanguageTests(unittest.TestCase):
    def test_default_and_model_fallback_do_not_change_preference(self):
        for preference in ("hi", "auto", "fr"):
            for model in ("distil-large-v3", "moonshine-small", "stream-en", "windows-speech"):
                session = SpokenLanguage(model, preference, lambda event: None)
                self.assertEqual(session.choices, ("en",))
                self.assertEqual(session.choice().code, "en")
            self.assertEqual(SpokenLanguage("large-v3", preference, print).choice().code, preference)
        self.assertNotIn("yue", SpokenLanguage("base", "en", print).choices)
        self.assertIn("yue", SpokenLanguage("large-v3", "en", print).choices)
        self.assertEqual(resolve("ja", ("fr-CA", "en-US")), "en-US")
        self.assertEqual(resolve("fr", ("fr-CA","en-US")), "fr-CA")
        self.assertEqual(resolve("es", ("auto", "de")), "auto")

    def test_revisions_pause_and_short_foreign_phrase(self):
        states = []
        session = SpokenLanguage("base", "auto", states.append)
        policy = PhraseLanguages(session)
        seen = []
        def detect(audio):
            seen.append(len(audio))
            return {1: ("en", .99), 2: ("es", .98), 3: ("es", .51)}[round(float(audio.mean()))]
        engine = SimpleNamespace(detect_language=detect)
        def run(uid, frames):
            job = SimpleNamespace(choice=session.choice(), utterance_id=uid, is_final=False, voiced=frames)
            return policy.for_job(engine, job)
        english = np.ones(48000, np.float32)
        spanish = 2 * english
        self.assertIsNone(run(0, ()))
        self.assertEqual(run(1, (english,)), "en")
        self.assertEqual(run(2, (spanish,)), "en")
        for _ in range(5):
            self.assertEqual(run(2, (spanish,)), "en")
        self.assertEqual(len(seen), 2)
        self.assertEqual(run(3, (english,)), "en")
        self.assertEqual(run(4, (spanish,)), "en")
        self.assertEqual(run(4, (spanish, spanish)), "es")
        self.assertEqual([s["detected"] for s in states], ["en","es"])
        self.assertEqual(run(5, ()), "es")

    def test_selection_is_captured_until_next_phrase_and_stale_results_are_ignored(self):
        events = []
        session = SpokenLanguage("base","auto",events.append)
        old = session.choice()
        session.select("fr")
        policy = PhraseLanguages(session)
        job = SimpleNamespace(choice=old, utterance_id=10, is_final=True, voiced=(np.ones(48000),))
        self.assertEqual(policy.for_job(SimpleNamespace(detect_language=lambda _: ("de",1)),job), "de")
        self.assertIsNone(session.state()["detected"])
        for uid in (11,12):
            job = SimpleNamespace(choice=session.choice(), utterance_id=uid, is_final=True, voiced=())
            self.assertEqual(policy.for_job(None,job), "fr")
        self.assertEqual(events[-1]["accepted"],"fr")


if __name__ == "__main__":
    unittest.main()

