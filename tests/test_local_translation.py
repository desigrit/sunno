import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.recorder import Recorder, _read_lines
from server.translation import TranslationQueue, LocalTranslator


class TranslationTests(unittest.TestCase):
    def test_original_timestamp_and_translation_are_recoverable(self):
        with tempfile.TemporaryDirectory() as root:
            rec = Recorder(Path(root))
            event = dict(id=3, started_at=rec.started_at+1, text="Bonjour", language="fr")
            rec.add_line(event)
            rec.add_translation(dict(event, type="translation", original="Bonjour", text="Olá", target_language="pt"))
            rec.detach()
            row, = _read_lines(rec.folder)
            self.assertEqual((row["text"],row["at"],row["language"]), ("Bonjour",1.0,"fr"))
            self.assertEqual(row["translation"], {"language":"pt","text":"Olá"})

    def test_partial_revisions_and_disabled_translation_do_not_replace_source(self):
        entered, resume, ready = threading.Event(), threading.Event(), threading.Event()
        events = []
        class Translator:
            def translate(self, text, source, target, current):
                entered.set()
                resume.wait(4)
                return f"{target} {text}" if current() else ""
        def emit(event):
            events.append(event)
            if event["type"] == "translation":
                ready.set()
        worker = TranslationQueue(emit, Translator)
        try:
            worker.select("pt")
            original = dict(id=1, started_at=55, language="fr", text="Salut")
            worker.submit(original)
            self.assertTrue(entered.wait(3))
            worker.submit(dict(original, text="Salut !"))
            resume.set()
            self.assertTrue(ready.wait(3))
            self.assertEqual([e["original"] for e in events if e["type"] == "translation"], ["Salut !"])
            worker.select("off")
            self.assertEqual(worker.state()["selected"], "off")
        finally:
            worker.close()

    @unittest.skipUnless("--real" in sys.argv or "SUNNO_TRANSLATION_TEST_MODEL" in __import__("os").environ,
                         "requires the local translation model")
    def test_real_french_to_portuguese(self):
        import os
        translator = LocalTranslator(os.environ.get("SUNNO_TRANSLATION_TEST_MODEL"))
        result = translator.translate("Bonjour, je voudrais un verre d'eau, s'il vous plait.", "fr", "pt")
        self.assertIn("água", result.lower())
        self.assertIn("copo", result.lower())


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])

