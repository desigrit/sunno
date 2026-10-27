import unittest

from bench.bench_engines import normalized_words, word_errors


class BenchmarkMetricTests(unittest.TestCase):
    def test_case_and_punctuation_do_not_count_as_word_errors(self):
        self.assertEqual(word_errors("Hello, world!", "HELLO WORLD."), (0, 2))

    def test_insertions_deletions_and_substitutions_count(self):
        self.assertEqual(word_errors("one two three", "one four five six"), (3, 3))
        self.assertEqual(word_errors("one two three", "one"), (2, 3))
        self.assertEqual(word_errors("one", "one two"), (1, 1))

    def test_empty_transcription_is_not_a_successful_decode(self):
        self.assertEqual(word_errors("please switch the microphone", ""), (4, 4))

    def test_smart_apostrophes_are_normalized(self):
        self.assertEqual(normalized_words("It’s Sunno."), ["it's", "sunno"])


if __name__ == "__main__":
    unittest.main()
