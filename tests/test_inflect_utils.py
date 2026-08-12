"""Tests for preprocessing/inflect_utils.py"""

import os
import sys
import unittest

# Ensure preprocessing/ is importable
_preprocessing_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'preprocessing')
if _preprocessing_dir not in sys.path:
    sys.path.insert(0, _preprocessing_dir)

import inflect

from inflect_utils import safe_singular_noun, SINGULAR_NOUN_FALSE_POSITIVES


class TestSafeSingularNoun(unittest.TestCase):
    """Guard against inflect.singular_noun() stripping a trailing 's' from
    words that aren't actually plural (the "bus" -> "bu" bug)."""

    def setUp(self):
        self.engine = inflect.engine()

    def test_denylisted_words_are_left_alone(self):
        for word in SINGULAR_NOUN_FALSE_POSITIVES:
            self.assertEqual(safe_singular_noun(word, self.engine), word)

    def test_bus_specifically(self):
        # Regression check for the AU-AIR bug this guard was added for.
        self.assertEqual(self.engine.singular_noun("bus"), "bu")
        self.assertEqual(safe_singular_noun("bus", self.engine), "bus")

    def test_denylist_check_is_case_insensitive(self):
        self.assertEqual(safe_singular_noun("Bus", self.engine), "Bus")
        self.assertEqual(safe_singular_noun("BUS", self.engine), "BUS")

    def test_genuine_plurals_still_singularize(self):
        cases = {
            "buses": "bus",
            "cars": "car",
            "trucks": "truck",
            "trailers": "trailer",
            "motorbikes": "motorbike",
            "bicycles": "bicycle",
        }
        for plural, singular in cases.items():
            self.assertEqual(safe_singular_noun(plural, self.engine), singular)

    def test_already_singular_non_denylisted_word_is_unchanged(self):
        # inflect.singular_noun() returns False for words it can't singularize
        # further; safe_singular_noun() should fall back to the input word.
        for word in ("human", "van", "motorbike", "bicycle", "trailer"):
            self.assertEqual(safe_singular_noun(word, self.engine), word)


if __name__ == "__main__":
    unittest.main()
