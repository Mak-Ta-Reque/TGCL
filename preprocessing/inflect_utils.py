#!/usr/bin/env python3
"""Guard around `inflect.engine().singular_noun()` for words it mis-singularizes.

inflect strips a trailing "s" whenever a word merely *ends* in "s", which
misfires on words that end in "s" but aren't a plural of some shorter word
("bus" -> "bu", "glass" -> "glas", "class" -> "clas", ...). Concept names
that hit this get silently corrupted downstream (crop filenames, feature
files, concept-bank entries), and any later string comparison against the
correct spelling (e.g. AU-AIR's "bus" vocab entry or eval ground truth)
breaks. Guard with a small denylist rather than trusting singular_noun()
on every word.
"""

# Words ending in "s" that are already singular, but that
# inflect.engine().singular_noun() incorrectly strips further (verified
# against inflect's actual behavior -- see tests/test_inflect_utils.py).
# Extend as new false positives are found in project vocabularies.
SINGULAR_NOUN_FALSE_POSITIVES = {
    "bus", "gas", "glass", "class", "atlas", "bias", "bonus", "campus",
    "canvas", "chorus", "circus", "corpus", "focus", "octopus", "plus",
    "virus",
}


def safe_singular_noun(word: str, engine) -> str:
    """`engine.singular_noun(word) or word`, but skip known false positives."""
    if word.lower() in SINGULAR_NOUN_FALSE_POSITIVES:
        return word
    return engine.singular_noun(word) or word
