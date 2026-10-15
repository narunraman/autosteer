"""Metadata lookups: values with quotes in them, and the closest instruction match."""
import unittest

import pandas as pd

from autosteer import datasets as sd


def instructions(*rows):
    return pd.DataFrame([dict(zip(("type", "difficulty_level", "domain", "tags", "instructions"), r)) for r in rows])


class MetadataMaskTests(unittest.TestCase):
    def test_text_with_quotes_matches_instead_of_crashing(self):
        df = instructions(("t", 1, "children's toys", "x", "a"), ("t", 1, 'the "best" shop', "x", "b"), ("t", 1, "other", "x", "c"))
        self.assertEqual(df[sd.metadata_mask(df, {"domain": "children's toys"})]["instructions"].tolist(), ["a"])
        self.assertEqual(df[sd.metadata_mask(df, {"domain": 'the "best" shop'})]["instructions"].tolist(), ["b"])

    def test_difficulty_compares_as_a_number_and_lists_match_any(self):
        df = instructions(("t", 1, "d", "x", "a"), ("t", 2, "d", "x", "b"), ("u", 3, "d", "x", "c"))
        self.assertEqual(df[sd.metadata_mask(df, {"difficulty_level": "2"})]["instructions"].tolist(), ["b"])
        self.assertEqual(df[sd.metadata_mask(df, [{"type": "u"}, {"difficulty_level": 1}])]["instructions"].tolist(), ["a", "c"])


class GetInstructionsTests(unittest.TestCase):
    def test_exact_match(self):
        df = instructions(("t", 1, "d", "x", "exact"), ("t", 1, "e", "x", "other"))
        got = sd.get_instructions(df, {"type": "t", "difficulty_level": 1, "domain": "d", "tags": "x"})
        self.assertEqual(got["instructions"].tolist(), ["exact"])

    def test_the_closest_match_wins_over_a_looser_one(self):
        # no exact match; one row shares type, difficulty and domain (only tags differ), another only tags
        df = instructions(("u", 2, "other", "x", "shares only tags"), ("t", 1, "d", "y", "differs only in tags"))
        got = sd.get_instructions(df, {"type": "t", "difficulty_level": 1, "domain": "d", "tags": "x"})
        self.assertEqual(got["instructions"].tolist(), ["differs only in tags"])

    def test_the_least_important_key_is_left_out_first(self):
        # each row differs from the target in one key: tags is left out before domain, domain before type
        df = instructions(("t", 1, "e", "x", "differs in domain"), ("t", 1, "d", "y", "differs in tags"), ("u", 1, "d", "x", "differs in type"))
        got = sd.get_instructions(df, {"type": "t", "difficulty_level": 1, "domain": "d", "tags": "x"})
        self.assertEqual(got["instructions"].tolist(), ["differs in tags"])

    def test_rows_without_instructions_are_skipped(self):
        df = instructions(("t", 1, "d", "x", None), ("t", 1, "d", "y", "has text"))
        got = sd.get_instructions(df, {"type": "t", "difficulty_level": 1, "domain": "d", "tags": "x"})
        self.assertEqual(got["instructions"].tolist(), ["has text"])


if __name__ == "__main__":
    unittest.main()
