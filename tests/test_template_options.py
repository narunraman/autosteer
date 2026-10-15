"""Models write option text for seed candidates only where the filler reads it (five elements)."""
import unittest

from autosteer.services.generated_seed_service import (
    _normalize_generated_options, _without_empty_options, build_target_instructions, uses_template_options,
)

SEED = {"question_1": "Which bundle? {utility_function}", "options_1": ["{num_x} apples and {num_y} oranges", "{num_x} pears"],
        "answer_1": "{num_x} apples and {num_y} oranges", "instructions": "Please give an MCQ example testing utility. More.",
        "metadata": {"domain": "base", "type": "t", "difficulty_level": 1, "tags": ""}}
TARGET = {"domain": "base", "type": "t", "difficulty_level": 1, "tags": ""}


class TemplateOptionTests(unittest.TestCase):
    def test_only_the_five(self):
        self.assertTrue(uses_template_options("completeness"))
        self.assertFalse(uses_template_options("micro_utility"))

    def test_no_options_asked_when_the_filler_writes_them(self):
        # a seed without options shows none in the prompt's example JSON (so the model is not asked for empty ones)
        shown = _without_empty_options(_normalize_generated_options(SEED, False))
        self.assertFalse([k for k in shown if k.startswith(("options_", "answer_"))])
        self.assertIn("options_1", _without_empty_options(SEED))


if __name__ == "__main__":
    unittest.main()
