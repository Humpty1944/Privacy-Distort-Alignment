import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.ifeval_eval import (
    IFEvalInstance,
    evaluate_ifeval,
    follows_instruction,
    _loose_variants,
)
from src.ifeval_checks import CHECKERS


class TestLooseVariants(unittest.TestCase):
    def test_strips_wrapper_first_line(self):
        variants = _loose_variants("Sure, here:\napple banana")
        self.assertIn("apple banana", variants)
        self.assertIn("Sure, here:\napple banana", variants)

    def test_strips_markdown_emphasis(self):
        self.assertIn("bold", _loose_variants("**bold**"))

    def test_raw_kept_even_if_empty(self):
        self.assertEqual(_loose_variants(""), [""])


class TestFollowsInstruction(unittest.TestCase):
    def test_unknown_id_raises(self):
        with self.assertRaises(KeyError):
            follows_instruction("x", "does:not_exist", {}, CHECKERS)

    def test_loose_never_stricter_than_strict(self):
        r = "Sure, friend:\nno commas here"
        strict = follows_instruction(r, "punctuation:no_comma", {}, CHECKERS, loose=False)
        loose = follows_instruction(r, "punctuation:no_comma", {}, CHECKERS, loose=True)
        self.assertFalse(strict)
        self.assertTrue(loose)


class TestEvaluateIfeval(unittest.TestCase):
    def test_four_metrics_and_overall(self):
        records = [
            (IFEvalInstance("p1", ["punctuation:no_comma", "keywords:existence"],
                            [{}, {"keywords": ["apple"]}]), "apple banana cherry"),
            (IFEvalInstance("p2", ["punctuation:no_comma"], [{}]), "Sure, here:\napple banana"),
        ]
        m = evaluate_ifeval(records, CHECKERS)
        self.assertAlmostEqual(m["prompt_level_strict_acc"], 0.5)
        self.assertAlmostEqual(m["prompt_level_loose_acc"], 1.0)
        self.assertAlmostEqual(m["instruction_level_strict_acc"], 2 / 3)
        self.assertAlmostEqual(m["instruction_level_loose_acc"], 1.0)
        self.assertAlmostEqual(m["ifeval_overall"], (0.5 + 1.0 + 2 / 3 + 1.0) / 4)
        self.assertEqual(m["n_prompts"], 2)
        self.assertEqual(m["n_instructions"], 3)

    def test_empty_instruction_prompt_skipped(self):
        m = evaluate_ifeval([(IFEvalInstance("p", []), "whatever")], CHECKERS)
        self.assertEqual(m["n_prompts"], 0)
        self.assertTrue(math.isnan(m["prompt_level_strict_acc"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
