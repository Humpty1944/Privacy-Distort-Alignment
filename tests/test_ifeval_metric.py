import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.ifeval_eval import IFEvalInstance
from src.ifeval_metric import IFEvalMetricComputer


class TestIFEvalMetricComputer(unittest.TestCase):
    def test_generation_scores_hypotheses(self):
        instances = [
            IFEvalInstance("p1", ["punctuation:no_comma"], [{}]),
            IFEvalInstance("p2", ["keywords:existence"], [{"keywords": ["apple"]}]),
        ]
        out = IFEvalMetricComputer(instances).compute(
            "generation",
            {"hypotheses": ["no commas here", "an apple a day"], "references": ["", ""]},
        )
        self.assertAlmostEqual(out["prompt_level_strict_acc"], 1.0)
        self.assertEqual(out["n_prompts"], 2)
        self.assertEqual(out["n_instructions"], 2)

    def test_length_mismatch_raises(self):
        mc = IFEvalMetricComputer([IFEvalInstance("p", ["punctuation:no_comma"], [{}])])
        with self.assertRaises(ValueError):
            mc.compute("generation", {"hypotheses": ["a", "b"], "references": ["", ""]})

    def test_rejects_non_generation_task(self):
        with self.assertRaises(ValueError):
            IFEvalMetricComputer([]).compute("classification", {"predictions": [], "labels": []})


if __name__ == "__main__":
    unittest.main(verbosity=2)
