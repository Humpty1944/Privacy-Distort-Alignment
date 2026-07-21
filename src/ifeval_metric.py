from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .ifeval_checks import CHECKERS
from .ifeval_eval import IFEvalInstance, evaluate_ifeval


class IFEvalMetricComputer:
    def __init__(self, instances: Sequence[IFEvalInstance],
                 checkers: Optional[Dict[str, Any]] = None):
        self.instances = list(instances)
        self.checkers = CHECKERS if checkers is None else checkers

    def compute(self, task: str, eval_output: Dict[str, Any]) -> Dict[str, Any]:
        if task != "generation":
            raise ValueError(f"IFEvalMetricComputer supports task 'generation', got {task!r}")
        hypotheses = eval_output["hypotheses"]
        if len(hypotheses) != len(self.instances):
            raise ValueError(
                f"hypotheses ({len(hypotheses)}) != instances ({len(self.instances)})"
            )
        return evaluate_ifeval(list(zip(self.instances, hypotheses)), self.checkers)
