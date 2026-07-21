from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class BenchmarkResult:

    benchmark: str

    metrics: Dict[str, float]

    samples: int


@dataclass
class EvaluationSummary:

    truthfulness: float

    hallucination_rate: float

    instruction_following: float

    helpfulness: float

    overall_score: float

    def to_dict(self):
        return asdict(self)