
from dataclasses import dataclass
from typing import List

from src.data import (
    ProbeExample,
    load_truthfulqa_probes,
    load_ifeval_probes,
    load_alpaca_probes,
)


@dataclass
class BenchmarkBundle:
    truthfulqa: List[ProbeExample]
    ifeval: List[ProbeExample]
    alpaca: List[ProbeExample]


class BenchmarkLoader:

    @staticmethod
    def load_truthfulqa(**kwargs):
        return load_truthfulqa_probes(**kwargs)

    @staticmethod
    def load_ifeval(**kwargs):
        return load_ifeval_probes(**kwargs)

    @staticmethod
    def load_alpaca(**kwargs):
        return load_alpaca_probes(**kwargs)

    @classmethod
    def load_all(
        cls,
        n_truthfulqa=20,
        n_ifeval=15,
        n_alpaca=15,
        seed=0,
    ):
        return BenchmarkBundle(
            truthfulqa=cls.load_truthfulqa(
                n=n_truthfulqa,
                seed=seed,
            ),
            ifeval=cls.load_ifeval(
                n=n_ifeval,
                seed=seed,
            ),
            alpaca=cls.load_alpaca(
                n=n_alpaca,
                seed=seed,
            ),
        )
