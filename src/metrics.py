from __future__ import annotations

import csv
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional
import pandas as pd


class MetricComputer(ABC):
    """
    task: "classification" or "generation"
    eval_output:
          {"predictions": [...], "labels": [...]} for classification
          {"hypotheses": [...], "references": [...]} for generation

    Returns a dict of metrics
    """

    @abstractmethod
    def compute(self, task: str, eval_output: Dict[str, Any]) -> Dict[str, Any]: ...


@dataclass
class ExperimentRecord:
    model: str
    training_type: str  #  DP Full SFT/Non-private Full SFT/etc
    epsilon: float
    delta: Optional[float]
    sigma: Optional[float]
    decoding_temperature: Optional[float] = None
    runtime_seconds: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row.pop("extra")
        row.update(self.extra)
        return row


def apply_metrics(record: ExperimentRecord, metrics: Dict[str, Any]):
    record.extra.update(metrics)


class MetricsStore:

    def __init__(self):
        self.records: list[ExperimentRecord] = []

    def add(self, record: ExperimentRecord):
        self.records.append(record)

    def to_dicts(self):
        return [r.as_row() for r in self.records]

    def to_dataframe(self):
        return pd.DataFrame(self.to_dicts())

    def to_csv(self, path):
        rows = self.to_dicts()
        if not rows:
            return
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
