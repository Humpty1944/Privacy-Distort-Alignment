"""
Module 7 — Metrics
====================

Stores, per experiment:
  model, training type, DP (bool), epsilon, delta, sigma,
  accuracy, F1, BLEU, ROUGE, runtime

Classification metrics use scikit-learn if available (falls back to a
pure-Python implementation so the module has no hard dependency).
BLEU/ROUGE use sacrebleu / rouge-score if installed; otherwise those
columns are left as None with a warning, rather than failing the run.
"""

from __future__ import annotations

import csv
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import evaluate

try:
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
except Exception:
    rouge = None
    bleu = None
# ---------------------------------------------------------------------- #
# Generation metrics
# ---------------------------------------------------------------------- #
def compute_bleu(hypotheses: Sequence[str], references: Sequence[str]) -> Optional[float]:
    formatted_references = [[ref] for ref in references]
    
    results = bleu.compute(
        predictions=hypotheses,
        references=formatted_references,
    )
    
    return {"bleu": results["bleu"]}


def compute_rouge(hypotheses: Sequence[str], references: Sequence[str]) -> Optional[Dict[str, float]]:
    results = rouge.compute(
        predictions=hypotheses,
        references=references,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    return results


# ---------------------------------------------------------------------- #
# Pluggable metrics boundary
# ---------------------------------------------------------------------- #
# Anyone can return any class implementing compute() and
# nothing else in the pipeline changes
class MetricComputer(ABC):
    """
    task: "classification" or "generation"
    eval_output: 
          {"predictions": [...], "labels": [...]} for classification
          {"hypotheses": [...], "references": [...]} for generation

    Returns a dict of metric_name: value}
    """

    @abstractmethod
    def compute(self, task: str, eval_output: Dict[str, Any]) -> Dict[str, Any]:
        ...


class DefaultMetricComputer(MetricComputer):
    """
    This is only the placeholder
    """

    def compute(self, task: str, eval_output: Dict[str, Any]) -> Dict[str, Any]:
        if task == "classification":
            preds, labels = eval_output["predictions"], eval_output["labels"]
            return {
                "accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds),
                "precision": precision_score(labels, preds)
            }
        elif task == "generation":
            hyps, refs = eval_output["hypotheses"], eval_output["references"]
            result: Dict[str, Any] = {"bleu": compute_bleu(hyps, refs)}
            rouge = compute_rouge(hyps, refs)
            if rouge:
                result.update(rouge)
            return result
        raise ValueError(f"DefaultMetricComputer: unrecognized task '{task}'")


# ---------------------------------------------------------------------- #
# Results table
# ---------------------------------------------------------------------- #
@dataclass
class ExperimentRecord:
    model: str
    training_type: str          #  DP Full SFT/Non-private Full SFT/etc
    dp: bool
    epsilon: float
    delta: Optional[float]
    sigma: Optional[float]
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    bleu: Optional[float] = None
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougeL: Optional[float] = None
    decoding_temperature: Optional[float] = None
    runtime_seconds: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row.pop("extra")
        row.update(self.extra)
        return row


_KNOWN_RECORD_METRIC_FIELDS = {"accuracy", "f1", "precision", "recall", "bleu", "rouge1", "rouge2", "rougeL"}


def apply_metrics(record: ExperimentRecord, metrics: Dict[str, Any]) -> None:
    """Write a MetricComputer's output onto a record: known fields get
    set directly, anything else (a custom metric name) goes to `extra`
    so it still surfaces as a results-table column."""
    for key, value in metrics.items():
        if key in _KNOWN_RECORD_METRIC_FIELDS:
            setattr(record, key, value)
        else:
            record.extra[key] = value


class MetricsStore:

    def __init__(self):
        self.records: List[ExperimentRecord] = []

    def add(self, record: ExperimentRecord) -> None:
        self.records.append(record)

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [r.as_row() for r in self.records]

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.to_dicts())

    def to_csv(self, path: str) -> None:
        rows = self.to_dicts()
        if not rows:
            return
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


class Timer:
    """Small context manager for the `runtime_seconds` field."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
