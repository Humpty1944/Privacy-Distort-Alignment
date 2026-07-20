from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

from torch import nn


@dataclass
class DecodingConfig:
    classification_temperature: float = 0.0
    generation_temperatures: Sequence[float] = (0.0, 0.7, 1.0)
    max_new_tokens: int = 64
    eos_token_id: Optional[int] = None


@dataclass
class ExperimentConfig:
    model_name: str
    training_type: str = "DP Full SFT"
    task: str = "classification"
    max_steps: int = 1000
    user_batch_size: int = 256
    clip_norm: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    delta: float = 1e-5
    default_k: int = 1
    max_length: int = 256
    records_per_user: Dict[str, int] = field(default_factory=dict)
    loss_fn: Callable = field(default_factory=nn.CrossEntropyLoss)
    device: str = "cpu"
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    log_every: int = 50
    collator: Optional[Callable] = None
    metric_computer: Any = field(
        default_factory=lambda: __import__("dp_finetune.metrics", fromlist=["DefaultMetricComputer"]).DefaultMetricComputer()
    )
    seed: int = 42
