from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from torch import nn


@dataclass
class LoRAConfig:
    enabled: bool = False

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05

    bias: str = "none"

    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    )


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
        default_factory=lambda: __import__(
            "dp_finetune.metrics", fromlist=["DefaultMetricComputer"]
        ).DefaultMetricComputer()
    )
    seed: int = 42
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    eval_batch_size: int = 32
    min_records_per_user: int = 1
    max_records_per_user: Optional[int] = None  # None = no cap

    def __post_init__(self):
        numeric_fields = [
            "learning_rate",
            "weight_decay",
            "clip_norm",
            "warmup_ratio",
            "delta",
            "max_length",
            "default_k",
            "max_steps",
            "eval_batch_size",
            "min_records_per_user"
        ]

        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, float(value))
