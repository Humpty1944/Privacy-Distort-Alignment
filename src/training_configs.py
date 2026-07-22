from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from torch import nn


@dataclass
class DPConfig:
    max_steps: int  # T
    user_batch_size: int  # n
    clip_norm: float  # C
    learning_rate: float  # eta
    delta: float
    weight_decay: float
    warmup_ratio: float
    target_epsilon: float
    default_k: int = 1  # records-per-user
    records_per_user: Dict[str, int] = field(default_factory=dict)
    loss_fn: Callable = field(default_factory=nn.CrossEntropyLoss)
    device: str = "cpu"
    log_every: int = 1
    seed: int = 42
    gradient_checkpointing: bool = True
    mixed_precision: str = "bfloat16"  # "bfloat16" / "float16" / "fp32" / None
    collator: Optional[Callable] = None
    fused_optimizer: bool = True

