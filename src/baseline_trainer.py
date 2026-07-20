from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn

from dp_finetune.training_configs import NonPrivateConfig


from .privacy_accountant import PrivacyReport, non_private_report
from .user_level_dataset import UserLevelDataset
from .dp_trainer import _compute_user_gradient_batched


@dataclass
class NonPrivateResult:
    privacy_report: PrivacyReport
    loss_curve: List[float]


class NonPrivateTrainer:
    """Same model, same user-batched pipeline, no DP mechanism at all."""

    def __init__(self, model: nn.Module, dataset: UserLevelDataset, config: NonPrivateConfig):
        self.model = model.to(config.device)
        self.dataset = dataset
        self.config = config
        self.rng = np.random.default_rng()

    def train(self) -> NonPrivateResult:
        cfg = self.config
        loss_curve: List[float] = []

        collator = cfg.collator or _compute_user_gradient_batched.__globals__["_default_hf_user_collator"]

        for t in range(1, cfg.max_steps + 1):
            user_batch = self.dataset.sample_users(cfg.user_batch_size, self.rng)
            while len(user_batch) == 0:
                user_batch = self.dataset.sample_users(cfg.user_batch_size, self.rng)

            summed = None
            step_loss = 0.0
            for uid in user_batch:
                k_i = cfg.records_per_user.get(uid, cfg.default_k)
                records = self.dataset.sample_user_records(uid, k_i, self.rng)
                grads, loss_val = _compute_user_gradient_batched(
                    model = self.model,
                    collator = collator,
                    records = records,
                    device = cfg.device,
                    autocast_dtype=None,
                    loss_fn = cfg.loss_fn
                )
                if summed is None:
                    summed = [g.clone() for g in grads]
                else:
                    for s, g in zip(summed, grads):
                        s += g
                step_loss += loss_val

            g_bar = [s / cfg.user_batch_size for s in summed]

            with torch.no_grad():
                for p, g in zip(self.model.parameters(), g_bar):
                    p.sub_(cfg.learning_rate * g)

            loss_curve.append(step_loss / len(user_batch))

            if t % cfg.log_every == 0 or t == cfg.max_steps:
                print(
                    f"[Non-private] step {t}/{cfg.max_steps} "
                    f"loss={loss_curve[-1]:.4f} (epsilon=inf)"
                )

        return NonPrivateResult(
            privacy_report=non_private_report(steps_taken=cfg.max_steps),
            loss_curve=loss_curve,
        )
