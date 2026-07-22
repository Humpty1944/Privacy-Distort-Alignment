"""
Automatically runs DP across the epsilon grid
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from torch import nn

from .experiment_runner import ExperimentConfig, run_experiment
from .metrics import MetricsStore
from .user_level_dataset import UserLevelDataset

DEFAULT_EPSILONS: Sequence[float] = (8, 4, 2)


def run_privacy_sweep(
    model_factory: Callable[[], nn.Module],
    train_dataset: UserLevelDataset,
    eval_dataloader,
    config: ExperimentConfig,
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
    decode_fn: Optional[Callable] = None,
) -> MetricsStore:
    store = MetricsStore()

    for target_epsilon in epsilons:
        print(f"\n===== Privacy sweep: target epsilon = {target_epsilon} =====")

        result = run_experiment(
            model_factory=model_factory,
            train_dataset=train_dataset,
            eval_dataloader=eval_dataloader,
            target_epsilon=target_epsilon,
            config=config,
            metrics_store=store,
            decode_fn=decode_fn,
        )

        rows = result if isinstance(result, list) else [result]
        for row in rows:
            print(
                f"epsilon={row.epsilon} sigma={row.sigma} "
                f"acc={row.accuracy} f1={row.f1} bleu={row.bleu} "
                f"runtime={row.runtime_seconds:.1f}s"
            )

    return store
