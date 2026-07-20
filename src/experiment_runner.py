from __future__ import annotations

import math
from typing import Callable, Dict, Optional

from torch import nn

from dp_finetune.experiment_configs import ExperimentConfig
from dp_finetune.training_configs import DPConfig

from .baseline_trainer import NonPrivateConfig, NonPrivateTrainer
from .decoder_eval import evaluate_classification, evaluate_generation
from .dp_trainer import DPTrainer
from .metrics import (
    ExperimentRecord,
    MetricsStore,
    Timer,
    apply_metrics,
)
from .user_level_dataset import UserLevelDataset



def run_experiment(
    model_factory: Callable[[], nn.Module],
    train_dataset: UserLevelDataset,
    eval_dataloader,
    target_epsilon: float,
    config: ExperimentConfig,
    metrics_store: Optional[MetricsStore] = None,
    decode_fn: Optional[Callable] = None,
) -> ExperimentRecord:
    """
    Trains and evaluates one epsilon
    model_factory() must return a new initialized model
    """
    model = model_factory()
    is_dp = not math.isinf(target_epsilon)
    training_type = config.training_type if is_dp else f"Non-private {config.training_type}"

    with Timer() as timer:
        if is_dp:
            dp_config = DPConfig(
                max_steps=config.max_steps,
                user_batch_size=config.user_batch_size,
                clip_norm=config.clip_norm,
                learning_rate=config.learning_rate,
                delta=config.delta,
                target_epsilon=target_epsilon,
                default_k=config.default_k,
                records_per_user=config.records_per_user,
                loss_fn=config.loss_fn,
                device=config.device,
                log_every=config.log_every,
                weight_decay=config.weight_decay,
                warmup_ratio=config.warmup_ratio,
                collator=config.collator,
            )
            trainer = DPTrainer(model, train_dataset, dp_config)
            result = trainer.train()
        else:
            np_config = NonPrivateConfig(
                max_steps=config.max_steps,
                user_batch_size=config.user_batch_size,
                learning_rate=config.learning_rate,
                default_k=config.default_k,
                records_per_user=config.records_per_user,
                loss_fn=config.loss_fn,
                device=config.device,
                log_every=config.log_every,
                collator=config.collator,
            )
            trainer = NonPrivateTrainer(model, train_dataset, np_config)
            result = trainer.train()

    privacy_report = result.privacy_report


    # decoder evaluation
    record = ExperimentRecord(
        model=config.model_name,
        training_type=training_type,
        dp=is_dp,
        epsilon=privacy_report.achieved_epsilon,
        delta=privacy_report.delta if is_dp else None,
        sigma=privacy_report.sigma if is_dp else None,
        runtime_seconds=timer.elapsed,
        extra={"target_epsilon": target_epsilon},
    )

    if config.task == "classification":
        eval_out = evaluate_classification(
            model,
            eval_dataloader,
            device=config.device,
            temperature=config.decoding.classification_temperature,
        )
        # this runner never computes a metric itself,
        # it only hands output to whatever MetricComputer 
        # supplied and merges the result back onto the row.
        apply_metrics(record, config.metric_computer.compute(config.task, eval_out))
    elif config.task == "generation":
        gen_out = evaluate_generation(
            model, eval_dataloader, config.decoding, device=config.device, decode_fn=decode_fn
        )
        # Report metrics per temperature as separate rows 
        base_record = record
        rows = []
        for temperature, io in gen_out.items():
            row = ExperimentRecord(**{**base_record.__dict__, "extra": dict(base_record.extra)})
            row.decoding_temperature = temperature
            apply_metrics(row, config.metric_computer.compute(config.task, io))
            rows.append(row)
            if metrics_store is not None:
                metrics_store.add(row)
        return rows  # list of ExperimentRecord, one per temperature
    else:
        raise ValueError(f"Unknown task: {config.task}")

    if metrics_store is not None:
        metrics_store.add(record)
    return record
