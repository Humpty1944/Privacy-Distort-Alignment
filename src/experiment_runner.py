from __future__ import annotations

from typing import Callable, Optional

from torch import nn

from src.experiment_configs import ExperimentConfig
from src.training_configs import DPConfig

from .eval_data import evaluate_classification, evaluate_generation
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
    """
    model = model_factory()

    with Timer() as timer:
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

    privacy_report = result.privacy_report # metrics from training

    record = ExperimentRecord(
        model=config.model_name,
        training_type=config.training_type,
        epsilon=privacy_report.achieved_epsilon,
        delta=privacy_report.delta,
        sigma=privacy_report.sigma,
        runtime_seconds=timer.elapsed,
        extra={"target_epsilon": target_epsilon},
    )

    if config.task == "classification": # metrics from eval
        eval_out = evaluate_classification(
            model,
            eval_dataloader,
            device=config.device,
            temperature=config.decoding.classification_temperature,
        )#return outputs

        apply_metrics(record, config.metric_computer.compute(config.task, eval_out))
    elif config.task == "generation":
        gen_out = evaluate_generation(
            model,
            eval_dataloader,
            config.decoding,
            device=config.device,
            decode_fn=decode_fn,
        )#return outputs
        
        # Report metrics per temperature as separate rows
        base_record = record
        rows = []
        for temperature, io in gen_out.items():
            row = ExperimentRecord(
                **{**base_record.__dict__, "extra": dict(base_record.extra)}
            )
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
