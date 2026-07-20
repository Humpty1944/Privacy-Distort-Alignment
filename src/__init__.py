"""dp_finetune package root.

This package exposes a small public API surface from the package root while
avoiding eager imports of every submodule during package initialization.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict



__all__ = [
    "DPConfig",
    "DataPreprocessor",
    "FeatureEncoder",
    "IdentityTensorEncoder",
    "PreprocessingConfig",
    "RawRecord",
    "PrivacyAccountant",
    "PrivacyReport",
    "non_private_report",
    "DPResult",
    "DPTrainer",
    "UserLevelDataset",
    "NonPrivateConfig",
    "NonPrivateResult",
    "NonPrivateTrainer",
    "ExperimentConfig",
    "run_experiment",
    "DEFAULT_EPSILONS",
    "run_privacy_sweep",
    "DecodingConfig",
    "evaluate_classification",
    "evaluate_generation",
    "DefaultMetricComputer",
    "ExperimentRecord",
    "MetricComputer",
    "MetricsStore",
    "apply_metrics",
    "HFClassificationEncoder",
    "HFEncoderConfig",
    "HFGenerationEncoder",
    "HFWrapper",
    "hf_collator",
    "load_tokenizer",
    "make_model_factory",
]

_EXPORTS: Dict[str, str] = {
    "DPConfig": "dp_finetune.training_configs",
    "DataPreprocessor": "dp_finetune.data_preprocessing",
    "FeatureEncoder": "dp_finetune.data_preprocessing",
    "IdentityTensorEncoder": "dp_finetune.data_preprocessing",
    "PreprocessingConfig": "dp_finetune.data_preprocessing",
    "RawRecord": "dp_finetune.data_preprocessing",
    "PrivacyAccountant": "dp_finetune.privacy_accountant",
    "PrivacyReport": "dp_finetune.privacy_accountant",
    "non_private_report": "dp_finetune.privacy_accountant",
    "DPResult": "dp_finetune.dp_trainer",
    "DPTrainer": "dp_finetune.dp_trainer",
    "UserLevelDataset": "dp_finetune.user_level_dataset",
    "NonPrivateConfig": "dp_finetune.training_configs",
    "NonPrivateResult": "dp_finetune.baseline_trainer",
    "NonPrivateTrainer": "dp_finetune.baseline_trainer",
    "ExperimentConfig": "dp_finetune.experiment_configs",
    "run_experiment": "dp_finetune.experiment_runner",
    "DEFAULT_EPSILONS": "dp_finetune.privacy_sweep",
    "run_privacy_sweep": "dp_finetune.privacy_sweep",
    "DecodingConfig": "dp_finetune.experiment_configs",
    "evaluate_classification": "dp_finetune.decoder_eval",
    "evaluate_generation": "dp_finetune.decoder_eval",
    "DefaultMetricComputer": "dp_finetune.metrics",
    "ExperimentRecord": "dp_finetune.metrics",
    "MetricComputer": "dp_finetune.metrics",
    "MetricsStore": "dp_finetune.metrics",
    "apply_metrics": "dp_finetune.metrics",
    "HFClassificationEncoder": "dp_finetune.hf_integration",
    "HFEncoderConfig": "dp_finetune.hf_integration",
    "HFGenerationEncoder": "dp_finetune.encoders",
    "HFWrapper": "dp_finetune.hf_integration",
    "hf_collator": "dp_finetune.hf_integration",
    "load_tokenizer": "dp_finetune.hf_integration",
    "make_model_factory": "dp_finetune.hf_integration",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
