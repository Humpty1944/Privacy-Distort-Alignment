"""src package root.

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
    "DPConfig": "src.training_configs",
    "DataPreprocessor": "src.data_preprocessing",
    "FeatureEncoder": "src.data_preprocessing",
    "IdentityTensorEncoder": "src.data_preprocessing",
    "PreprocessingConfig": "src.data_preprocessing",
    "RawRecord": "src.data_preprocessing",
    "PrivacyAccountant": "src.privacy_accountant",
    "PrivacyReport": "src.privacy_accountant",
    "non_private_report": "src.privacy_accountant",
    "DPResult": "src.dp_trainer",
    "DPTrainer": "src.dp_trainer",
    "UserLevelDataset": "src.user_level_dataset",
    "NonPrivateConfig": "src.training_configs",
    "ExperimentConfig": "src.experiment_configs",
    "run_experiment": "src.experiment_runner",
    "DEFAULT_EPSILONS": "src.privacy_sweep",
    "run_privacy_sweep": "src.privacy_sweep",
    "DecodingConfig": "src.experiment_configs",
    "evaluate_classification": "src.eval_data",
    "evaluate_generation": "src.eval_data",
    "DefaultMetricComputer": "src.metrics",
    "ExperimentRecord": "src.metrics",
    "MetricComputer": "src.metrics",
    "MetricsStore": "src.metrics",
    "apply_metrics": "src.metrics",
    "HFClassificationEncoder": "src.hf_integration",
    "HFGenerationEncoder": "src.hf_integration",
    "HFEncoderConfig": "src.hf_integration",
    "HFWrapper": "src.hf_integration",
    "hf_collator": "src.hf_integration",
    "load_tokenizer": "src.hf_integration",
    "make_model_factory": "src.hf_integration",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
