"""
Module 0 — Data preprocessing pipeline
=========================================

Model-agnostic boundary that turns raw, per-user records into the two
objects every trainer/evaluator in this package consumes:

    UserLevelDataset   -> Module 2 (DPTrainer) / Module 3 (NonPrivateTrainer)
    eval DataLoader    -> Module 6 (decoder_eval)

This is shared across all models — classification or generation, tabular
or text.

Contract for a new model/dataset:
    1. Implement `FeatureEncoder` (or reuse `IdentityTensorEncoder` for
       already-numeric data).
    2. Wrap your raw source into a list of `RawRecord`s (user_id, input,
       target, split).
    3. `DataPreprocessor(encoder).build(records)` -> (UserLevelDataset, DataLoader).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .user_level_dataset import UserLevelDataset


# ---------------------------------------------------------------------- #
# The one pluggable, model-specific piece
# ---------------------------------------------------------------------- #
class FeatureEncoder(ABC):
    """
    Implement this per model/dataset
    """

    @abstractmethod
    def encode_input(self, raw_input: Any) -> torch.Tensor:
        """Raw field(s) -> a single, unbatched, model-ready input tensor"""
        ...

    @abstractmethod
    def encode_target(self, raw_target: Any, split: str) -> Any:
        """
        Raw label/target -> whatever the rest of the pipeline needs for
        this split:
          - split == "train": must return a training-ready tensor
          - split == "eval": for classification, a tensor label; for
            generation
        """
        ...


class IdentityTensorEncoder(FeatureEncoder):
    """Reference encoder for already-numeric/tabular data."""

    def encode_input(self, raw_input: Any) -> torch.Tensor:
        return torch.as_tensor(raw_input, dtype=torch.float32)

    def encode_target(self, raw_target: Any, split: str) -> Any:
        return torch.as_tensor(raw_target, dtype=torch.long)


@dataclass
class RawRecord:
    user_id: str
    input: Any
    target: Any
    split: str = "train"        # "train" or "eval"


@dataclass
class PreprocessingConfig:
    eval_batch_size: int = 32
    min_records_per_user: int = 1
    max_records_per_user: Optional[int] = None   # None = no cap


class _ListDataset(Dataset):
    """Thin Dataset wrapper so eval batches can mix tensors with plain
    objects (e.g. generation reference strings) and still use DataLoader."""

    def __init__(self, items: List[Tuple[Any, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


class DataPreprocessor:
    """
    Usage:
        pre = DataPreprocessor(encoder=YourFeatureEncoder())
        train_dataset, eval_dataloader = pre.build(raw_records)
    """

    def __init__(
        self,
        encoder: FeatureEncoder,
        config: PreprocessingConfig = None,
    ):
        self.encoder = encoder
        self.config = config or PreprocessingConfig()

    def build(
        self, raw_records: Sequence[RawRecord]
    ) -> Tuple[UserLevelDataset, DataLoader]:
        train_records = [r for r in raw_records if r.split == "train"]
        eval_records = [r for r in raw_records if r.split == "eval"]
        if not train_records:
            raise ValueError("No records with split='train' found.")
        if not eval_records:
            raise ValueError("No records with split='eval' found.")

        base_dataset, user_indices = self._build_user_level(train_records)
        eval_dataloader = self._build_eval_dataloader(eval_records)
        return UserLevelDataset(base_dataset, user_indices), eval_dataloader

    def _build_user_level(
        self, records: Sequence[RawRecord]
    ) -> Tuple[List[Tuple[torch.Tensor, Any]], Dict[str, List[int]]]:
        base_dataset: List[Tuple[torch.Tensor, Any]] = []
        user_indices: Dict[str, List[int]] = {}

        for r in records:
            x = self.encoder.encode_input(r.input)
            y = self.encoder.encode_target(r.target, split="train")
            base_dataset.append((x, y))
            user_indices.setdefault(r.user_id, []).append(len(base_dataset) - 1)

        cfg = self.config
        user_indices = {
            uid: (idxs[: cfg.max_records_per_user] if cfg.max_records_per_user else idxs)
            for uid, idxs in user_indices.items()
            if len(idxs) >= cfg.min_records_per_user
        }
        if not user_indices:
            raise ValueError(
                "No users met min_records_per_user "
                f"({cfg.min_records_per_user}) after preprocessing."
            )
        return base_dataset, user_indices

    def _build_eval_dataloader(self, records: Sequence[RawRecord]) -> DataLoader:
        items = [
            (
                self.encoder.encode_input(r.input),
                self.encoder.encode_target(r.target, split="eval"),
            )
            for r in records
        ]
        return DataLoader(
            _ListDataset(items),
            batch_size=self.config.eval_batch_size,
            shuffle=False,
        )
