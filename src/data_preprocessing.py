"""
Data preprocessing pipeline

Steps to use it:
    1. Implement FeatureEncoder
    2. Wrap your raw source into a list of RawRecord's (user_id, input,
       target, split)
    3. DataPreprocessor(encoder).build(records) -> (UserLevelDataset, DataLoader).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from src.experiment_configs import ExperimentConfig

from .user_level_dataset import UserLevelDataset


class FeatureEncoder(ABC):
    """
    Implement this per model/dataset
    """

    @abstractmethod
    def encode_input(self, raw_input: Any): ...

    @abstractmethod
    def encode_target(self, raw_target: Any, split: str) -> Any:
        """
        Raw label/target -> whatever the rest of the pipeline needs for
        this split:
          - split == "train": must return a training-ready tensor
          - split == "eval": for classification, a tensor label; for
            generation raw reference text or tokens
        """
        ...


class IdentityTensorEncoder(FeatureEncoder):
    """Reference encoder"""

    def encode_input(self, raw_input):
        return torch.as_tensor(raw_input, dtype=torch.float32)

    def encode_target(self, raw_target, split):
        return torch.as_tensor(raw_target, dtype=torch.long)


@dataclass
class RawRecord:
    user_id: str
    input: Any
    target: Any
    split: str = "train"  # "train" or "eval"


class SimpleDataset(Dataset):
    """Dataset wrapper so eval batches can mix tensors with plain
    objects and still use DataLoader"""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class DataPreprocessor:
    def __init__(
        self,
        encoder: FeatureEncoder,
        config: ExperimentConfig,
    ):
        self.encoder = encoder
        self.config = config

    def build(self, raw_records: Sequence[RawRecord]):
        train_records = [r for r in raw_records if r.split == "train"]
        eval_records = [r for r in raw_records if r.split == "eval"]
        if not train_records:
            raise ValueError("No records with split='train' found.")
        if not eval_records:
            raise ValueError("No records with split='eval' found.")

        base_dataset, user_indices = self._build_user_level(train_records)
        eval_dataloader = self._build_eval_dataloader(eval_records)
        return UserLevelDataset(base_dataset, user_indices), eval_dataloader

    def _build_user_level(self, records: Sequence[RawRecord]):
        base_dataset = []
        user_indices = {}

        for r in records:
            x = self.encoder.encode_input(r.input)
            y = self.encoder.encode_target(r.target, split="train")
            base_dataset.append((x, y))
            user_indices.setdefault(r.user_id, []).append(len(base_dataset) - 1)

        user_indices = {
            uid: (
                idxs[: self.config.max_records_per_user]
                if self.config.max_records_per_user
                else idxs
            )
            for uid, idxs in user_indices.items()
            if len(idxs) >= self.config.min_records_per_user
        }
        if not user_indices:
            raise ValueError(
                "No users met min_records_per_user "
                f"({self.config.min_records_per_user}) after preprocessing."
            )
        return base_dataset, user_indices

    def _build_eval_dataloader(self, records: Sequence[RawRecord]):
        items = [
            (
                self.encoder.encode_input(r.input),
                self.encoder.encode_target(r.target, split="eval"),
            )
            for r in records
        ]
        return DataLoader(
            SimpleDataset(items),
            batch_size=self.config.eval_batch_size,
            shuffle=False,
        )
