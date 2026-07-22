from __future__ import annotations

from typing import Dict, List

import numpy as np


class UserLevelDataset:
    """
    Wraps a flat, indexable base dataset grouped by user id
    """

    def __init__(self, base_dataset, user_indices: Dict[str, List[int]]):
        self.base_dataset = base_dataset
        self.user_indices = user_indices
        self.user_ids = list(user_indices.keys())

    @property
    def num_users(self):
        return len(self.user_ids)

    def records_per_user(self):
        return {uid: len(idxs) for uid, idxs in self.user_indices.items()}

    def sample_users(self, expected_batch_size, rng: np.random.Generator):
        q = expected_batch_size / self.num_users
        mask = rng.random(self.num_users) < q
        return np.asarray(self.user_ids)[mask]

    def sample_user_records(self, user_id, k_i, rng: np.random.Generator):
        idxs = self.user_indices[user_id]
        if k_i >= len(idxs):
            chosen = idxs
        else:
            chosen = rng.choice(idxs, size=k_i, replace=False)
        return [self.base_dataset[i] for i in chosen]
