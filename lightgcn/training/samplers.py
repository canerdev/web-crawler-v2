"""
Sampling Utilities for Recommendation Training

Provides efficient negative sampling and batch generation.
"""

import torch
from typing import Dict, List
import numpy as np
from dataclasses import dataclass, field


@dataclass
class PairwiseSampler:
    """
    Efficient pairwise sampler for BPR training.

    Pre-generates negative samples for each epoch to improve throughput.
    """

    n_users: int
    n_items: int
    train_interactions: Dict[int, List[int]]
    batch_size: int = 1024
    n_neg: int = 1
    max_samples: int = 0
    shuffle: bool = True
    exclude_positive_negatives: bool = True

    # Computed attributes (set in __post_init__)
    users: List[int] = field(init=False)
    items: List[int] = field(init=False)
    n_samples: int = field(init=False)
    pair_keys_sorted: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # Flatten interactions with pre-allocated arrays (faster than append loop)
        total = sum(len(items) for items in self.train_interactions.values())

        self.users = np.empty(total, dtype=np.int64)
        self.items = np.empty(total, dtype=np.int64)

        idx = 0
        for u, item_list in self.train_interactions.items():
            n = len(item_list)
            self.users[idx:idx + n] = u
            self.items[idx:idx + n] = item_list
            idx += n

        self.n_samples = total

        # Encoded (user, item) keys for vectorized membership checks.
        # key = user * n_items + item
        pair_keys = self.users.astype(np.int64) * np.int64(self.n_items) + self.items.astype(np.int64)
        self.pair_keys_sorted = np.sort(pair_keys)

    def __len__(self) -> int:
        effective_samples = self.n_samples if self.max_samples <= 0 else min(self.n_samples, self.max_samples)
        return (effective_samples + self.batch_size - 1) // self.batch_size

    def _is_positive_pair(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        keys = users.astype(np.int64) * np.int64(self.n_items) + items.astype(np.int64)
        idx = np.searchsorted(self.pair_keys_sorted, keys, side='left')
        valid = idx < self.pair_keys_sorted.size
        out = np.zeros(keys.shape, dtype=bool)
        out[valid] = self.pair_keys_sorted[idx[valid]] == keys[valid]
        return out

    def _sample_negatives(self, batch_users: np.ndarray, batch_size: int) -> np.ndarray:
        if self.n_neg == 1:
            neg = np.random.randint(0, self.n_items, size=batch_size, dtype=np.int64)
            if not self.exclude_positive_negatives:
                return neg
            users_flat = batch_users
        else:
            neg = np.random.randint(0, self.n_items, size=(batch_size, self.n_neg), dtype=np.int64)
            if not self.exclude_positive_negatives:
                return neg
            users_flat = np.repeat(batch_users, self.n_neg)
            neg = neg.reshape(-1)

        # Vectorized rejection sampling: only re-draw collided negatives.
        collision = self._is_positive_pair(users_flat, neg)
        retries = 0
        while collision.any():
            neg[collision] = np.random.randint(0, self.n_items, size=int(collision.sum()), dtype=np.int64)
            collision = self._is_positive_pair(users_flat, neg)
            retries += 1
            if retries > 100:
                break

        if self.n_neg == 1:
            return neg
        return neg.reshape(batch_size, self.n_neg)

    def __iter__(self):
        effective_samples = self.n_samples if self.max_samples <= 0 else min(self.n_samples, self.max_samples)

        if self.shuffle:
            perm = np.random.permutation(self.n_samples)
            users = self.users[perm]
            items = self.items[perm]
        else:
            users = self.users
            items = self.items

        users = users[:effective_samples]
        items = items[:effective_samples]

        for start in range(0, effective_samples, self.batch_size):
            end = min(start + self.batch_size, effective_samples)

            batch_users = users[start:end]
            batch_pos_items = items[start:end]

            batch_size = end - start
            batch_neg_items = self._sample_negatives(batch_users, batch_size)

            yield (
                torch.as_tensor(batch_users, dtype=torch.long).pin_memory(),
                torch.as_tensor(batch_pos_items, dtype=torch.long).pin_memory(),
                torch.as_tensor(batch_neg_items, dtype=torch.long).pin_memory()
            )
