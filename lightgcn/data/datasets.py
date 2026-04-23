"""
Dataset Classes for Recommendation

Container classes for recommendation datasets.
"""

import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .loaders import load_interactions_from_file, reindex_interactions
from .splitters import split_train_test

def create_interaction_pairs(
    train_interactions: Dict[int, List[int]]
) -> Tensor:
    """
    Create tensor of user-item pairs for adjacency matrix.
    
    Args:
        train_interactions: {user_id: [item_ids]}
    
    Returns:
        Tensor of shape (2, n_interactions) with [users, items]
    """
    user_ids = np.array(list(train_interactions.keys()))
    item_lists = list(train_interactions.values())
    lengths = np.array([len(items) for items in item_lists])
    
    users = np.repeat(user_ids, lengths)
    items = np.concatenate([np.array(items) for items in item_lists])
    
    return torch.from_numpy(np.stack([users, items]))


@dataclass
class RecommendationDataset:
    """
    Container for recommendation dataset with train/test split.
    """
    n_users: int
    n_items: int
    train_interactions: Dict[int, List[int]]
    test_interactions: Dict[int, Set[int]]
    user_mapping: Optional[Dict[int, int]] = None
    item_mapping: Optional[Dict[int, int]] = None
    
    # Computed attributes (set in __post_init__)
    n_train_interactions: int = field(init=False)
    n_test_interactions: int = field(init=False)
    _interaction_pairs: Optional[Tensor] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Compute statistics after initialization."""
        self.n_train_interactions = sum(len(items) for items in self.train_interactions.values())
        self.n_test_interactions = sum(len(items) for items in self.test_interactions.values())
    
    @property
    def interaction_pairs(self) -> Tensor:
        if self._interaction_pairs is None:
            self._interaction_pairs = create_interaction_pairs(self.train_interactions)
        return self._interaction_pairs
    
    @property
    def density(self) -> float:
        return self.n_train_interactions / (self.n_users * self.n_items)
    
    def __repr__(self) -> str:
        return (
            f"RecommendationDataset(\n"
            f"  n_users={self.n_users},\n"
            f"  n_items={self.n_items},\n"
            f"  n_train={self.n_train_interactions},\n"
            f"  n_test={self.n_test_interactions},\n"
            f"  density={self.density:.6f}\n"
            f")"
        )
    
    @classmethod
    def from_file(
        cls,
        filepath: str,
        test_ratio: float = 0.2,
        split_strategy: str = 'random',
        seed: int = 42,
        max_rows: int = None
    ) -> 'RecommendationDataset':
        """
        Create dataset from interaction file.
        
        Args:
            filepath: Path to interaction file
            test_ratio: Test set ratio
            split_strategy: How to split train/test
            seed: Random seed
            max_rows: Maximum rows to load (None for all)
        
        Returns:
            RecommendationDataset instance
        """
        # Load and process
        print(f"Loading interactions from {filepath}...")
        interactions = load_interactions_from_file(filepath, max_rows=max_rows)
        print(f"Loaded {len(interactions)} interactions")
        
        print("Reindexing user and item IDs...")
        reindexed, user_mapping, item_mapping = reindex_interactions(interactions)
        
        n_users = len(user_mapping)
        n_items = len(item_mapping)
        print(f"Found {n_users} users and {n_items} items")
        
        train, test = split_train_test(reindexed, test_ratio, split_strategy, seed=seed)
        
        return cls(
            n_users=n_users,
            n_items=n_items,
            train_interactions=train,
            test_interactions=test,
            user_mapping=user_mapping,
            item_mapping=item_mapping
        )

    @classmethod
    def from_separate_files(
        cls,
        train_filepath: str,
        test_filepath: str,
        max_rows: int = None
    ) -> 'RecommendationDataset':
        """
        Create dataset from separate train and test files.
        
        Useful for cross-domain training where train and test data
        come from different sources.
        
        Args:
            train_filepath: Path to training interaction file
            test_filepath: Path to test interaction file
            max_rows: Maximum rows to load (None for all)
        
        Returns:
            RecommendationDataset instance
        """
        from collections import defaultdict
        
        # Load train interactions
        print(f"Loading training interactions from {train_filepath}...")
        train_raw = load_interactions_from_file(train_filepath, max_rows=max_rows)
        print(f"Loaded {len(train_raw)} training interactions")
        
        # Load test interactions
        print(f"Loading test interactions from {test_filepath}...")
        test_raw = load_interactions_from_file(test_filepath, max_rows=max_rows)
        print(f"Loaded {len(test_raw)} test interactions")
        
        # Combine all interactions for unified ID mapping
        all_interactions = train_raw + test_raw
        
        print("Building unified user/item ID mappings...")
        _, user_mapping, item_mapping = reindex_interactions(all_interactions)
        
        n_users = len(user_mapping)
        n_items = len(item_mapping)
        print(f"Found {n_users} users and {n_items} items")
        
        # Convert train interactions using unified mapping
        train_interactions: Dict[int, List[int]] = defaultdict(list)
        for row in train_raw:
            user, item = row[0], row[1]
            if user in user_mapping and item in item_mapping:
                train_interactions[user_mapping[user]].append(item_mapping[item])
        
        # Convert test interactions using unified mapping
        test_interactions: Dict[int, Set[int]] = defaultdict(set)
        skipped_test = 0
        for row in test_raw:
            user, item = row[0], row[1]
            if user in user_mapping and item in item_mapping:
                test_interactions[user_mapping[user]].add(item_mapping[item])
            else:
                skipped_test += 1
        
        if skipped_test > 0:
            print(f"Warning: Skipped {skipped_test} test interactions with unknown users/items")
        
        print(f"Train users: {len(train_interactions)}, Test users: {len(test_interactions)}")
        
        return cls(
            n_users=n_users,
            n_items=n_items,
            train_interactions=dict(train_interactions),
            test_interactions=dict(test_interactions),
            user_mapping=user_mapping,
            item_mapping=item_mapping
        )
