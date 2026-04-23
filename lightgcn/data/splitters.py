"""
Train/Test Splitting Utilities

Functions for splitting interaction data into train and test sets.
"""

import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def split_train_test(
    interactions: List[Tuple[int, ...]],
    test_ratio: float = 0.2,
    strategy: str = 'random',  # 'random', 'leave_one_out', 'all_train'
    min_train_per_user: int = 1,
    seed: int = 42
) -> Tuple[Dict[int, List[int]], Dict[int, Set[int]]]:
    """
    Split interactions into train and test sets.
    
    Args:
        interactions: List of (user_id, item_id) tuples, optionally with timestamps
        test_ratio: Fraction of interactions for test (for 'random')
        strategy: Splitting strategy
            - 'random': Random split with test_ratio
            - 'leave_one_out': Keep 1 item for test per user
            - 'all_train': All interactions go to train (no test set)
        min_train_per_user: Minimum training interactions per user
        seed: Random seed
    Returns:
        train_interactions: {user_id: [item_ids]}
        test_interactions: {user_id: {item_ids}}
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Group by user
    user_items = defaultdict(list)
    has_timestamp = bool(interactions) and len(interactions[0]) >= 3
    if has_timestamp:
        for row in interactions:
            if len(row) < 3:
                continue
            u, i, ts = row[0], row[1], row[2]
            user_items[u].append((i, ts))
    else:
        for row in interactions:
            if len(row) < 2:
                continue
            u, i = row[0], row[1]
            user_items[u].append(i)
    
    train_interactions = defaultdict(list)
    test_interactions = defaultdict(set)
    
    for user, items in user_items.items():
        if strategy == 'all_train':
            # All items go to training, no test set
            if has_timestamp:
                train_interactions[user] = [i for i, _ in items]
            else:
                train_interactions[user] = items
            continue
        
        if len(items) <= min_train_per_user:
            # All items go to training
            if has_timestamp:
                train_interactions[user] = [i for i, _ in items]
            else:
                train_interactions[user] = items
            continue
        
        if strategy == 'random':
            n_test = max(1, int(len(items) * test_ratio))
            n_test = min(n_test, len(items) - min_train_per_user)
            
            if has_timestamp:
                shuffled = items.copy()
            else:
                shuffled = items.copy()
            random.shuffle(shuffled)
            
            if has_timestamp:
                test_interactions[user] = set(i for i, _ in shuffled[:n_test])
                train_interactions[user] = [i for i, _ in shuffled[n_test:]]
            else:
                test_interactions[user] = set(shuffled[:n_test])
                train_interactions[user] = shuffled[n_test:]
        
        elif strategy == 'leave_one_out':
            # Last item for test (uses timestamp order if available)
            if has_timestamp:
                items_sorted = sorted(items, key=lambda x: x[1])
                test_interactions[user] = {items_sorted[-1][0]}
                train_interactions[user] = [i for i, _ in items_sorted[:-1]]
            else:
                test_interactions[user] = {items[-1]}
                train_interactions[user] = items[:-1]
        
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")
    
    return dict(train_interactions), dict(test_interactions)
