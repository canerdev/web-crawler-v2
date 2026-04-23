"""
Training Utilities

Provides training loop functions for single-domain models.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .samplers import PairwiseSampler


def train_epoch(
    model: nn.Module,
    sampler: PairwiseSampler,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: GNN model
        sampler: Training data sampler
        optimizer: Optimizer
        loss_fn: Loss function (BPR or Softmax)
        device: Training device
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    total_loss = 0.0
    n_batches = 0
    
    for users, pos_items, neg_items in sampler:
        users = users.to(device, non_blocking=True)
        pos_items = pos_items.to(device, non_blocking=True)
        neg_items = neg_items.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        outputs = model(users, pos_items, neg_items)
        
        # Compute loss (model-native if provided, else external criterion)
        if 'loss' in outputs:
            loss = outputs['loss']
        else:
            loss = loss_fn(
                outputs['pos_scores'],
                outputs['neg_scores'],
                outputs['reg_loss'],
                user_emb=outputs.get('user_emb'),
                pos_item_emb=outputs.get('pos_item_emb'),
                neg_item_emb=outputs.get('neg_item_emb'),
            )
        
        # Backward pass
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches
    }


def print_training_header(
    model_name: str,
    n_users: int,
    n_items: int,
    n_train: int,
    n_test: int,
    extra_info: Optional[Dict[str, str]] = None
) -> None:
    """
    Print a pretty colored header for training scripts.
    
    Args:
        model_name: Name of the model being trained
        n_users: Number of users in dataset
        n_items: Number of items in dataset
        n_train: Number of training interactions
        n_test: Number of test interactions
        extra_info: Optional dict of additional info to display (e.g., {'Neighbors': '[50, 25]'})
    """
    density = n_train / (n_users * n_items) if (n_users * n_items) > 0 else 0
    
    # ANSI color codes
    CYAN = "\033[1;36m"
    WHITE = "\033[1;37m"
    YELLOW = "\033[1;33m"
    GREEN = "\033[1;32m"
    RESET = "\033[0m"
    
    print(f"\n{CYAN}" + "="*60 + f"{RESET}")
    print(f"{WHITE}  {model_name} Training{RESET}")
    print(f"{CYAN}" + "="*60 + f"{RESET}")
    
    # Dataset stats
    print(f"{YELLOW}  Users:{RESET} {n_users:,}  {YELLOW}Items:{RESET} {n_items:,}  {YELLOW}Density:{RESET} {density:.6f}")
    print(f"{GREEN}  Train:{RESET} {n_train:,}  {GREEN}Test:{RESET} {n_test:,}")
    
    # Extra info (e.g., neighbors for minibatch)
    if extra_info:
        info_str = "  ".join(f"{GREEN}{k}:{RESET} {v}" for k, v in extra_info.items())
        print(f"  {info_str}")
