"""
Sampled Metrics Evaluator

Evaluates by ranking each positive against N random negatives (e.g., 99).
Fast but known to inflate metrics compared to full ranking evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Set


class SampledEvaluator:
    """
    Sampled evaluation: rank each positive among N random negatives.
    
    Common in older papers (NCF, etc.) but known to have biases.
    """
    
    def __init__(
        self,
        test_interactions: Dict[int, Set[int]],
        n_items: int,
        n_negatives: int = 99,
        k_values: List[int] = [10, 20],
        seed: int = 42
    ):
        """
        Args:
            test_interactions: Ground truth {user_id: {item_ids}}
            n_items: Total number of items
            n_negatives: Number of negative samples per positive (default: 99)
            k_values: K values for metrics
            seed: Random seed for reproducibility
        """
        self.n_items = n_items
        self.n_negatives = n_negatives
        self.k_values = k_values
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Flatten test interactions
        users, pos_items = [], []
        for user, items in test_interactions.items():
            for item in items:
                users.append(user)
                pos_items.append(item)
        
        self.users = torch.tensor(users, dtype=torch.long)
        self.pos_items = torch.tensor(pos_items, dtype=torch.long)
        self.n_samples = len(users)
        
        # Sample all negatives at once (single vectorized call)
        self.neg_items = torch.randint(0, n_items, (self.n_samples, n_negatives))
    
    def evaluate(
        self,
        model,
        device: torch.device,
        batch_size: int = 8192
    ) -> Dict[str, float]:
        """
        Evaluate model using sampled metrics.
        
        Args:
            model: Model with get_embeddings() method
            device: Compute device
            batch_size: Batch size for evaluation
        
        Returns:
            Dictionary with HR@K and NDCG@K for each K, plus MRR
        """
        model.eval()
        
        with torch.no_grad():
            user_emb, item_emb = model.get_embeddings()
            user_emb = user_emb.to(device)
            item_emb = item_emb.to(device)
        
        all_ranks = []
        
        for start in range(0, self.n_samples, batch_size):
            end = min(start + batch_size, self.n_samples)
            
            batch_users = self.users[start:end].to(device)
            batch_pos = self.pos_items[start:end].to(device)
            batch_neg = self.neg_items[start:end].to(device)
            
            with torch.no_grad():
                u_emb = user_emb[batch_users]
                pos_emb = item_emb[batch_pos]
                neg_emb = item_emb[batch_neg]
                
                # Positive scores: (batch,) -> (batch, 1)
                pos_scores = (u_emb * pos_emb).sum(dim=-1, keepdim=True)
                
                # Negative scores: (batch, n_neg)
                neg_scores = torch.bmm(neg_emb, u_emb.unsqueeze(-1)).squeeze(-1)
                
                # Rank = number of negatives scoring higher + 1
                ranks = (neg_scores > pos_scores).sum(dim=-1) + 1
                all_ranks.append(ranks.cpu())
        
        ranks = torch.cat(all_ranks).float().numpy()
        
        # Compute metrics
        metrics = {}
        for k in self.k_values:
            hits = ranks <= k
            metrics[f'HR@{k}'] = float(hits.mean())
            metrics[f'NDCG@{k}'] = float(np.where(hits, 1.0 / np.log2(ranks + 1), 0.0).mean())
        
        metrics['MRR'] = float((1.0 / ranks).mean())
        
        return metrics


