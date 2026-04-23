"""
Bayesian Personalized Ranking (BPR) Loss

Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss (Rendle et al., 2009).
    
    The BPR optimization criterion maximizes the posterior probability
    of the ranking of observed over unobserved items:
    
        L_BPR = -sum_{(u,i,j) in D_s} ln(sigma(x_ui - x_uj)) + lambda * ||Theta||^2
    
    where:
        - (u, i, j) is a triple: user u, positive item i, negative item j
        - x_ui is the predicted score for user u and item i
        - sigma is the sigmoid function
        - lambda is the regularization weight
    
    This loss encourages the model to rank positive items higher than negative ones.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self, 
        pos_scores: Tensor, 
        neg_scores: Tensor,
        reg_loss: Optional[Tensor] = None,
        **_: Tensor,
    ) -> Tensor:
        """
        Compute BPR loss.
        
        Args:
            pos_scores: Scores for positive user-item pairs (batch_size,)
            neg_scores: Scores for negative user-item pairs (batch_size,)
            reg_loss: Optional regularization loss to add
        
        Returns:
            BPR loss value
        """
        # Handle multiple negatives: neg_scores may be (batch,) or (batch, n_neg)
        if neg_scores.dim() > pos_scores.dim():
            pos_scores = pos_scores.unsqueeze(-1)  # (batch,) -> (batch, 1)
        diff = pos_scores - neg_scores
        
        # Log sigmoid loss
        loss = -F.logsigmoid(diff)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        if reg_loss is not None:
            loss = loss + reg_loss
        
        return loss
