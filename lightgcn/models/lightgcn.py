"""LightGCN model."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, Tuple

from .graph_utils import create_adj_matrix


class LightGCN(nn.Module):
    """LightGCN model for collaborative filtering (He et al., SIGIR 2020)."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.0,
        reg_weight: float = 1e-4,
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.reg_weight = reg_weight

        # Embedding layers (only trainable parameters)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Graph convolution layers
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(n_layers)])

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Initialize embeddings
        self._init_embeddings()

        self.adj: Optional[Tensor] = None

    def _init_embeddings(self) -> None:
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def set_adjacency_matrix(self, user_item_pairs: Tensor, device: torch.device) -> None:
        self.adj = create_adj_matrix(
            self.n_users, self.n_items, user_item_pairs, device
        )

    def get_embeddings(self) -> Tuple[Tensor, Tensor]:
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        if self.dropout is not None:
            x = self.dropout(x)

        all_embeddings = [x]
        for conv in self.convs:
            x = conv(x, self.adj)
            all_embeddings.append(x)

        final_embedding = torch.stack(all_embeddings, dim=0).mean(dim=0)
        return final_embedding[:self.n_users], final_embedding[self.n_users:]

    def forward(
        self,
        users: Tensor,
        pos_items: Tensor,
        neg_items: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        user_emb_all, item_emb_all = self.get_embeddings()

        # Get embeddings for batch
        u_emb = user_emb_all[users]
        pos_i_emb = item_emb_all[pos_items]

        # Positive scores
        pos_scores = (u_emb * pos_i_emb).sum(dim=-1)

        result = {
            'pos_scores': pos_scores,
            'user_emb': u_emb,
            'pos_item_emb': pos_i_emb,
        }

        if neg_items is not None:
            neg_i_emb = item_emb_all[neg_items]
            if neg_items.dim() == 1:
                neg_scores = (u_emb * neg_i_emb).sum(dim=-1)
            else:
                neg_scores = (u_emb.unsqueeze(1) * neg_i_emb).sum(dim=-1)
            result['neg_scores'] = neg_scores
            result['neg_item_emb'] = neg_i_emb

        # L2 regularization on embeddings used in batch
        # Uses initial embeddings (0-th layer) for regularization
        u_emb_0 = self.user_embedding(users)
        pos_i_emb_0 = self.item_embedding(pos_items)

        reg_loss = self.reg_weight * (
            u_emb_0.norm(2).pow(2) +
            pos_i_emb_0.norm(2).pow(2)
        )

        if neg_items is not None:
            neg_indices = neg_items.flatten() if neg_items.dim() > 1 else neg_items
            neg_i_emb_0 = self.item_embedding(neg_indices)
            neg_reg = neg_i_emb_0.norm(2).pow(2)
            reg_loss += self.reg_weight * neg_reg

        reg_loss = reg_loss / users.size(0)
        result['reg_loss'] = reg_loss

        return result

    def predict(self, users: Tensor, items: Tensor) -> Tensor:
        with torch.no_grad():
            user_emb, item_emb = self.get_embeddings()
            u_emb = user_emb[users]
            i_emb = item_emb[items]
            return (u_emb * i_emb).sum(dim=-1)

    def recommend(self, users: Tensor, k: int = 10) -> Tensor:
        with torch.no_grad():
            user_emb, item_emb = self.get_embeddings()
            u_emb = user_emb[users]  # (batch_size, dim)

            # Compute scores for all items
            scores = torch.mm(u_emb, item_emb.t())  # (batch_size, n_items)

            # Get top-k items
            _, topk_items = torch.topk(scores, k, dim=-1)

            return topk_items


class LightGCNConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        return torch.sparse.mm(adj, x)

