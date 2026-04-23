"""Graph utilities for LightGCN."""

import torch
from torch import Tensor


def create_adj_matrix(
    n_users: int,
    n_items: int,
    user_item_pairs: Tensor,
    device: torch.device
) -> Tensor:
    """
    Create normalized adjacency matrix for user-item bipartite graph.

    The adjacency matrix A has the structure:
        A = | 0    R   |
            | R^T  0   |

    where R is the user-item interaction matrix.

    Normalization follows symmetric normalization: D^{-1/2} A D^{-1/2}

    Args:
        n_users: Number of users
        n_items: Number of items
        user_item_pairs: Tensor of shape (2, num_interactions) containing
                        [user_indices, item_indices]
        device: Target device for the tensor

    Returns:
        Sparse tensor of shape (n_users + n_items, n_users + n_items)
    """
    users = user_item_pairs[0]
    items = user_item_pairs[1] + n_users  # Offset item indices

    # Create edges in both directions (user->item and item->user)
    row = torch.cat([users, items]).to(device)
    col = torch.cat([items, users]).to(device)

    n_nodes = n_users + n_items

    # Compute degree for normalization
    deg = torch.zeros(n_nodes, device=device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=device, dtype=torch.float))

    # D^{-1/2}
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # Symmetric normalization weights: 1 / sqrt(|N_u| * |N_i|)
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # Create sparse adjacency matrix
    indices = torch.stack([row, col])
    adj = torch.sparse_coo_tensor(
        indices, edge_weight, (n_nodes, n_nodes), device=device
    )

    return adj.coalesce()
