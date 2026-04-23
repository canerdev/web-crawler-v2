import numpy as np


def calculate_metrics(top_k_indices, target_item_ids, k=10):
    """
    Computes HR@K and NDCG@K for a batch of recommendations.

    Args:
        top_k_indices   : (B, k) — recommended item indices (PyTorch Tensor or ndarray)
        target_item_ids : (B,)   — ground-truth item indices (PyTorch Tensor or ndarray)
        k               : rank cutoff

    Returns:
        mean_hr   : float — Hit Rate @ K averaged over the batch
        mean_ndcg : float — NDCG @ K averaged over the batch
    """
    # Convert to numpy if tensors
    if hasattr(top_k_indices,   'cpu'): top_k_indices   = top_k_indices.cpu().numpy()
    if hasattr(target_item_ids, 'cpu'): target_item_ids = target_item_ids.cpu().numpy()

    # Ensure we evaluate at most k items
    top_k_indices = top_k_indices[:, :k]                        # (B, k)

    # ── HR@K ─────────────────────────────────────────────────────────────
    # hit_matrix[i, j] = True if recommendation j is the target for user i
    hit_matrix = (top_k_indices == target_item_ids[:, None])    # (B, k) bool
    hr         = hit_matrix.any(axis=1).astype(float)           # (B,)

    # ── NDCG@K ───────────────────────────────────────────────────────────
    # For users with a hit, rank = position of first match (1-indexed)
    # argmax returns the first True index along axis=1
    ranks      = hit_matrix.argmax(axis=1) + 1                  # (B,) 1-indexed
    ndcg       = np.where(hr > 0, 1.0 / np.log2(ranks + 1), 0.0)  # (B,)

    return float(np.mean(hr)), float(np.mean(ndcg))