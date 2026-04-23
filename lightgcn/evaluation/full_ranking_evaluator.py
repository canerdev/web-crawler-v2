"""
Full-Ranking Evaluator

Evaluates recommendation quality by ranking each user's candidate items
against the full item set (with train interactions masked out).
"""

import math
from typing import Dict, List, Set

import torch


class FullRankingEvaluator:
    """
    Full-ranking evaluation (all items), masking train interactions.

    Metrics are averaged over users with at least one test interaction.
    """

    def __init__(
        self,
        train_interactions: Dict[int, List[int]],
        test_interactions: Dict[int, Set[int]],
        n_items: int,
        k_values: List[int] = [10, 20],
    ):
        self.n_items = int(n_items)
        self.k_values = sorted(set(int(k) for k in k_values))
        self.max_k = max(self.k_values) if self.k_values else 20

        self.eval_users = torch.tensor(
            [u for u, items in test_interactions.items() if items],
            dtype=torch.long,
        )
        self.n_eval_users = int(self.eval_users.numel())

        max_train_user = max(train_interactions.keys(), default=-1)
        max_test_user = max(test_interactions.keys(), default=-1)
        self.n_users = max(max_train_user, max_test_user) + 1

        self.test_counts = torch.zeros(self.n_users, dtype=torch.long)

        train_rows = []
        train_cols = []
        for user_id, items in train_interactions.items():
            if not items:
                continue
            item_tensor = torch.as_tensor(items, dtype=torch.long)
            train_rows.append(torch.full((item_tensor.numel(),), int(user_id), dtype=torch.long))
            train_cols.append(item_tensor)

        if train_rows:
            self.train_rows = torch.cat(train_rows)
            self.train_cols = torch.cat(train_cols)
        else:
            self.train_rows = torch.empty(0, dtype=torch.long)
            self.train_cols = torch.empty(0, dtype=torch.long)

        test_rows = []
        test_cols = []
        for user_id, items in test_interactions.items():
            if not items:
                continue
            item_tensor = torch.as_tensor(list(items), dtype=torch.long)
            self.test_counts[int(user_id)] = int(item_tensor.numel())
            test_rows.append(torch.full((item_tensor.numel(),), int(user_id), dtype=torch.long))
            test_cols.append(item_tensor)

        if test_rows:
            all_test_rows = torch.cat(test_rows)
            all_test_cols = torch.cat(test_cols)
            pair_keys = all_test_rows.to(torch.int64) * self.n_items + all_test_cols.to(torch.int64)
            self.test_pair_keys = torch.sort(pair_keys).values
        else:
            self.test_pair_keys = torch.empty(0, dtype=torch.int64)

        self.discounts = 1.0 / torch.log2(torch.arange(2, self.max_k + 2, dtype=torch.float32))
        self.idcg_tables = {}
        for k in self.k_values:
            table = torch.zeros(k + 1, dtype=torch.float32)
            table[1:] = torch.cumsum(self.discounts[:k], dim=0)
            self.idcg_tables[k] = table

    def evaluate(
        self,
        model,
        device: torch.device,
        batch_size: int = 512,
    ) -> Dict[str, float]:
        """
        Evaluate model with full ranking.

        Args:
            model: Model with get_embeddings() method.
            device: Compute device.
            batch_size: Number of users per batch.

        Returns:
            Dict with HR@K, Recall@K, NDCG@K, and MRR.
        """
        if self.n_eval_users == 0:
            metrics = {}
            for k in self.k_values:
                metrics[f"HR@{k}"] = 0.0
                metrics[f"Recall@{k}"] = 0.0
                metrics[f"NDCG@{k}"] = 0.0
            metrics["MRR"] = 0.0
            return metrics

        model.eval()
        with torch.no_grad():
            user_emb, item_emb = model.get_embeddings()
            user_emb = user_emb.to(device)
            item_emb = item_emb.to(device)

        test_pair_keys_device = self.test_pair_keys.to(device)
        test_counts_device = self.test_counts.to(device)
        discounts_device = self.discounts.to(device)
        idcg_tables_device = {k: v.to(device) for k, v in self.idcg_tables.items()}

        hr_sum = {k: 0.0 for k in self.k_values}
        recall_sum = {k: 0.0 for k in self.k_values}
        ndcg_sum = {k: 0.0 for k in self.k_values}
        mrr_sum = 0.0

        # Global->local user index mapping, reused per batch.
        user_to_local = torch.full((self.n_users,), -1, dtype=torch.long)

        for start in range(0, self.n_eval_users, batch_size):
            end = min(start + batch_size, self.n_eval_users)
            batch_users_cpu = self.eval_users[start:end]
            batch_users = batch_users_cpu.to(device)
            bsz = int(batch_users_cpu.numel())

            with torch.no_grad():
                batch_u = user_emb[batch_users]  # (B, D)
                scores = torch.matmul(batch_u, item_emb.t())  # (B, I)

                if self.train_rows.numel() > 0:
                    selected = torch.isin(self.train_rows, batch_users_cpu)
                    if selected.any():
                        selected_rows = self.train_rows[selected]
                        selected_cols = self.train_cols[selected]

                        user_to_local.fill_(-1)
                        user_to_local[batch_users_cpu] = torch.arange(bsz, dtype=torch.long)
                        local_rows = user_to_local[selected_rows]

                        scores[local_rows.to(device), selected_cols.to(device)] = -torch.inf

                topk_items = torch.topk(scores, k=self.max_k, dim=1).indices  # (B, K)

                # Vectorized relevance lookup via encoded (user,item) pair membership.
                pair_keys = batch_users.to(torch.int64).unsqueeze(1) * self.n_items + topk_items.to(torch.int64)
                flat_keys = pair_keys.reshape(-1)
                lookup_idx = torch.searchsorted(test_pair_keys_device, flat_keys)
                valid = lookup_idx < test_pair_keys_device.numel()
                hit_flat = torch.zeros_like(valid, dtype=torch.bool)
                hit_flat[valid] = test_pair_keys_device[lookup_idx[valid]] == flat_keys[valid]
                relevance = hit_flat.view(bsz, self.max_k).to(torch.float32)

                batch_counts = test_counts_device[batch_users].clamp_min(1).to(torch.float32)

                for k in self.k_values:
                    rel_k = relevance[:, :k]
                    hit_count_k = rel_k.sum(dim=1)

                    hr_sum[k] += (hit_count_k > 0).to(torch.float32).sum().item()
                    recall_sum[k] += (hit_count_k / batch_counts).sum().item()

                    dcg_k = (rel_k * discounts_device[:k]).sum(dim=1)
                    ideal_hits = torch.minimum(test_counts_device[batch_users], torch.tensor(k, device=device))
                    idcg_k = idcg_tables_device[k][ideal_hits]
                    ndcg_k = torch.where(idcg_k > 0, dcg_k / idcg_k, torch.zeros_like(dcg_k))
                    ndcg_sum[k] += ndcg_k.sum().item()

                has_hit = relevance.bool().any(dim=1)
                first_hit_pos = torch.argmax(relevance, dim=1) + 1
                mrr_batch = torch.where(
                    has_hit,
                    1.0 / first_hit_pos.to(torch.float32),
                    torch.zeros_like(first_hit_pos, dtype=torch.float32),
                )
                mrr_sum += mrr_batch.sum().item()

        denom = float(self.n_eval_users)
        metrics = {}
        for k in self.k_values:
            metrics[f"HR@{k}"] = hr_sum[k] / denom
            metrics[f"Recall@{k}"] = recall_sum[k] / denom
            metrics[f"NDCG@{k}"] = ndcg_sum[k] / denom
        metrics["MRR"] = mrr_sum / denom
        return metrics
