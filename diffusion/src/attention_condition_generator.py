import torch
import torch.nn as nn


class AttentionConditionGenerator(nn.Module):
    """
    GNN-Anchored Cross-Domain Condition Generator.

    Input vectors:
      h_u_cross  : (B, D) — GNN user embedding (Query / Anchor)
      h_u_source : (B, D) — Source domain history aggregator output (Source Domain Intent)
      h_u_target : (B, D) — Target domain history aggregator output (Target Domain Intent)

    Mechanism:
      GNN embedding acts as Query over [source_intent, target_intent] as Key/Value.
      The model learns per-user, per-step dynamic weighting between domains.

    Returns:
      c_ud : (B, D) — Diffusion condition vector
    """

    def __init__(self, embed_dim=256, num_heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, h_u_cross, h_u_source, h_u_target):
        """
        Args:
            h_u_cross  : (B, D)
            h_u_source : (B, D)
            h_u_target : (B, D)

        Returns:
            c_ud : (B, D)
        """
        # 1. KV: Source + Target intent side by side -> (B, 2, D)
        kv = torch.stack([h_u_source, h_u_target], dim=1)

        # 2. Query: GNN identity -> (B, 1, D)
        query = h_u_cross.unsqueeze(1)

        # 3. Pre-Norm Cross-Attention + Residual
        # TODO: The attention weights can be retrieved from here by rotating them. This is beneficial for explainability.
        attn_out, _ = self.cross_attention(
            query=self.norm1(query),
            key=self.norm1(kv),
            value=self.norm1(kv)
        )
        x = query + self.dropout(attn_out)              # (B, 1, D)

        # 4. Pre-Norm FFN + Residual
        x = x + self.dropout(self.ffn(self.norm2(x)))  # (B, 1, D)

        return x.squeeze(1)                            # (B, D)