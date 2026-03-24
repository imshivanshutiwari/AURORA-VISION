import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger

logger = get_logger("aurora.fusion.attention_pooling")


class AttentionPooling(nn.Module):
    """
    Attention-weighted pooling over a sequence of embeddings.
    Learns a query vector that attends over sequence tokens to produce
    a single summary vector.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, sequence: torch.Tensor, mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pool a sequence to a single vector using learned attention.

        Args:
            sequence: (batch, seq_len, input_dim)
            mask: Optional (batch, seq_len) boolean mask (True = valid)

        Returns:
            (pooled: (batch, input_dim), attention_weights: (batch, seq_len))
        """
        scores = self.attention_net(sequence).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights_clamped = torch.nan_to_num(weights, nan=1.0 / scores.shape[-1])

        pooled = torch.bmm(weights_clamped.unsqueeze(1), sequence).squeeze(1)
        return pooled, weights_clamped


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling: multiple learned query vectors
    producing diverse summary representations.
    """

    def __init__(self, input_dim: int, n_heads: int = 4, output_dim: int = 512):
        super().__init__()
        self.n_heads = n_heads
        self.head_pools = nn.ModuleList([
            AttentionPooling(input_dim, input_dim // 2) for _ in range(n_heads)
        ])
        self.output_proj = nn.Linear(input_dim * n_heads, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self, sequence: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply multiple attention pooling heads and concatenate.

        Returns:
            (batch, output_dim)
        """
        head_outputs = []
        for pool in self.head_pools:
            pooled, _ = pool(sequence, mask)
            head_outputs.append(pooled)

        concat = torch.cat(head_outputs, dim=-1)
        out = self.output_proj(concat)
        return self.norm(out)
