from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger

logger = get_logger("aurora.fusion.cross_modal")


@dataclass
class FusedRepresentation:
    fused: torch.Tensor
    visual_attended: torch.Tensor
    audio_attended: torch.Tensor
    text_attended: torch.Tensor
    gate_weights: Dict[str, float] = field(default_factory=dict)
    attention_weights: Dict[str, torch.Tensor] = field(default_factory=dict)


class CrossModalTransformer(nn.Module):
    """
    Cross-Modal Transformer that fuses visual, audio, and text modalities.

    Architecture:
        - Per-modality linear projections to hidden_dim (512)
        - 3 layers of cross-modal multi-head attention (8 heads)
        - Learnable modality gates (sigmoid)
        - Final projection + LayerNorm + GELU activation

    Attention pattern per layer:
        - Visual attends to Audio
        - Visual attends to Text
        - All → joint via concat + projection
    """

    def __init__(
        self,
        visual_dim: int = 768,
        audio_dim: int = 512,
        text_dim: int = 768,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.vis_audio_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.vis_text_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.aud_vis_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])

        self.joint_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.gate_v = nn.Linear(hidden_dim, 1)
        self.gate_a = nn.Linear(hidden_dim, 1)
        self.gate_t = nn.Linear(hidden_dim, 1)

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self._last_attn_weights: Dict[str, torch.Tensor] = {}

    def _ensure_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is (batch, seq, dim)."""
        if x.dim() == 1:
            return x.unsqueeze(0).unsqueeze(0)
        if x.dim() == 2:
            return x.unsqueeze(0)
        return x

    def forward(
        self,
        visual_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        text_feat: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> FusedRepresentation:
        """
        Forward pass through cross-modal transformer.

        Args:
            visual_feat: (batch, seq_v, visual_dim) or (batch, visual_dim)
            audio_feat:  (batch, seq_a, audio_dim) or (batch, audio_dim)
            text_feat:   (batch, seq_t, text_dim) or (batch, text_dim)

        Returns:
            FusedRepresentation with (batch, hidden_dim) fused tensor
        """
        v = self.visual_proj(self._ensure_3d(visual_feat))
        a = self.audio_proj(self._ensure_3d(audio_feat))
        t = self.text_proj(self._ensure_3d(text_feat))

        attn_va_all, attn_vt_all, attn_av_all = [], [], []

        for layer_idx in range(self.n_layers):
            v_attn_a, w_va = self.vis_audio_attn[layer_idx](query=v, key=a, value=a)
            v_attn_t, w_vt = self.vis_text_attn[layer_idx](query=v, key=t, value=t)
            a_attn_v, w_av = self.aud_vis_attn[layer_idx](query=a, key=v, value=v)

            v = self.dropout(v + v_attn_a + v_attn_t)
            a = self.dropout(a + a_attn_v)

            attn_va_all.append(w_va)
            attn_vt_all.append(w_vt)
            attn_av_all.append(w_av)

        v_pool = v.mean(dim=1)
        a_pool = a.mean(dim=1)
        t_pool = t.mean(dim=1)

        joint = torch.cat([v_pool, a_pool, t_pool], dim=-1)
        joint_fused = self.joint_proj(joint)

        gate_v = torch.sigmoid(self.gate_v(v_pool))
        gate_a = torch.sigmoid(self.gate_a(a_pool))
        gate_t = torch.sigmoid(self.gate_t(t_pool))

        gate_sum = gate_v + gate_a + gate_t + 1e-8
        gate_v_norm = gate_v / gate_sum
        gate_a_norm = gate_a / gate_sum
        gate_t_norm = gate_t / gate_sum

        fused = (
            gate_v_norm * v_pool
            + gate_a_norm * a_pool
            + gate_t_norm * t_pool
            + joint_fused
        ) / 2.0
        fused = self.output_norm(fused)

        self._last_attn_weights = {
            "visual_audio": attn_va_all[-1],
            "visual_text": attn_vt_all[-1],
            "audio_visual": attn_av_all[-1],
        }

        batch_size = fused.shape[0]
        return FusedRepresentation(
            fused=fused,
            visual_attended=v_pool,
            audio_attended=a_pool,
            text_attended=t_pool,
            gate_weights={
                "visual": float(gate_v_norm.mean().item()),
                "audio": float(gate_a_norm.mean().item()),
                "text": float(gate_t_norm.mean().item()),
            },
            attention_weights=self._last_attn_weights,
        )

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Return the last computed attention weight matrices."""
        return self._last_attn_weights

    def handle_missing_modality(self, modality: str) -> None:
        """
        Zero-out gate for a missing modality so it doesn't contribute to fusion.
        Call before forward() when a modality is unavailable.
        """
        if modality == "audio":
            with torch.no_grad():
                nn.init.constant_(self.gate_a.bias, -10.0)
        elif modality == "text":
            with torch.no_grad():
                nn.init.constant_(self.gate_t.bias, -10.0)
        elif modality == "visual":
            with torch.no_grad():
                nn.init.constant_(self.gate_v.bias, -10.0)
        logger.info(f"Gated out missing modality: {modality}")
