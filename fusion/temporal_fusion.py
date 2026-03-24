from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger

logger = get_logger("aurora.fusion.temporal")


class TemporalFusion(nn.Module):
    """
    Temporal alignment and fusion module.
    Aligns audio transcript timestamps with visual frame timestamps,
    then applies temporal position-aware pooling.
    """

    def __init__(self, hidden_dim: int = 512, max_seq_len: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1, batch_first=True)
        self.position_encoding = self._build_sinusoidal_pe(max_seq_len, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        """Build sinusoidal positional encoding matrix."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        return pe.unsqueeze(0)

    def align_audio_to_frames(
        self,
        audio_embeddings: torch.Tensor,
        audio_timestamps: List[Tuple[float, float]],
        frame_timestamps: List[float],
    ) -> torch.Tensor:
        """
        Align audio segment embeddings to frame-level temporal positions.

        Args:
            audio_embeddings: (N_segments, hidden_dim)
            audio_timestamps: List of (start_sec, end_sec) per segment
            frame_timestamps: List of frame timestamps in seconds

        Returns:
            Tensor of (N_frames, hidden_dim) aligned audio features
        """
        n_frames = len(frame_timestamps)
        if audio_embeddings.shape[0] == 0 or n_frames == 0:
            return torch.zeros(n_frames, self.hidden_dim, device=audio_embeddings.device)

        aligned = torch.zeros(n_frames, self.hidden_dim, device=audio_embeddings.device)
        counts = torch.zeros(n_frames, 1, device=audio_embeddings.device)

        for seg_idx, (start, end) in enumerate(audio_timestamps):
            if seg_idx >= audio_embeddings.shape[0]:
                break
            for frame_idx, ts in enumerate(frame_timestamps):
                if start <= ts < end:
                    aligned[frame_idx] += audio_embeddings[seg_idx]
                    counts[frame_idx] += 1

        counts = counts.clamp(min=1)
        aligned = aligned / counts

        fallback_mask = counts.squeeze(-1) == 1
        if not fallback_mask.all():
            non_zero = aligned[~fallback_mask.squeeze(-1)]
            if non_zero.shape[0] > 0:
                mean_emb = non_zero.mean(dim=0)
                aligned[fallback_mask] = mean_emb

        return aligned

    def forward(
        self,
        visual_seq: torch.Tensor,
        audio_seq: torch.Tensor,
        visual_timestamps: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Apply temporal attention-based fusion of visual and audio sequences.

        Args:
            visual_seq: (batch, T_v, hidden_dim)
            audio_seq:  (batch, T_a, hidden_dim)

        Returns:
            (batch, T_v, hidden_dim) temporally-fused representation
        """
        seq_len = visual_seq.shape[1]
        pe = self.position_encoding[:, :seq_len, :].to(visual_seq.device)
        visual_pe = visual_seq + pe

        fused, _ = self.temporal_attn(query=visual_pe, key=audio_seq, value=audio_seq)
        out = self.norm(fused + visual_seq)
        return out
