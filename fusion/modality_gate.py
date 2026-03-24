from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger

logger = get_logger("aurora.fusion.modality_gate")


class ModalityGate(nn.Module):
    """
    Learnable gating module that weights each modality's contribution
    to the fused representation based on content-aware signals.

    Each gate is a sigmoid-activated linear layer over the modality embedding.
    Missing modalities are handled by setting their gate to zero.
    """

    def __init__(self, hidden_dim: int = 512, n_modalities: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modalities = n_modalities

        self.visual_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.audio_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        text: torch.Tensor,
        missing_modalities: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gated representations for each modality.

        Args:
            visual: (batch, hidden_dim)
            audio: (batch, hidden_dim)
            text: (batch, hidden_dim)
            missing_modalities: list of modality names to zero out

        Returns:
            dict with 'visual', 'audio', 'text' gated tensors and gate values
        """
        gate_v = self.visual_gate(visual)
        gate_a = self.audio_gate(audio)
        gate_t = self.text_gate(text)

        if missing_modalities:
            device = visual.device
            if "audio" in missing_modalities:
                gate_a = torch.zeros_like(gate_a)
            if "text" in missing_modalities:
                gate_t = torch.zeros_like(gate_t)
            if "visual" in missing_modalities:
                gate_v = torch.zeros_like(gate_v)

        total = gate_v + gate_a + gate_t + 1e-8
        gate_v_norm = gate_v / total
        gate_a_norm = gate_a / total
        gate_t_norm = gate_t / total

        return {
            "visual": gate_v_norm * visual,
            "audio": gate_a_norm * audio,
            "text": gate_t_norm * text,
            "gate_v": gate_v_norm,
            "gate_a": gate_a_norm,
            "gate_t": gate_t_norm,
        }

    def get_gate_values(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        text: torch.Tensor,
    ) -> Dict[str, float]:
        """Return scalar gate values for dashboard display."""
        with torch.no_grad():
            gated = self.forward(visual, audio, text)
            return {
                "visual": float(gated["gate_v"].mean().item()),
                "audio": float(gated["gate_a"].mean().item()),
                "text": float(gated["gate_t"].mean().item()),
            }
