from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, SwinModel

from utils.logger import get_logger

logger = get_logger("aurora.visual.swin")

SWIN_MODEL_ID = "microsoft/swin-tiny-patch4-window7-224"


class SwinEncoder:
    """
    Swin Transformer encoder for frame-level patch features.
    Returns (196, 768) patch embeddings and attention maps per frame.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[SwinModel] = None
        self.feature_extractor = None
        self._load_model()

    def _load_model(self) -> None:
        logger.info(f"Loading Swin Transformer: {SWIN_MODEL_ID}")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(SWIN_MODEL_ID)
        self.model = SwinModel.from_pretrained(SWIN_MODEL_ID, output_attentions=True)
        self.model = self.model.to(self.device).eval()
        logger.info(f"Swin Transformer loaded on {self.device}")

    def encode_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Encode a single frame to patch embeddings.

        Args:
            frame: (H, W, 3) uint8 numpy array

        Returns:
            Tensor (196, 768) patch-level embeddings (last hidden state)
        """
        pil = Image.fromarray(frame.astype(np.uint8))
        inputs = self.feature_extractor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden = outputs.last_hidden_state.squeeze(0)
        return last_hidden

    def encode_batch(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Encode a batch of frames.

        Returns:
            Tensor (N, 196, 768)
        """
        embeddings = []
        for frame in frames:
            emb = self.encode_frame(frame)
            embeddings.append(emb.unsqueeze(0))

        if not embeddings:
            return torch.zeros(0, 196, 768, device=self.device)

        return torch.cat(embeddings, dim=0)

    def get_attention_maps(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract per-head attention maps for a frame.

        Returns:
            numpy array of shape (num_heads, seq_len, seq_len)
        """
        pil = Image.fromarray(frame.astype(np.uint8))
        inputs = self.feature_extractor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        attentions = outputs.attentions
        if attentions:
            last_attn = attentions[-1].squeeze(0).cpu().numpy()
            return last_attn

        n_patches = outputs.last_hidden_state.shape[1]
        return np.ones((1, n_patches, n_patches)) / n_patches
