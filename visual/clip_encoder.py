from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.logger import get_logger

logger = get_logger("aurora.visual.clip")


class CLIPFrameEncoder:
    """
    CLIP ViT-L/14 frame encoder — largest CLIP model (307M params).
    Encodes video frames and text into a shared 768-dim embedding space.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self._load_model()

    def _load_model(self) -> None:
        """Load CLIP ViT-L/14 model and preprocessing pipeline."""
        import clip

        logger.info("Loading CLIP ViT-L/14...")
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.model = self.model.eval()
        logger.info(f"CLIP ViT-L/14 loaded on {self.device}")

    def encode_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Encode a list of RGB frames to CLIP visual embeddings.

        Args:
            frames: List of (H, W, 3) uint8 numpy arrays

        Returns:
            Tensor of shape (N, 768) normalized visual embeddings
        """
        if not frames:
            return torch.zeros(0, 768, device=self.device)

        images = []
        for frame in frames:
            pil_img = Image.fromarray(frame.astype(np.uint8))
            images.append(self.preprocess(pil_img))

        batch = torch.stack(images).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_image(batch)
            embeddings = F.normalize(embeddings.float(), dim=-1)

        return embeddings

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text strings to CLIP text embeddings.

        Args:
            texts: List of text strings

        Returns:
            Tensor of shape (N, 768) normalized text embeddings
        """
        import clip

        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_text(tokens)
            embeddings = F.normalize(embeddings.float(), dim=-1)

        return embeddings

    def compute_similarity_matrix(
        self,
        visual_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between all frame-text pairs.

        Args:
            visual_emb: (N_frames, 768)
            text_emb: (N_texts, 768)

        Returns:
            Similarity matrix of shape (N_frames, N_texts)
        """
        visual_norm = F.normalize(visual_emb.float(), dim=-1)
        text_norm = F.normalize(text_emb.float(), dim=-1)
        return torch.matmul(visual_norm, text_norm.T)

    def retrieve_frames_by_query(
        self,
        query: str,
        frames: List[np.ndarray],
        top_k: int = 5,
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Find the top-k frames most similar to a text query.

        Returns:
            List of (frame_idx, similarity_score, frame) tuples
        """
        visual_emb = self.encode_frames(frames)
        text_emb = self.encode_text([query])
        sim = self.compute_similarity_matrix(visual_emb, text_emb).squeeze(-1)

        top_k = min(top_k, len(frames))
        values, indices = torch.topk(sim, top_k)

        return [
            (int(idx), float(val), frames[int(idx)])
            for idx, val in zip(indices, values)
        ]
