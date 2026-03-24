from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, TimesformerModel

from utils.logger import get_logger

logger = get_logger("aurora.visual.timesformer")

TIMESFORMER_MODEL_ID = "facebook/timesformer-base-finetuned-k400"


@dataclass
class ActionPrediction:
    label: str
    confidence: float
    top5: List[Dict[str, float]] = field(default_factory=list)


class TimeSformerEncoder:
    """
    TimeSformer encoder for temporal video understanding.
    Model: facebook/timesformer-base-finetuned-k400
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[TimesformerModel] = None
        self._load_model()

    def _load_model(self) -> None:
        logger.info(f"Loading TimeSformer: {TIMESFORMER_MODEL_ID}")
        self.processor = AutoProcessor.from_pretrained(TIMESFORMER_MODEL_ID)
        self.model = TimesformerModel.from_pretrained(TIMESFORMER_MODEL_ID)
        self.model = self.model.to(self.device).eval()
        logger.info(f"TimeSformer loaded on {self.device}")

    def encode_video_clip(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Encode a video clip (list of frames) to a video-level embedding.

        Args:
            frames: List of (H, W, 3) uint8 numpy arrays

        Returns:
            Tensor of shape (1, 768) video-level embedding
        """
        if not frames:
            return torch.zeros(1, 768, device=self.device)

        pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in frames]

        inputs = self.processor(videos=pil_frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    def classify_action(self, frames: List[np.ndarray]) -> ActionPrediction:
        """Classify the action in a video clip."""
        from transformers import TimesformerForVideoClassification

        try:
            classifier = TimesformerForVideoClassification.from_pretrained(
                TIMESFORMER_MODEL_ID
            )
            classifier = classifier.to(self.device).eval()

            pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in frames]
            inputs = self.processor(videos=pil_frames, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = classifier(**inputs)
                logits = outputs.logits

            probs = torch.softmax(logits, dim=-1).squeeze(0)
            top5_values, top5_indices = torch.topk(probs, k=min(5, len(probs)))

            label_map = classifier.config.id2label
            top_label = label_map.get(int(top5_indices[0]), "unknown")
            top_conf = float(top5_values[0])

            top5 = [
                {"label": label_map.get(int(idx), "unknown"), "confidence": float(val)}
                for idx, val in zip(top5_indices, top5_values)
            ]

            return ActionPrediction(label=top_label, confidence=top_conf, top5=top5)

        except Exception as e:
            logger.warning(f"Action classification failed: {e}")
            embedding = self.encode_video_clip(frames)
            return ActionPrediction(label="unknown", confidence=0.0, top5=[])

    def extract_temporal_features(
        self,
        frames: List[np.ndarray],
        window_size: int = 8,
    ) -> List[torch.Tensor]:
        """
        Extract per-window temporal embeddings using a sliding window.

        Returns:
            List of (1, 768) tensors, one per window
        """
        if len(frames) < window_size:
            return [self.encode_video_clip(frames)]

        window_embeddings = []
        step = max(1, window_size // 2)

        for start in range(0, len(frames) - window_size + 1, step):
            window = frames[start: start + window_size]
            emb = self.encode_video_clip(window)
            window_embeddings.append(emb)

        return window_embeddings
