from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("aurora.deployment.edge_runner")


class EdgeRunner:
    """
    Optimized inference pipeline for edge deployment.
    Runs CLIP + Swin + CrossModalTransformer via ONNX Runtime.
    """

    def __init__(self, model_dir: str = "deployment/onnx_models"):
        self.model_dir = Path(model_dir)
        self._sessions: Dict[str, Any] = {}

    def load_session(self, model_name: str) -> Any:
        """Load an ONNX Runtime session for a model."""
        if model_name in self._sessions:
            return self._sessions[model_name]

        import onnxruntime as ort

        model_path = self.model_dir / f"{model_name}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        session = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._sessions[model_name] = session
        logger.info(f"Loaded ONNX session: {model_name}")
        return session

    def run_clip(self, frame: np.ndarray) -> np.ndarray:
        """Run CLIP visual encoder inference."""
        session = self.load_session("clip_vit_l14")
        if frame.ndim == 3:
            frame = frame[np.newaxis]
        frame = frame.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        outputs = session.run(None, {"frame": frame})
        return outputs[0]

    def run_swin(self, frame: np.ndarray) -> np.ndarray:
        """Run Swin Transformer inference."""
        session = self.load_session("swin_transformer")
        if frame.ndim == 3:
            frame = frame[np.newaxis]
        frame = frame.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        outputs = session.run(None, {"frame": frame})
        return outputs[0]

    def run_fusion(
        self,
        visual: np.ndarray,
        audio: np.ndarray,
        text: np.ndarray,
    ) -> np.ndarray:
        """Run CrossModalTransformer fusion inference."""
        session = self.load_session("cross_modal_transformer")
        outputs = session.run(
            None,
            {
                "visual_feat": visual.astype(np.float32),
                "audio_feat": audio.astype(np.float32),
                "text_feat": text.astype(np.float32),
            },
        )
        return outputs[0]

    def run_full_pipeline(
        self, frames: List[np.ndarray], audio_embedding: np.ndarray, text: str
    ) -> np.ndarray:
        """Run the complete edge inference pipeline."""
        import cv2

        frame_embeddings = []
        for frame in frames[:8]:
            resized = cv2.resize(frame, (224, 224))
            emb = self.run_clip(resized)
            frame_embeddings.append(emb)

        visual = np.mean(frame_embeddings, axis=0, keepdims=True) if frame_embeddings else np.zeros((1, 1, 768))
        fused = self.run_fusion(visual, audio_embedding, np.zeros((1, 1, 768)))
        return fused
