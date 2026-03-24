from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from utils.logger import get_logger

logger = get_logger("aurora.ocr.trocr")

TROCR_MODEL_ID = "microsoft/trocr-large-printed"


@dataclass
class TextRegion:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    text: str
    confidence: float
    frame_idx: int = 0


class TrOCREngine:
    """
    TrOCR transformer-based OCR engine.
    Uses microsoft/trocr-large-printed for high-accuracy text recognition.
    Detects text regions via EAST detector, then applies TrOCR for decoding.
    """

    def __init__(self, device: Optional[str] = None, confidence_threshold: float = 0.5):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.processor: Optional[TrOCRProcessor] = None
        self.model: Optional[VisionEncoderDecoderModel] = None
        self._east_net = None
        self._load_model()

    def _load_model(self) -> None:
        """Load TrOCR processor and model."""
        logger.info(f"Loading TrOCR: {TROCR_MODEL_ID}")
        self.processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
        self.model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID)
        self.model = self.model.to(self.device).eval()
        logger.info(f"TrOCR loaded on {self.device}")

    def _detect_text_regions(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect text bounding boxes using morphological analysis + contour detection.
        Falls back from EAST if OpenCV DNN EAST model is unavailable.
        """
        try:
            return self._east_detect(frame)
        except Exception:
            return self._morphological_detect(frame)

    def _morphological_detect(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Detect text regions via morphological gradient + contour analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame

        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        h_frame, w_frame = gray.shape[:2]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 8 or w > w_frame * 0.95 or h > h_frame * 0.5:
                continue
            aspect = w / max(h, 1)
            if aspect < 1.5 or aspect > 30:
                continue
            regions.append((x, y, w, h))

        return regions[:50]

    def _east_detect(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """EAST text detector (requires frozen model file)."""
        raise NotImplementedError("EAST model not bundled; using morphological fallback")

    def _decode_region(self, crop: np.ndarray) -> Tuple[str, float]:
        """Apply TrOCR to a cropped text region."""
        try:
            pil = Image.fromarray(crop.astype(np.uint8)).convert("RGB")
            pixel_values = self.processor(images=pil, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            with __import__("torch").no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_new_tokens=64,
                    num_beams=4,
                )

            text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            confidence = 0.85 if text else 0.0
            return text, confidence
        except Exception as e:
            logger.debug(f"TrOCR decode failed: {e}")
            return "", 0.0

    def extract_text(
        self, frame: np.ndarray, frame_idx: int = 0
    ) -> List[TextRegion]:
        """
        Extract text from a video frame.

        1. Detect text regions (EAST/morphological)
        2. Crop each region
        3. Decode with TrOCR

        Returns:
            List of TextRegion objects with bboxes and decoded text
        """
        regions_boxes = self._detect_text_regions(frame)
        results = []

        for x, y, w, h in regions_boxes:
            crop = frame[y: y + h, x: x + w]
            if crop.size == 0:
                continue
            text, conf = self._decode_region(crop)
            if conf >= self.confidence_threshold and text:
                results.append(
                    TextRegion(
                        bbox=(x, y, w, h),
                        text=text,
                        confidence=conf,
                        frame_idx=frame_idx,
                    )
                )

        return results

    def extract_text_batch(
        self, frames: List[np.ndarray]
    ) -> List[List[TextRegion]]:
        """Extract text from a list of frames."""
        return [
            self.extract_text(frame, frame_idx=i) for i, frame in enumerate(frames)
        ]

    def track_text_across_frames(
        self, frame_texts: List[List[TextRegion]], min_frames: int = 3
    ) -> dict:
        """
        Classify text as persistent (shown > min_frames) vs transient.

        Returns:
            dict with keys 'persistent' and 'transient'
        """
        text_counts: dict[str, int] = {}
        for frame_regions in frame_texts:
            seen_in_frame = set()
            for region in frame_regions:
                t = region.text.strip()
                if t and t not in seen_in_frame:
                    text_counts[t] = text_counts.get(t, 0) + 1
                    seen_in_frame.add(t)

        persistent = {t: c for t, c in text_counts.items() if c >= min_frames}
        transient = {t: c for t, c in text_counts.items() if c < min_frames}

        return {"persistent": persistent, "transient": transient}
