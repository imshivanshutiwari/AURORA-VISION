from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ocr.trocr_engine import TextRegion
from utils.logger import get_logger

logger = get_logger("aurora.ocr.layout_analyzer")


@dataclass
class LayoutZone:
    zone_id: str
    region_type: str  # 'title', 'subtitle', 'body', 'watermark', 'ticker'
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    texts: List[TextRegion] = field(default_factory=list)
    normalized_position: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


class LayoutAnalyzer:
    """
    Analyzes the spatial layout of detected text regions within video frames.
    Classifies text regions by position (title, ticker, watermark, body text).
    """

    ZONE_RULES = {
        "title": {"y_max_frac": 0.2},
        "ticker": {"y_min_frac": 0.85},
        "watermark": {"x_min_frac": 0.7, "y_min_frac": 0.8},
        "body": {},
    }

    def analyze_frame(
        self,
        frame: np.ndarray,
        text_regions: List[TextRegion],
    ) -> List[LayoutZone]:
        """
        Classify text regions into layout zones based on position in frame.

        Args:
            frame: (H, W, 3) numpy array
            text_regions: detected text regions

        Returns:
            List of LayoutZone objects
        """
        h, w = frame.shape[:2]
        zones: Dict[str, LayoutZone] = {}

        for region in text_regions:
            x, y, bw, bh = region.bbox
            zone_type = self._classify_position(x, y, bw, bh, w, h)
            norm_pos = (x / w, y / h, (x + bw) / w, (y + bh) / h)

            zone_id = f"{zone_type}_{y // (h // 4)}"
            if zone_id not in zones:
                zones[zone_id] = LayoutZone(
                    zone_id=zone_id,
                    region_type=zone_type,
                    bbox=(x, y, bw, bh),
                    normalized_position=norm_pos,
                )
            zones[zone_id].texts.append(region)

        return list(zones.values())

    def _classify_position(
        self, x: int, y: int, w: int, h: int, frame_w: int, frame_h: int
    ) -> str:
        """Classify a text region type by its normalized position."""
        y_frac = y / max(frame_h, 1)
        x_frac = x / max(frame_w, 1)
        w_frac = w / max(frame_w, 1)

        if y_frac < 0.15 and w_frac > 0.3:
            return "title"
        if y_frac > 0.85:
            return "ticker"
        if x_frac > 0.7 and y_frac > 0.75:
            return "watermark"
        if y_frac < 0.35 and w_frac > 0.2:
            return "subtitle"
        return "body"

    def extract_titles(self, zones: List[LayoutZone]) -> List[str]:
        """Return all text content from title zones."""
        titles = []
        for zone in zones:
            if zone.region_type in ("title", "subtitle"):
                for region in zone.texts:
                    titles.append(region.text)
        return titles

    def extract_tickers(self, zones: List[LayoutZone]) -> List[str]:
        """Return all ticker/lower-third text content."""
        tickers = []
        for zone in zones:
            if zone.region_type == "ticker":
                for region in zone.texts:
                    tickers.append(region.text)
        return tickers

    def build_reading_order(self, zones: List[LayoutZone]) -> List[str]:
        """Return all text in reading order (top-to-bottom, left-to-right)."""
        all_regions = [(z.bbox[1], z.bbox[0], r.text) for z in zones for r in z.texts]
        all_regions.sort()
        return [text for _, _, text in all_regions]
