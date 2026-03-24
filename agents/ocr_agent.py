"""OCR agent: TrOCR text extraction, text persistence tracking, layout analysis."""
import time
from typing import List

import numpy as np

from agents.state import VideoQAState
from utils.logger import get_logger

logger = get_logger("aurora.agents.ocr")


def ocr_node(state: VideoQAState) -> VideoQAState:
    """
    Node 4 — OCR:
    1. TrOCR text extraction per frame
    2. Text persistence tracking
    3. Layout analysis
    """
    state.setdefault("pipeline_trace", [])
    state["pipeline_trace"].append("ocr_node")

    t0 = time.time()
    logger.info("OCR node started")

    frames: List[np.ndarray] = state.get("frames", [])

    if not frames:
        logger.warning("No frames for OCR")
        state["ocr_results"] = []
        state["layout_zones"] = []
        return state

    try:
        from ocr.trocr_engine import TrOCREngine

        engine = TrOCREngine()
        ocr_results = engine.extract_text_batch(frames)
        state["ocr_results"] = ocr_results

        persistence = engine.track_text_across_frames(ocr_results)
        state.setdefault("metadata", {})["persistent_texts"] = list(
            persistence.get("persistent", {}).keys()
        )

        n_total = sum(len(r) for r in ocr_results)
        logger.info(
            f"OCR: {n_total} text regions across {len(frames)} frames, "
            f"{len(persistence.get('persistent', {}))} persistent"
        )
    except Exception as e:
        logger.error(f"TrOCR extraction failed: {e}")
        state["ocr_results"] = []

    try:
        from ocr.layout_analyzer import LayoutAnalyzer

        analyzer = LayoutAnalyzer()
        all_zones = []
        ocr_results = state.get("ocr_results", [])

        for frame_idx, (frame, regions) in enumerate(zip(frames[:8], ocr_results[:8])):
            zones = analyzer.analyze_frame(frame, regions)
            all_zones.append([
                {
                    "zone_id": z.zone_id,
                    "region_type": z.region_type,
                    "text_count": len(z.texts),
                }
                for z in zones
            ])

        state["layout_zones"] = all_zones
    except Exception as e:
        logger.warning(f"Layout analysis failed: {e}")
        state["layout_zones"] = []

    elapsed = (time.time() - t0) * 1000
    state.setdefault("metadata", {})["ocr_latency_ms"] = round(elapsed, 1)
    logger.info(f"OCR node complete in {elapsed:.1f}ms")
    return state
