"""Tests for ocr/ modules: TrOCREngine, TextTracker, LayoutAnalyzer."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestTextRegionDataclass(unittest.TestCase):
    """Tests for TextRegion dataclass."""

    def test_text_region_fields(self):
        """TextRegion stores bbox, text, confidence, frame_idx correctly."""
        from ocr.trocr_engine import TextRegion

        region = TextRegion(bbox=(10, 20, 100, 30), text="HELLO WORLD", confidence=0.95, frame_idx=5)
        self.assertEqual(region.text, "HELLO WORLD")
        self.assertEqual(region.bbox, (10, 20, 100, 30))
        self.assertAlmostEqual(region.confidence, 0.95)
        self.assertEqual(region.frame_idx, 5)


class TestTextTracker(unittest.TestCase):
    """Tests for TextTracker logic."""

    def test_normalize_text_removes_punctuation(self):
        """TextTracker.normalize_text strips punctuation and lowercases."""
        from ocr.text_tracker import TextTracker

        tracker = TextTracker()
        normalized = tracker.normalize_text("Hello, World!")
        self.assertEqual(normalized, "hello world")

    def test_update_marks_persistent_text(self):
        """TextTracker marks text as persistent after persistence_threshold frames."""
        from ocr.text_tracker import TextTracker
        from ocr.trocr_engine import TextRegion

        tracker = TextTracker(persistence_threshold=3)
        region = TextRegion(bbox=(0, 0, 100, 20), text="BREAKING NEWS", confidence=0.9)

        for i in range(4):
            tracker.update(frame_idx=i, regions=[region])

        persistent = tracker.get_persistent_texts()
        texts = [t.text for t in persistent]
        self.assertIn("BREAKING NEWS", texts)

    def test_tracked_text_duration_frames(self):
        """TrackedText.duration_frames computes last_frame - first_frame + 1."""
        from ocr.text_tracker import TrackedText

        tracked = TrackedText(
            text="Sample",
            first_frame=2,
            last_frame=8,
            frame_indices=[2, 3, 4, 5, 6, 7, 8],
        )
        self.assertEqual(tracked.duration_frames, 7)


class TestLayoutAnalyzer(unittest.TestCase):
    """Tests for LayoutAnalyzer spatial zone classification."""

    def test_analyze_frame_classifies_top_text_as_title(self):
        """LayoutAnalyzer classifies text in top 20% of frame as 'title'."""
        from ocr.layout_analyzer import LayoutAnalyzer
        from ocr.trocr_engine import TextRegion

        analyzer = LayoutAnalyzer()
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        # Text at top of frame (y=10, well within top 20% = y < 80)
        region = TextRegion(bbox=(100, 10, 200, 20), text="TITLE TEXT", confidence=0.9, frame_idx=0)
        zones = analyzer.analyze_frame(frame, [region])

        zone_types = [z.region_type for z in zones]
        self.assertIn("title", zone_types)


if __name__ == "__main__":
    unittest.main()
