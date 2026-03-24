from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ocr.trocr_engine import TextRegion
from utils.logger import get_logger

logger = get_logger("aurora.ocr.text_tracker")


@dataclass
class TrackedText:
    text: str
    first_frame: int
    last_frame: int
    frame_indices: List[int] = field(default_factory=list)
    bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    is_persistent: bool = False

    @property
    def duration_frames(self) -> int:
        return self.last_frame - self.first_frame + 1

    @property
    def appearance_count(self) -> int:
        return len(self.frame_indices)


class TextTracker:
    """
    Tracks text regions across video frames to identify persistent vs transient text.
    Uses normalized text similarity for matching across frames.
    """

    def __init__(self, persistence_threshold: int = 3):
        self.persistence_threshold = persistence_threshold
        self._tracked: Dict[str, TrackedText] = {}

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison (lowercase, strip punctuation)."""
        import re

        return re.sub(r"[^\w\s]", "", text.lower().strip())

    def update(self, frame_idx: int, regions: List[TextRegion]) -> None:
        """Update tracker with text detections from a new frame."""
        seen_texts: Set[str] = set()

        for region in regions:
            norm = self.normalize_text(region.text)
            if not norm:
                continue

            seen_texts.add(norm)

            if norm in self._tracked:
                tracked = self._tracked[norm]
                tracked.last_frame = frame_idx
                tracked.frame_indices.append(frame_idx)
                tracked.bboxes.append(region.bbox)
                if tracked.appearance_count >= self.persistence_threshold:
                    tracked.is_persistent = True
            else:
                self._tracked[norm] = TrackedText(
                    text=region.text,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    frame_indices=[frame_idx],
                    bboxes=[region.bbox],
                )

    def process_video(
        self, frame_texts: List[List[TextRegion]]
    ) -> Dict[str, TrackedText]:
        """Process all frames and return complete tracking results."""
        self._tracked = {}
        for frame_idx, regions in enumerate(frame_texts):
            self.update(frame_idx, regions)

        for tracked in self._tracked.values():
            if tracked.appearance_count >= self.persistence_threshold:
                tracked.is_persistent = True

        return self._tracked

    def get_persistent_texts(self) -> List[TrackedText]:
        """Return only texts that appear persistently (>= threshold frames)."""
        return [t for t in self._tracked.values() if t.is_persistent]

    def get_transient_texts(self) -> List[TrackedText]:
        """Return texts appearing fewer than threshold frames."""
        return [t for t in self._tracked.values() if not t.is_persistent]

    def get_text_timeline(self) -> Dict[int, List[str]]:
        """Build a frame_idx -> [text, ...] mapping."""
        timeline: Dict[int, List[str]] = defaultdict(list)
        for tracked in self._tracked.values():
            for frame_idx in tracked.frame_indices:
                timeline[frame_idx].append(tracked.text)
        return dict(timeline)
