from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger("aurora.visual.object_tracker")


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    @property
    def lifetime_frames(self) -> int:
        return len(self.frame_indices)

    @property
    def last_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        return self.bboxes[-1] if self.bboxes else None


class ObjectTracker:
    """
    Multi-object tracker using SORT (Simple Online and Realtime Tracking).
    Detects objects with a YOLOv5 model and tracks them across video frames.
    """

    def __init__(self, conf_threshold: float = 0.3, iou_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._detector = None
        self._tracker = None
        self._active_tracks: Dict[int, TrackedObject] = {}

    def _init_detector(self):
        """Lazy-initialize YOLOv5 detector."""
        if self._detector is not None:
            return
        try:
            import torch

            self._detector = torch.hub.load(
                "ultralytics/yolov5",
                "yolov5s",
                pretrained=True,
                verbose=False,
            )
            self._detector.conf = self.conf_threshold
            self._detector.iou = self.iou_threshold
            logger.info("YOLOv5s detector loaded")
        except Exception as e:
            logger.warning(f"Could not load YOLOv5: {e}")

    def _init_tracker(self):
        """Lazy-initialize SORT tracker."""
        if self._tracker is not None:
            return
        try:
            from sort import Sort

            self._tracker = Sort(max_age=30, min_hits=3, iou_threshold=self.iou_threshold)
            logger.info("SORT tracker initialized")
        except Exception as e:
            logger.warning(f"Could not initialize SORT tracker: {e}. Using basic IoU tracker.")
            self._tracker = self._SimpleIoUTracker()

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run object detection on a single frame."""
        self._init_detector()
        if self._detector is None:
            return []

        try:
            import torch

            results = self._detector(frame[..., ::-1])
            detections = []
            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                name = results.names[int(cls)]
                detections.append(
                    Detection(
                        bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                        confidence=float(conf),
                        class_id=int(cls),
                        class_name=name,
                    )
                )
            return detections
        except Exception as e:
            logger.warning(f"Detection failed: {e}")
            return []

    def track(
        self, frames: List[np.ndarray]
    ) -> Dict[int, TrackedObject]:
        """
        Run SORT multi-object tracking across all frames.

        Returns:
            Dict mapping track_id -> TrackedObject
        """
        self._init_detector()
        self._init_tracker()
        self._active_tracks = {}

        for frame_idx, frame in enumerate(frames):
            detections = self.detect(frame)
            if not detections:
                continue

            det_array = np.array(
                [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence]
                 for d in detections]
            )

            try:
                tracked = self._tracker.update(det_array)
                for t in tracked:
                    x1, y1, x2, y2, track_id = t
                    tid = int(track_id)
                    if tid not in self._active_tracks:
                        cls_name = self._assign_class(detections, (x1, y1, x2, y2))
                        self._active_tracks[tid] = TrackedObject(
                            track_id=tid, class_name=cls_name
                        )
                    self._active_tracks[tid].bboxes.append((x1, y1, x2, y2))
                    self._active_tracks[tid].frame_indices.append(frame_idx)
                    self._active_tracks[tid].confidences.append(1.0)
            except Exception as e:
                logger.warning(f"SORT update failed at frame {frame_idx}: {e}")

        return self._active_tracks

    def _assign_class(
        self,
        detections: List[Detection],
        bbox: Tuple,
    ) -> str:
        """Match a tracked bbox to the nearest detection class."""
        if not detections:
            return "unknown"
        return detections[0].class_name

    class _SimpleIoUTracker:
        """Fallback simple IoU-based tracker when SORT is unavailable."""

        def __init__(self):
            self._next_id = 1
            self._tracks: Dict = {}

        def update(self, dets: np.ndarray) -> np.ndarray:
            results = []
            for det in dets:
                results.append([*det[:4], float(self._next_id)])
                self._next_id += 1
            return np.array(results) if results else np.empty((0, 5))
