from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from data.processors.video_decoder import VideoDecoder
from utils.logger import get_logger

logger = get_logger("aurora.visual.scene_detector")


@dataclass
class Scene:
    scene_id: int
    start_sec: float
    end_sec: float
    duration: float
    start_frame: int
    end_frame: int
    representative_frame: Optional[np.ndarray] = None


class SceneDetector:
    """
    Detects scene changes in video using PySceneDetect ContentDetector.
    Returns rich scene metadata including representative frames.
    """

    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold
        self.decoder = VideoDecoder()

    def detect_scenes(self, video_path: str) -> List[Scene]:
        """
        Run content-aware scene detection on a video file.

        Returns:
            List of Scene objects with temporal boundaries
        """
        from scenedetect import ContentDetector, SceneManager, open_video

        try:
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=self.threshold))
            scene_manager.detect_scenes(video, show_progress=False)
            scene_list = scene_manager.get_scene_list()
        except Exception as e:
            logger.error(f"Scene detection failed for {video_path}: {e}")
            return []

        info = self.decoder.get_video_info(video_path)
        fps = info.get("fps", 25.0)
        scenes = []

        for i, (start_tc, end_tc) in enumerate(scene_list):
            start_sec = start_tc.get_seconds()
            end_sec = end_tc.get_seconds()

            rep_frame = self.decoder.extract_frame_at(
                video_path, (start_sec + end_sec) / 2.0
            )

            scenes.append(
                Scene(
                    scene_id=i,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration=end_sec - start_sec,
                    start_frame=int(start_sec * fps),
                    end_frame=int(end_sec * fps),
                    representative_frame=rep_frame,
                )
            )

        logger.info(f"Detected {len(scenes)} scenes in {video_path}")
        return scenes

    def get_scene_boundaries(self, video_path: str) -> List[float]:
        """Return list of scene boundary timestamps in seconds."""
        scenes = self.detect_scenes(video_path)
        return [s.start_sec for s in scenes]

    def sample_per_scene(
        self, video_path: str, n_per_scene: int = 1
    ) -> List[Tuple[int, float, np.ndarray]]:
        """Sample n_per_scene frames from each scene. Returns (scene_id, ts, frame)."""
        scenes = self.detect_scenes(video_path)
        results = []

        for scene in scenes:
            timestamps = np.linspace(
                scene.start_sec, scene.end_sec, n_per_scene + 2
            )[1:-1]

            for ts in timestamps:
                frame = self.decoder.extract_frame_at(video_path, float(ts))
                if frame is not None:
                    results.append((scene.scene_id, float(ts), frame))

        return results
