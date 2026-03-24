from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
from scenedetect import ContentDetector, SceneManager, open_video

from data.processors.video_decoder import VideoDecoder
from utils.logger import get_logger

logger = get_logger("aurora.data.frame_sampler")


@dataclass
class KeyFrame:
    frame_idx: int
    timestamp: float
    scene_id: int
    is_boundary: bool
    frame: Optional[np.ndarray] = None


class AdaptiveFrameSampler:
    """
    Two-strategy frame sampler:
    1. Uniform: sample every N frames from the video
    2. Scene-Change Adaptive: detect scene boundaries via PySceneDetect
       ContentDetector and sample at transitions + uniform fill
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        scene_threshold: float = 30.0,
    ):
        self.target_size = target_size
        self.scene_threshold = scene_threshold
        self.decoder = VideoDecoder(target_size=target_size)

    def decode_video(self, path: str, target_fps: float = 1.0) -> Iterator[np.ndarray]:
        """Yield frames as numpy arrays using FFmpeg pipe."""
        yield from self.decoder.decode_frames(path, target_fps=target_fps)

    def sample_uniform(self, video_path: str, n_frames: int = 32) -> List[np.ndarray]:
        """Sample exactly n_frames uniformly spaced frames from the video."""
        info = self.decoder.get_video_info(video_path)
        duration = info.get("duration", 0.0)
        if duration <= 0:
            logger.warning(f"Could not determine duration for {video_path}")
            return []

        timestamps = np.linspace(0, duration, n_frames, endpoint=False)
        frames = []
        for ts in timestamps:
            frame = self.decoder.extract_frame_at(video_path, float(ts))
            if frame is not None:
                frames.append(frame)

        if len(frames) < n_frames:
            logger.warning(
                f"Only got {len(frames)}/{n_frames} frames from {video_path}"
            )

        return frames

    def sample_adaptive(
        self,
        video_path: str,
        min_frames: int = 16,
        max_frames: int = 64,
    ) -> List[np.ndarray]:
        """
        Detect scene changes and sample at boundaries + uniformly fill to
        ensure between min_frames and max_frames are returned.
        """
        scene_list = self._detect_scenes(video_path)
        info = self.decoder.get_video_info(video_path)
        duration = info.get("duration", 0.0)

        boundary_timestamps = []
        for start_tc, end_tc in scene_list:
            boundary_timestamps.append(start_tc.get_seconds())

        if not boundary_timestamps:
            logger.info(f"No scene changes detected, falling back to uniform sampling")
            return self.sample_uniform(video_path, n_frames=min_frames)

        all_timestamps = sorted(set(boundary_timestamps))

        if len(all_timestamps) < min_frames:
            extra = np.linspace(0, duration, min_frames, endpoint=False)
            combined = sorted(set(all_timestamps) | set(extra.tolist()))
            all_timestamps = combined[:max_frames]
        else:
            all_timestamps = all_timestamps[:max_frames]

        frames = []
        for ts in all_timestamps:
            frame = self.decoder.extract_frame_at(video_path, ts)
            if frame is not None:
                frames.append(frame)

        logger.info(
            f"Adaptive sampling: {len(scene_list)} scenes, {len(frames)} frames sampled"
        )
        return frames

    def extract_keyframes(self, video_path: str) -> List[KeyFrame]:
        """Extract keyframes with metadata: scene_id, is_boundary, timestamp."""
        scene_list = self._detect_scenes(video_path)
        info = self.decoder.get_video_info(video_path)
        fps = info.get("fps", 25.0)

        keyframes = []
        for scene_id, (start_tc, end_tc) in enumerate(scene_list):
            start_sec = start_tc.get_seconds()
            frame_idx = int(start_sec * fps)
            frame = self.decoder.extract_frame_at(video_path, start_sec)

            keyframes.append(
                KeyFrame(
                    frame_idx=frame_idx,
                    timestamp=start_sec,
                    scene_id=scene_id,
                    is_boundary=True,
                    frame=frame,
                )
            )

        return keyframes

    def _detect_scenes(self, video_path: str) -> list:
        """Run PySceneDetect ContentDetector to find scene boundaries."""
        try:
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=self.scene_threshold))
            scene_manager.detect_scenes(video, show_progress=False)
            return scene_manager.get_scene_list()
        except Exception as e:
            logger.warning(f"Scene detection failed for {video_path}: {e}")
            return []
