from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import ffmpeg
import numpy as np

from utils.logger import get_logger

logger = get_logger("aurora.data.video_decoder")


class VideoDecoder:
    """Decode video frames using FFmpeg, returning numpy arrays."""

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size

    def get_video_info(self, video_path: str) -> dict:
        """Probe video file to get duration, fps, resolution."""
        try:
            probe = ffmpeg.probe(video_path)
            video_streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
            if not video_streams:
                return {}
            vs = video_streams[0]
            fps_str = vs.get("r_frame_rate", "25/1")
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if int(den) != 0 else 25.0
            return {
                "width": int(vs.get("width", 0)),
                "height": int(vs.get("height", 0)),
                "fps": fps,
                "duration": float(probe.get("format", {}).get("duration", 0)),
                "n_frames": int(vs.get("nb_frames", 0)) or int(fps * float(probe.get("format", {}).get("duration", 0))),
                "codec": vs.get("codec_name", ""),
            }
        except Exception as e:
            logger.warning(f"Could not probe video {video_path}: {e}")
            return {}

    def decode_frames(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        target_fps: float = 1.0,
    ) -> Iterator[np.ndarray]:
        """Yield decoded RGB frames as (H, W, 3) numpy arrays."""
        info = self.get_video_info(video_path)
        duration = info.get("duration", 0)
        if end_sec is None:
            end_sec = duration

        w, h = self.target_size
        try:
            input_kwargs = {"ss": start_sec}
            if end_sec and end_sec > start_sec:
                input_kwargs["t"] = end_sec - start_sec

            stream = (
                ffmpeg.input(video_path, **input_kwargs)
                .filter("fps", fps=target_fps)
                .filter("scale", w, h)
                .output("pipe:", format="rawvideo", pix_fmt="rgb24")
                .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
            )

            frame_size = h * w * 3
            while True:
                raw = stream.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                yield frame

            stream.wait()
        except Exception as e:
            logger.error(f"Frame decoding failed for {video_path}: {e}")

    def decode_all_frames(
        self,
        video_path: str,
        target_fps: float = 1.0,
    ) -> List[np.ndarray]:
        """Decode all frames into a list of numpy arrays."""
        return list(self.decode_frames(video_path, target_fps=target_fps))

    def extract_frame_at(self, video_path: str, timestamp_sec: float) -> Optional[np.ndarray]:
        """Extract a single frame at a specific timestamp."""
        frames = list(self.decode_frames(video_path, start_sec=timestamp_sec, end_sec=timestamp_sec + 1.0))
        return frames[0] if frames else None
