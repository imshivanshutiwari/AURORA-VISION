import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import ffmpeg

from utils.logger import get_logger

logger = get_logger("aurora.data.metadata_parser")


@dataclass
class VideoMetadata:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    n_frames: int
    codec: str
    has_audio: bool
    audio_sample_rate: int
    file_size_mb: float
    format_name: str


class VideoMetadataParser:
    """Parses video file metadata using FFmpeg probe."""

    def parse(self, video_path: str) -> VideoMetadata:
        """Extract comprehensive metadata from a video file."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        try:
            probe = ffmpeg.probe(str(path))
        except Exception as e:
            raise RuntimeError(f"FFmpeg probe failed for {video_path}: {e}") from e

        fmt = probe.get("format", {})
        streams = probe.get("streams", [])

        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

        vs = video_streams[0] if video_streams else {}
        aud = audio_streams[0] if audio_streams else {}

        fps_str = vs.get("r_frame_rate", "25/1")
        try:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if int(den) != 0 else 25.0
        except Exception:
            fps = 25.0

        duration = float(fmt.get("duration", vs.get("duration", 0)))
        n_frames = int(vs.get("nb_frames", 0)) or int(fps * duration)

        return VideoMetadata(
            path=str(path),
            duration=duration,
            width=int(vs.get("width", 0)),
            height=int(vs.get("height", 0)),
            fps=fps,
            n_frames=n_frames,
            codec=vs.get("codec_name", "unknown"),
            has_audio=bool(audio_streams),
            audio_sample_rate=int(aud.get("sample_rate", 0)),
            file_size_mb=float(fmt.get("size", 0)) / 1024 / 1024,
            format_name=fmt.get("format_name", "unknown"),
        )

    def parse_batch(self, video_paths: List[str]) -> List[VideoMetadata]:
        """Parse metadata for multiple video files."""
        results = []
        for path in video_paths:
            try:
                results.append(self.parse(path))
            except Exception as e:
                logger.warning(f"Failed to parse {path}: {e}")
        return results

    def to_dict(self, metadata: VideoMetadata) -> Dict:
        """Convert VideoMetadata to a plain dict."""
        return {
            "path": metadata.path,
            "duration": metadata.duration,
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "n_frames": metadata.n_frames,
            "codec": metadata.codec,
            "has_audio": metadata.has_audio,
            "audio_sample_rate": metadata.audio_sample_rate,
            "file_size_mb": round(metadata.file_size_mb, 2),
            "format_name": metadata.format_name,
        }


def main():
    parser = argparse.ArgumentParser(description="Fetch and cache real video metadata")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Trigger Kinetics + ActivityNet annotation download",
    )
    args = parser.parse_args()

    if args.fetch:
        logger.info("Fetching Kinetics-400 annotations...")
        try:
            from data.fetchers.kinetics_fetcher import KineticsFetcher

            kf = KineticsFetcher()
            df = kf.download_annotations()
            logger.info(f"Kinetics-400: {len(df)} annotations cached")
        except Exception as e:
            logger.error(f"Kinetics fetch failed: {e}")

        logger.info("Fetching ActivityNet annotations...")
        try:
            from data.fetchers.activitynet_fetcher import ActivityNetFetcher

            af = ActivityNetFetcher()
            ann = af.fetch_annotations()
            logger.info(f"ActivityNet: {len(ann)} videos cached")
        except Exception as e:
            logger.error(f"ActivityNet fetch failed: {e}")


if __name__ == "__main__":
    main()
