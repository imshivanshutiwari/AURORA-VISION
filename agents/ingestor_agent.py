"""Ingestor agent: Downloads and decodes video, extracts audio and frames."""
import time

from agents.state import VideoQAState
from utils.logger import get_logger

logger = get_logger("aurora.agents.ingestor")


def ingestor_node(state: VideoQAState) -> VideoQAState:
    """
    Node 1 — Ingestor:
    1. Download video via yt-dlp (or use existing path)
    2. Extract audio via FFmpeg
    3. Sample frames adaptively (scene-change + uniform)
    4. Extract video metadata
    """
    state.setdefault("pipeline_trace", [])
    state.setdefault("metadata", {})
    state["pipeline_trace"].append("ingestor_node")

    t0 = time.time()
    logger.info("Ingestor node started")

    video_url = state.get("video_url", "")
    video_path = state.get("video_path", "")

    if not video_path and video_url:
        from data.fetchers.youtube_fetcher import YouTubeFetcher

        fetcher = YouTubeFetcher()
        video_path = fetcher.download_video(video_url)
        state["video_path"] = video_path
        logger.info(f"Downloaded video: {video_path}")

    if video_path:
        from data.processors.audio_extractor import AudioExtractor
        from data.processors.frame_sampler import AdaptiveFrameSampler
        from data.processors.metadata_parser import VideoMetadataParser

        audio_extractor = AudioExtractor()
        try:
            audio_path = audio_extractor.extract(video_path)
            state["audio_path"] = audio_path
        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}")
            state["audio_path"] = ""

        sampler = AdaptiveFrameSampler()
        try:
            frames = sampler.sample_adaptive(video_path)
            state["frames"] = frames
            logger.info(f"Sampled {len(frames)} frames adaptively")
        except Exception as e:
            logger.warning(f"Adaptive sampling failed, using uniform: {e}")
            try:
                frames = sampler.sample_uniform(video_path, n_frames=32)
                state["frames"] = frames
            except Exception as e2:
                logger.error(f"Frame sampling failed: {e2}")
                state["frames"] = []

        parser = VideoMetadataParser()
        try:
            meta = parser.parse(video_path)
            state["metadata"] = {
                "duration": meta.duration,
                "fps": meta.fps,
                "width": meta.width,
                "height": meta.height,
                "n_frames": meta.n_frames,
                "codec": meta.codec,
                "has_audio": meta.has_audio,
            }
        except Exception as e:
            logger.warning(f"Metadata parsing failed: {e}")

    elapsed = (time.time() - t0) * 1000
    state["metadata"]["ingestor_latency_ms"] = round(elapsed, 1)
    logger.info(f"Ingestor complete in {elapsed:.1f}ms")

    return state
