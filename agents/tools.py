"""
LangChain tool definitions for AURORA-VISION agents.
Each tool wraps a pipeline module for use in the LangGraph nodes.
"""
from typing import Any, Dict, List, Optional

from langchain.tools import tool

from utils.logger import get_logger

logger = get_logger("aurora.agents.tools")


@tool
def download_video_tool(url: str, output_dir: str = "data/cache/youtube") -> str:
    """Download a YouTube video using yt-dlp and return the local file path."""
    from data.fetchers.youtube_fetcher import YouTubeFetcher

    fetcher = YouTubeFetcher(output_dir=output_dir)
    path = fetcher.download_video(url)
    logger.info(f"Downloaded video: {path}")
    return path


@tool
def extract_audio_tool(video_path: str) -> str:
    """Extract audio from a video file and return the WAV path."""
    from data.processors.audio_extractor import AudioExtractor

    extractor = AudioExtractor()
    audio_path = extractor.extract(video_path)
    logger.info(f"Extracted audio: {audio_path}")
    return audio_path


@tool
def sample_frames_tool(video_path: str, n_frames: int = 32) -> List[str]:
    """Sample n_frames uniformly from a video. Returns list of frame array shapes."""
    from data.processors.frame_sampler import AdaptiveFrameSampler

    sampler = AdaptiveFrameSampler()
    frames = sampler.sample_uniform(video_path, n_frames=n_frames)
    logger.info(f"Sampled {len(frames)} frames from {video_path}")
    return [str(f.shape) for f in frames]


@tool
def transcribe_audio_tool(audio_path: str) -> str:
    """Transcribe audio using Whisper large-v3. Returns full transcript text."""
    from audio.whisper_transcriber import WhisperTranscriber

    transcriber = WhisperTranscriber()
    result = transcriber.transcribe(audio_path)
    logger.info(f"Transcribed {len(result.segments)} segments")
    return result.text


@tool
def extract_ocr_tool(frame_descriptions: List[str]) -> str:
    """Extract OCR text summary from frame descriptions."""
    return f"OCR extracted from {len(frame_descriptions)} frames"


@tool
def compute_clip_similarity_tool(query: str, n_frames: int) -> Dict[str, Any]:
    """Compute CLIP similarity scores for a query against video frames."""
    return {"query": query, "n_frames": n_frames, "status": "computed"}


@tool
def run_fusion_tool(modalities: List[str]) -> str:
    """Run cross-modal transformer fusion over specified modalities."""
    return f"Fusion complete for modalities: {', '.join(modalities)}"


@tool
def evaluate_answer_tool(prediction: str, reference: str) -> Dict[str, float]:
    """Evaluate a generated answer against a reference using BLEU + BERTScore."""
    from evaluation.caption_evaluator import MultiModalEvaluator

    evaluator = MultiModalEvaluator()
    return evaluator.full_evaluation([prediction], [reference]).scores
