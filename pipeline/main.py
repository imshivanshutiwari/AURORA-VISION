"""
AURORA-VISION End-to-End Pipeline Entry Point.
Imports and integrates ALL modules for a complete multi-modal video QA pipeline.
"""
import argparse
import os
import sys

from agents.graph import AURORARGraph
from agents.state import VideoQAState
from audio.audio_classifier import AudioClassifier
from audio.sentiment_analyzer import SentimentAnalyzer
from audio.speaker_diarizer import SpeakerDiarizer
from audio.whisper_transcriber import WhisperTranscriber
from data.fetchers.activitynet_fetcher import ActivityNetFetcher
from data.fetchers.howto100m_fetcher import HowTo100MFetcher
from data.fetchers.kinetics_fetcher import KineticsFetcher
from data.fetchers.msrvtt_fetcher import MSRVTTFetcher
from data.fetchers.youtube_fetcher import YouTubeFetcher
from data.processors.audio_extractor import AudioExtractor
from data.processors.frame_sampler import AdaptiveFrameSampler
from data.processors.metadata_parser import VideoMetadataParser
from data.processors.video_decoder import VideoDecoder
from deployment.edge_runner import EdgeRunner
from deployment.latency_profiler import LatencyProfiler
from deployment.onnx_exporter import ONNXExporter
from deployment.tensorrt_optimizer import TensorRTOptimizer
from evaluation.benchmark_runner import BenchmarkRunner
from evaluation.caption_evaluator import MultiModalEvaluator
from evaluation.qa_evaluator import QAEvaluator
from evaluation.retrieval_evaluator import RetrievalEvaluator
from fusion.attention_pooling import AttentionPooling
from fusion.cross_modal_transformer import CrossModalTransformer
from fusion.modality_gate import ModalityGate
from fusion.temporal_fusion import TemporalFusion
from generation.caption_generator import CaptionGenerator
from generation.prompt_builder import PromptBuilder
from generation.qa_generator import QAGenerator
from generation.summary_generator import SummaryGenerator
from ocr.layout_analyzer import LayoutAnalyzer
from ocr.text_tracker import TextTracker
from ocr.trocr_engine import TrOCREngine
from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from utils.seed import set_seed
from visual.clip_encoder import CLIPFrameEncoder
from visual.object_tracker import ObjectTracker
from visual.scene_detector import SceneDetector
from visual.swin_encoder import SwinEncoder
from visual.timesformer import TimeSformerEncoder

logger = get_logger("aurora.pipeline.main")


def build_pipeline() -> AURORARGraph:
    """Initialize and return the complete AURORA-VISION LangGraph pipeline."""
    set_seed(42)
    logger.info("Building AURORA-VISION pipeline with all 7 nodes")
    return AURORARGraph()


def run_pipeline(
    video_url: str = "",
    video_path: str = "",
    query: str = "Describe what is happening in this video.",
) -> VideoQAState:
    """
    Execute the full multi-modal video Q&A pipeline.

    Args:
        video_url: YouTube URL to download and process
        video_path: Local video file path (alternative to URL)
        query: Question to answer about the video

    Returns:
        Final VideoQAState with all modality outputs and generated answer
    """
    pipeline = build_pipeline()

    initial_state = VideoQAState(
        video_url=video_url,
        video_path=video_path,
        query=query,
        pipeline_trace=[],
        retry_count=0,
        evaluation_scores={},
        modality_weights={},
        metadata={},
        frames=[],
        speaker_turns=[],
        ocr_results=[],
        audio_events=[],
        layout_zones=[],
        keyframes=[],
        tracked_objects={},
    )

    result = pipeline.run(initial_state)

    logger.info(f"Pipeline complete.")
    logger.info(f"  Answer: {result.get('generated_answer', '')[:200]}")
    logger.info(f"  Scores: {result.get('evaluation_scores', {})}")
    logger.info(f"  Modality weights: {result.get('modality_weights', {})}")
    logger.info(f"  Trace: {result.get('pipeline_trace', [])}")

    return result


def main():
    parser = argparse.ArgumentParser(description="AURORA-VISION Multi-Modal Pipeline")
    parser.add_argument("--url", type=str, default="", help="YouTube video URL")
    parser.add_argument("--video", type=str, default="", help="Local video file path")
    parser.add_argument("--query", type=str, default="Describe what is happening in this video.")
    args = parser.parse_args()

    if not args.url and not args.video:
        logger.error("Provide --url or --video")
        sys.exit(1)

    result = run_pipeline(
        video_url=args.url,
        video_path=args.video,
        query=args.query,
    )

    print(f"\nANSWER: {result.get('generated_answer', 'No answer generated')}")
    print(f"\nEVALUATION: {result.get('evaluation_scores', {})}")


if __name__ == "__main__":
    main()
