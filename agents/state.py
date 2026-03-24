from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

import numpy as np
import torch


class VideoQAState(TypedDict, total=False):
    """Complete pipeline state for multi-modal video Q&A."""

    video_url: str
    video_path: str
    frames: List[Any]
    audio_path: str
    transcription: Any
    speaker_turns: List[Any]
    ocr_results: List[List[Any]]
    clip_embeddings: Any
    swin_features: Any
    timesformer_embedding: Any
    fused_representation: Any
    query: str
    generated_answer: str
    evaluation_scores: Dict[str, float]
    pipeline_trace: List[str]
    modality_weights: Dict[str, float]
    scene_list: List[Any]
    keyframes: List[Any]
    tracked_objects: Dict[int, Any]
    action_prediction: Any
    audio_events: List[Any]
    sentiment_result: Any
    layout_zones: List[List[Any]]
    retry_count: int
    error: Optional[str]
    metadata: Dict[str, Any]
