"""Visual agent: CLIP encoding, Swin features, TimeSformer, scene detection, tracking."""
import time
from typing import List

import numpy as np

from agents.state import VideoQAState
from utils.logger import get_logger

logger = get_logger("aurora.agents.visual")


def visual_node(state: VideoQAState) -> VideoQAState:
    """
    Node 2 — Visual:
    1. CLIP ViT-L/14 frame encoding
    2. Swin Transformer patch features
    3. TimeSformer temporal video embedding
    4. Scene detection + keyframe extraction
    5. Object tracking per frame
    """
    state.setdefault("pipeline_trace", [])
    state["pipeline_trace"].append("visual_node")

    t0 = time.time()
    logger.info("Visual node started")

    frames: List[np.ndarray] = state.get("frames", [])
    video_path: str = state.get("video_path", "")

    if not frames:
        logger.warning("No frames available for visual processing")
        state["clip_embeddings"] = None
        state["swin_features"] = None
        state["timesformer_embedding"] = None
        return state

    try:
        from visual.clip_encoder import CLIPFrameEncoder

        clip_encoder = CLIPFrameEncoder()
        clip_embeddings = clip_encoder.encode_frames(frames)
        state["clip_embeddings"] = clip_embeddings
        logger.info(f"CLIP embeddings: {clip_embeddings.shape}")
    except Exception as e:
        logger.error(f"CLIP encoding failed: {e}")
        state["clip_embeddings"] = None

    try:
        from visual.swin_encoder import SwinEncoder

        swin = SwinEncoder()
        swin_features = swin.encode_batch(frames[:8])
        state["swin_features"] = swin_features
        logger.info(f"Swin features: {swin_features.shape}")
    except Exception as e:
        logger.error(f"Swin encoding failed: {e}")
        state["swin_features"] = None

    try:
        from visual.timesformer import TimeSformerEncoder

        tsformer = TimeSformerEncoder()
        ts_emb = tsformer.encode_video_clip(frames[:8])
        state["timesformer_embedding"] = ts_emb
        action_pred = tsformer.classify_action(frames[:8])
        state["action_prediction"] = action_pred
        logger.info(f"TimeSformer embedding: {ts_emb.shape}, action: {action_pred.label}")
    except Exception as e:
        logger.error(f"TimeSformer failed: {e}")
        state["timesformer_embedding"] = None
        state["action_prediction"] = None

    if video_path:
        try:
            from visual.scene_detector import SceneDetector

            detector = SceneDetector()
            scenes = detector.detect_scenes(video_path)
            state["scene_list"] = scenes
            state["keyframes"] = [
                {"scene_id": s.scene_id, "start_sec": s.start_sec, "end_sec": s.end_sec}
                for s in scenes
            ]
            logger.info(f"Detected {len(scenes)} scenes")
        except Exception as e:
            logger.warning(f"Scene detection failed: {e}")
            state["scene_list"] = []
            state["keyframes"] = []

    try:
        from visual.object_tracker import ObjectTracker

        tracker = ObjectTracker()
        tracked = tracker.track(frames[:8])
        state["tracked_objects"] = {
            tid: {"class_name": obj.class_name, "lifetime": obj.lifetime_frames}
            for tid, obj in tracked.items()
        }
        logger.info(f"Tracked {len(tracked)} objects")
    except Exception as e:
        logger.warning(f"Object tracking failed: {e}")
        state["tracked_objects"] = {}

    elapsed = (time.time() - t0) * 1000
    state.setdefault("metadata", {})["visual_latency_ms"] = round(elapsed, 1)
    logger.info(f"Visual node complete in {elapsed:.1f}ms")
    return state
