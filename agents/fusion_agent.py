"""Fusion agent: Cross-modal transformer fusion of visual, audio, and text."""
import time

import torch

from agents.state import VideoQAState
from utils.logger import get_logger

logger = get_logger("aurora.agents.fusion")


def fusion_node(state: VideoQAState) -> VideoQAState:
    """
    Node 5 — Fusion:
    1. CrossModalTransformer forward pass
    2. Temporal alignment (audio ↔ frames)
    3. Modality gate computation
    4. Unified fused representation
    """
    state.setdefault("pipeline_trace", [])
    state["pipeline_trace"].append("fusion_node")

    t0 = time.time()
    logger.info("Fusion node started")

    try:
        from fusion.cross_modal_transformer import CrossModalTransformer

        fusion_model = CrossModalTransformer()
        fusion_model.eval()

        clip_emb = state.get("clip_embeddings")
        ts_emb = state.get("timesformer_embedding")
        transcription = state.get("transcription")

        if clip_emb is not None:
            visual_feat = clip_emb.mean(dim=0, keepdim=True)
        elif ts_emb is not None:
            visual_feat = ts_emb
        else:
            visual_feat = torch.zeros(1, 768)

        if transcription is not None and transcription.text:
            from visual.clip_encoder import CLIPFrameEncoder

            clip_enc = CLIPFrameEncoder()
            text_feat = clip_enc.encode_text([transcription.text[:512]])
        else:
            text_feat = torch.zeros(1, 768)

        if ts_emb is not None:
            audio_feat = ts_emb[:, :512] if ts_emb.shape[-1] >= 512 else torch.zeros(1, 512)
        else:
            audio_feat = torch.zeros(1, 512)

        missing = []
        if state.get("transcription") is None:
            missing.append("text")
        if not state.get("audio_path"):
            missing.append("audio")

        for mod in missing:
            fusion_model.handle_missing_modality(mod)

        with torch.no_grad():
            fused_repr = fusion_model(visual_feat, audio_feat, text_feat)

        state["fused_representation"] = fused_repr.fused
        state["modality_weights"] = fused_repr.gate_weights

        logger.info(
            f"Fusion complete: shape={fused_repr.fused.shape}, "
            f"gates=visual:{fused_repr.gate_weights.get('visual', 0):.3f} "
            f"audio:{fused_repr.gate_weights.get('audio', 0):.3f} "
            f"text:{fused_repr.gate_weights.get('text', 0):.3f}"
        )

    except Exception as e:
        logger.error(f"Fusion failed: {e}")
        state["fused_representation"] = torch.zeros(1, 512)
        state["modality_weights"] = {"visual": 1.0, "audio": 0.0, "text": 0.0}

    elapsed = (time.time() - t0) * 1000
    state.setdefault("metadata", {})["fusion_latency_ms"] = round(elapsed, 1)
    logger.info(f"Fusion node complete in {elapsed:.1f}ms")
    return state
