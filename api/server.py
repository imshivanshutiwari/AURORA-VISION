import os
from typing import List, Optional

import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    ModalityWeights,
    TranscriptionSegment,
    VideoProcessRequest,
    VideoProcessResponse,
)
from utils.logger import get_logger

logger = get_logger("aurora.api.server")

app = FastAPI(
    title="AURORA-VISION API",
    description="Multi-Modal Video Understanding System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and loaded models."""
    return HealthResponse(
        status="online",
        version="1.0.0",
        models_loaded=["CLIP-ViT-L/14", "Whisper-large-v3", "TrOCR-large", "CrossModalTransformer"],
        gpu_available=torch.cuda.is_available(),
    )


@app.post("/process", response_model=VideoProcessResponse)
async def process_video(request: VideoProcessRequest):
    """
    Process a video through the full AURORA-VISION pipeline.
    Returns transcription, OCR, embeddings, Q&A answer, and evaluation scores.
    """
    if not request.video_url and not request.video_path:
        raise HTTPException(status_code=400, detail="Provide video_url or video_path")

    try:
        from agents.graph import AURORARGraph
        from agents.state import VideoQAState

        pipeline = AURORARGraph()
        state = VideoQAState(
            video_url=request.video_url or "",
            video_path=request.video_path or "",
            query=request.query,
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

        result = pipeline.run(state)

        transcription = result.get("transcription")
        segments = []
        if transcription:
            speaker_turns = result.get("speaker_turns", [])
            for seg in (transcription.segments or [])[:50]:
                speaker = None
                for st in speaker_turns:
                    if hasattr(st, "start") and abs(st.start - seg.start) < 1.0:
                        speaker = getattr(st, "speaker_id", None)
                        break
                segments.append(
                    TranscriptionSegment(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                        speaker=speaker,
                        confidence=seg.confidence,
                    )
                )

        ocr_texts = []
        for frame_regions in (result.get("ocr_results") or []):
            for region in frame_regions:
                if hasattr(region, "text") and region.text.strip():
                    ocr_texts.append(region.text.strip())
        ocr_texts = list(dict.fromkeys(ocr_texts))[:30]

        scores = result.get("evaluation_scores", {})
        weights = result.get("modality_weights", {})

        action_pred = result.get("action_prediction")
        action_label = action_pred.label if action_pred else None

        return VideoProcessResponse(
            video_url=request.video_url or "",
            video_path=request.video_path or "",
            generated_answer=result.get("generated_answer", ""),
            transcription_text=transcription.text if transcription else "",
            transcription_segments=segments,
            ocr_texts=ocr_texts,
            action_label=action_label,
            evaluation_scores={"bleu": scores.get("bleu", 0), "rouge": scores.get("rouge", 0), "meteor": scores.get("meteor", 0), "bertscore": scores.get("bertscore", 0), "aggregate": scores.get("aggregate", 0)},
            modality_weights=ModalityWeights(
                visual=weights.get("visual", 0.33),
                audio=weights.get("audio", 0.33),
                text=weights.get("text", 0.34),
            ),
            pipeline_trace=result.get("pipeline_trace", []),
            n_frames_processed=len(result.get("frames") or []),
            n_scenes=len(result.get("scene_list") or []),
            metadata=result.get("metadata", {}),
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_captions(request: EvaluationRequest):
    """Evaluate generated captions using BLEU/ROUGE/METEOR/BERTScore."""
    try:
        from evaluation.caption_evaluator import MultiModalEvaluator

        evaluator = MultiModalEvaluator()
        report = evaluator.full_evaluation(request.predictions, request.references)
        scores = report.scores

        return EvaluationResponse(
            bleu=scores.get("bleu", 0.0),
            rouge_l=scores.get("rouge", 0.0),
            meteor=scores.get("meteor", 0.0),
            bertscore=scores.get("bertscore", 0.0),
            aggregate=scores.get("aggregate", 0.0),
            n_samples=report.n_samples,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
