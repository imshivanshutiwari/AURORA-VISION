from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VideoProcessRequest(BaseModel):
    video_url: Optional[str] = Field(None, description="YouTube URL to process")
    video_path: Optional[str] = Field(None, description="Local video file path")
    query: str = Field("Describe what is happening in this video.", description="Question to answer")
    n_frames: int = Field(32, ge=1, le=128, description="Number of frames to sample")


class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: float = 1.0


class EvaluationScores(BaseModel):
    bleu: float = 0.0
    rouge: float = 0.0
    meteor: float = 0.0
    bertscore: float = 0.0
    aggregate: float = 0.0


class ModalityWeights(BaseModel):
    visual: float = 0.33
    audio: float = 0.33
    text: float = 0.34


class VideoProcessResponse(BaseModel):
    video_url: str = ""
    video_path: str = ""
    generated_answer: str = ""
    transcription_text: str = ""
    transcription_segments: List[TranscriptionSegment] = Field(default_factory=list)
    ocr_texts: List[str] = Field(default_factory=list)
    action_label: Optional[str] = None
    evaluation_scores: EvaluationScores = Field(default_factory=EvaluationScores)
    modality_weights: ModalityWeights = Field(default_factory=ModalityWeights)
    pipeline_trace: List[str] = Field(default_factory=list)
    n_frames_processed: int = 0
    n_scenes: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "online"
    version: str = "1.0.0"
    models_loaded: List[str] = Field(default_factory=list)
    gpu_available: bool = False


class EvaluationRequest(BaseModel):
    predictions: List[str]
    references: List[str]


class EvaluationResponse(BaseModel):
    bleu: float
    rouge_l: float
    meteor: float
    bertscore: float
    aggregate: float
    n_samples: int
