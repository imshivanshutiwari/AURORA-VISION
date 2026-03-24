"""
AURORA-VISION LangGraph: 7-node multi-modal QA pipeline.

Graph structure:
    ingestor → [visual, audio, ocr] (parallel)
    visual → fusion
    audio → fusion
    ocr → fusion
    fusion → qa
    qa → evaluator
    evaluator → END (quality high)
    evaluator → qa (retry if quality low, max 2 retries)
"""
from typing import Literal

from langgraph.graph import END, StateGraph

from agents.audio_agent import audio_node
from agents.evaluator_agent import evaluator_node
from agents.fusion_agent import fusion_node
from agents.ingestor_agent import ingestor_node
from agents.ocr_agent import ocr_node
from agents.qa_agent import qa_node
from agents.state import VideoQAState
from agents.visual_agent import visual_node
from utils.logger import get_logger

logger = get_logger("aurora.agents.graph")

MAX_RETRIES = 2


def _should_retry(state: VideoQAState) -> Literal["qa_node", "__end__"]:
    """Conditional edge: retry QA if quality is low and retries remain."""
    scores = state.get("evaluation_scores", {})
    aggregate = scores.get("aggregate", 0.0)
    retry_count = state.get("retry_count", 0)
    quality_flag = state.get("metadata", {}).get("quality_flag", "low")

    if quality_flag == "high" or aggregate >= 0.6 or retry_count >= MAX_RETRIES:
        logger.info(f"Pipeline complete. Quality={quality_flag}, aggregate={aggregate:.3f}")
        return END

    logger.info(f"Quality low ({aggregate:.3f}), retrying QA (attempt {retry_count + 1})")
    return "qa_node"


def _increment_retry(state: VideoQAState) -> VideoQAState:
    """Increment retry counter before QA retry."""
    state["retry_count"] = state.get("retry_count", 0) + 1
    return state


class AURORARGraph:
    """
    LangGraph StateGraph implementing the 7-node AURORA-VISION pipeline.

    Nodes:
        1. ingestor_node  — video download + decoding
        2. visual_node    — CLIP + Swin + TimeSformer
        3. audio_node     — Whisper + diarization + sentiment
        4. ocr_node       — TrOCR text extraction
        5. fusion_node    — CrossModalTransformer fusion
        6. qa_node        — multi-modal Q&A generation
        7. evaluator_node — BLEU/ROUGE/METEOR/BERTScore

    Parallel: visual, audio, ocr run in parallel after ingestor.
    Conditional: evaluator → qa retry if quality < threshold.
    """

    def __init__(self):
        self.graph = None
        self._build_graph()

    def _build_graph(self) -> None:
        """Build the LangGraph StateGraph with all 7 nodes and edges."""
        builder = StateGraph(VideoQAState)

        builder.add_node("ingestor_node", ingestor_node)
        builder.add_node("visual_node", visual_node)
        builder.add_node("audio_node", audio_node)
        builder.add_node("ocr_node", ocr_node)
        builder.add_node("fusion_node", fusion_node)
        builder.add_node("qa_node", qa_node)
        builder.add_node("evaluator_node", evaluator_node)

        builder.set_entry_point("ingestor_node")

        builder.add_edge("ingestor_node", "visual_node")
        builder.add_edge("ingestor_node", "audio_node")
        builder.add_edge("ingestor_node", "ocr_node")

        builder.add_edge("visual_node", "fusion_node")
        builder.add_edge("audio_node", "fusion_node")
        builder.add_edge("ocr_node", "fusion_node")

        builder.add_edge("fusion_node", "qa_node")
        builder.add_edge("qa_node", "evaluator_node")

        builder.add_conditional_edges(
            "evaluator_node",
            _should_retry,
            {
                "qa_node": "qa_node",
                END: END,
            },
        )

        self.graph = builder.compile()
        logger.info("AURORA-VISION LangGraph compiled: 7 nodes, parallel visual/audio/ocr")

    def run(self, initial_state: VideoQAState) -> VideoQAState:
        """Execute the full pipeline on an initial state."""
        initial_state.setdefault("retry_count", 0)
        initial_state.setdefault("pipeline_trace", [])
        initial_state.setdefault("evaluation_scores", {})
        initial_state.setdefault("modality_weights", {})

        logger.info(
            f"Starting AURORA-VISION pipeline for: {initial_state.get('video_url', 'local file')}"
        )

        result = self.graph.invoke(initial_state)
        logger.info(
            f"Pipeline complete. Trace: {result.get('pipeline_trace', [])}"
        )
        return result

    def stream(self, initial_state: VideoQAState):
        """Stream pipeline execution, yielding state after each node."""
        initial_state.setdefault("retry_count", 0)
        initial_state.setdefault("pipeline_trace", [])
        for chunk in self.graph.stream(initial_state):
            yield chunk

    def get_node_names(self) -> list:
        return [
            "ingestor_node",
            "visual_node",
            "audio_node",
            "ocr_node",
            "fusion_node",
            "qa_node",
            "evaluator_node",
        ]
