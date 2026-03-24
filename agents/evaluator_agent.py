"""Evaluator agent: BLEU + ROUGE + METEOR + BERTScore quality assessment."""
import time

from agents.state import VideoQAState
from utils.logger import get_logger

logger = get_logger("aurora.agents.evaluator")

QUALITY_THRESHOLD = 0.6


def evaluator_node(state: VideoQAState) -> VideoQAState:
    """
    Node 7 — Evaluator:
    1. BLEU-4 + ROUGE-L + METEOR + BERTScore
    2. Modality contribution analysis
    3. Quality flag: high/medium/low
    """
    state.setdefault("pipeline_trace", [])
    state["pipeline_trace"].append("evaluator_node")

    t0 = time.time()
    logger.info("Evaluator node started")

    answer = state.get("generated_answer", "")
    query = state.get("query", "")
    transcription = state.get("transcription")

    reference = ""
    if transcription and transcription.text:
        reference = transcription.text[:500]
    elif query:
        reference = query

    if not answer or not reference:
        state["evaluation_scores"] = {
            "bleu": 0.0,
            "rouge": 0.0,
            "meteor": 0.0,
            "bertscore": 0.0,
            "aggregate": 0.0,
        }
        return state

    try:
        from evaluation.caption_evaluator import MultiModalEvaluator

        evaluator = MultiModalEvaluator()
        report = evaluator.full_evaluation([answer], [reference])
        state["evaluation_scores"] = report.scores
        logger.info(f"Evaluation scores: {report.scores}")
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
        state["evaluation_scores"] = {
            "bleu": 0.0,
            "rouge": 0.0,
            "meteor": 0.0,
            "bertscore": 0.0,
            "aggregate": 0.0,
        }

    aggregate = state["evaluation_scores"].get("aggregate", 0.0)
    if aggregate >= QUALITY_THRESHOLD:
        state.setdefault("metadata", {})["quality_flag"] = "high"
    elif aggregate >= 0.4:
        state.setdefault("metadata", {})["quality_flag"] = "medium"
    else:
        state.setdefault("metadata", {})["quality_flag"] = "low"

    elapsed = (time.time() - t0) * 1000
    state.setdefault("metadata", {})["evaluator_latency_ms"] = round(elapsed, 1)
    logger.info(f"Evaluator node complete in {elapsed:.1f}ms, quality={state['metadata']['quality_flag']}")
    return state
