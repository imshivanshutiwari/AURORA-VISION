"""QA agent: Multi-modal Q&A generation using LLM with fused context."""
import os
import time

from agents.state import VideoQAState
from utils.logger import get_logger

logger = get_logger("aurora.agents.qa")


def qa_node(state: VideoQAState) -> VideoQAState:
    """
    Node 6 — QA:
    1. Build multi-modal context prompt
    2. Call LLM (OpenAI GPT-4o) with fused representation + text context
    3. Extract citations from all modalities
    """
    state.setdefault("pipeline_trace", [])
    state["pipeline_trace"].append("qa_node")

    t0 = time.time()
    logger.info("QA node started")

    query = state.get("query", "Describe what is happening in this video.")

    context_parts = []

    transcription = state.get("transcription")
    if transcription and transcription.text:
        context_parts.append(f"[AUDIO TRANSCRIPT]\n{transcription.text[:2000]}")

    ocr_results = state.get("ocr_results", [])
    all_ocr_text = []
    for frame_regions in ocr_results:
        for region in frame_regions:
            if region.text.strip():
                all_ocr_text.append(region.text.strip())
    if all_ocr_text:
        unique_ocr = list(dict.fromkeys(all_ocr_text))[:20]
        context_parts.append(f"[ON-SCREEN TEXT]\n{'; '.join(unique_ocr)}")

    action_pred = state.get("action_prediction")
    if action_pred:
        context_parts.append(f"[ACTION CLASSIFICATION]\n{action_pred.label} (confidence: {action_pred.confidence:.2f})")

    keyframes = state.get("keyframes", [])
    if keyframes:
        context_parts.append(f"[VIDEO STRUCTURE]\n{len(keyframes)} scenes detected")

    sentiment = state.get("sentiment_result")
    if sentiment:
        context_parts.append(
            f"[SPEECH SENTIMENT]\n{sentiment.overall_sentiment} (score: {sentiment.overall_score:.2f})"
        )

    context = "\n\n".join(context_parts)
    prompt = f"""You are a multi-modal video understanding AI.
Analyze the following video evidence and answer the question.

{context}

QUESTION: {query}

Provide a detailed answer with citations to specific timestamps, 
on-screen text, or audio segments where relevant.
"""

    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            from langchain.schema import HumanMessage, SystemMessage

            llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=1024)
            messages = [
                SystemMessage(content="You are AURORA-VISION, an expert multi-modal video analyst."),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(messages)
            answer = response.content
            logger.info(f"LLM answer generated: {len(answer)} chars")
        except Exception as e:
            logger.warning(f"LLM call failed: {e}, using context-based answer")
            answer = _context_based_answer(query, context)
    else:
        logger.info("No OpenAI API key, generating context-based answer")
        answer = _context_based_answer(query, context)

    state["generated_answer"] = answer

    elapsed = (time.time() - t0) * 1000
    state.setdefault("metadata", {})["qa_latency_ms"] = round(elapsed, 1)
    logger.info(f"QA node complete in {elapsed:.1f}ms")
    return state


def _context_based_answer(query: str, context: str) -> str:
    """Generate an answer from context without an LLM API call."""
    lines = [l.strip() for l in context.split("\n") if l.strip() and not l.startswith("[")]
    if lines:
        summary = " ".join(lines[:5])
        return f"Based on the video analysis: {summary}"
    return f"The video content relates to: {query}"
