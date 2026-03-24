import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("aurora.generation.caption")


@dataclass
class Caption:
    text: str
    confidence: float
    start_sec: float = 0.0
    end_sec: float = 0.0
    source_modalities: List[str] = field(default_factory=list)


class CaptionGenerator:
    """Generates dense video captions from multi-modal features using GPT-4o."""

    def __init__(self):
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            try:
                from langchain_openai import ChatOpenAI

                self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=512)
            except Exception as e:
                logger.warning(f"Could not init LLM: {e}")

    def generate_frame_caption(
        self,
        clip_similarity_scores: Dict[str, float],
        ocr_text: List[str],
        transcript_segment: Optional[str] = None,
    ) -> Caption:
        """Generate a caption for a single video frame using multi-modal cues."""
        context_parts = []

        if clip_similarity_scores:
            top = sorted(clip_similarity_scores.items(), key=lambda x: -x[1])[:3]
            context_parts.append("Visual scene: " + ", ".join(f"{k}({v:.2f})" for k, v in top))

        if ocr_text:
            context_parts.append(f"On-screen text: {'; '.join(ocr_text[:5])}")

        if transcript_segment:
            context_parts.append(f"Audio: {transcript_segment[:100]}")

        context = " | ".join(context_parts) or "video frame"

        if self.llm:
            try:
                from langchain.schema import HumanMessage

                msg = HumanMessage(content=f"Caption this video frame in one sentence: {context}")
                response = self.llm.invoke([msg])
                return Caption(
                    text=response.content.strip(),
                    confidence=0.9,
                    source_modalities=["visual", "audio", "ocr"],
                )
            except Exception as e:
                logger.warning(f"LLM caption failed: {e}")

        return Caption(
            text=f"A video frame showing: {context}",
            confidence=0.6,
            source_modalities=["visual"],
        )

    def generate_video_caption(
        self,
        frame_captions: List[Caption],
        transcript: Optional[str] = None,
        action_label: Optional[str] = None,
    ) -> str:
        """Generate a holistic video caption from frame-level captions."""
        summaries = [c.text for c in frame_captions[:10]]

        parts = []
        if action_label:
            parts.append(f"Activity: {action_label}.")
        if summaries:
            parts.append("Frames show: " + " → ".join(summaries[:5]))
        if transcript:
            parts.append(f"Audio: {transcript[:200]}")

        caption = " ".join(parts)

        if self.llm and caption:
            try:
                from langchain.schema import HumanMessage

                msg = HumanMessage(
                    content=f"Summarize this video in 2 sentences based on: {caption}"
                )
                response = self.llm.invoke([msg])
                return response.content.strip()
            except Exception as e:
                logger.warning(f"LLM video caption failed: {e}")

        return caption or "Multi-modal video content."
