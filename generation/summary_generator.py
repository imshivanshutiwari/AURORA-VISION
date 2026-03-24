import os
from dataclasses import dataclass
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger("aurora.generation.summary")


@dataclass
class VideoSummary:
    brief: str
    detailed: str
    bullet_points: List[str]
    key_moments: List[dict]
    word_count: int


class SummaryGenerator:
    """
    Multi-level video summarization:
    - Brief: 1-2 sentence executive summary
    - Detailed: paragraph-level description
    - Bullet points: key events list
    - Key moments: timestamped highlights
    """

    def __init__(self):
        self.llm = None
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            try:
                from langchain_openai import ChatOpenAI

                self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=2048)
            except Exception as e:
                logger.warning(f"Could not init summary LLM: {e}")

    def summarize(
        self,
        transcript: Optional[str] = None,
        ocr_texts: Optional[List[str]] = None,
        action_label: Optional[str] = None,
        keyframe_timestamps: Optional[List[float]] = None,
        duration: float = 0.0,
    ) -> VideoSummary:
        """Generate a multi-level summary of video content."""
        context_parts = []
        if transcript:
            context_parts.append(f"Transcript: {transcript[:2000]}")
        if ocr_texts:
            context_parts.append(f"On-screen text: {'; '.join(ocr_texts[:15])}")
        if action_label:
            context_parts.append(f"Primary action: {action_label}")
        if duration:
            context_parts.append(f"Duration: {duration:.1f}s")

        context = "\n".join(context_parts)

        if self.llm and context:
            try:
                return self._llm_summarize(context, keyframe_timestamps, duration)
            except Exception as e:
                logger.warning(f"LLM summary failed: {e}")

        return self._rule_based_summary(context, keyframe_timestamps, duration)

    def _llm_summarize(
        self,
        context: str,
        timestamps: Optional[List[float]],
        duration: float,
    ) -> VideoSummary:
        from langchain.schema import HumanMessage

        prompt = f"""Summarize this video content:

{context}

Generate:
1. BRIEF: One sentence summary
2. DETAILED: 2-3 paragraph description  
3. BULLETS: 5 key points (one per line starting with -)
4. MOMENTS: 3 key moments with timestamps if available
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        text = response.content

        lines = text.split("\n")
        brief = next((l.replace("BRIEF:", "").strip() for l in lines if "BRIEF:" in l), text[:100])
        detailed = "\n".join(lines[:10])
        bullets = [l[1:].strip() for l in lines if l.strip().startswith("-")][:5]

        key_moments = []
        if timestamps:
            for i, ts in enumerate(timestamps[:3]):
                key_moments.append({"timestamp": ts, "description": f"Key moment {i + 1}"})

        return VideoSummary(
            brief=brief,
            detailed=detailed,
            bullet_points=bullets,
            key_moments=key_moments,
            word_count=len(text.split()),
        )

    def _rule_based_summary(
        self,
        context: str,
        timestamps: Optional[List[float]],
        duration: float,
    ) -> VideoSummary:
        brief = context[:150] if context else "A multi-modal video."
        bullets = [s.strip() for s in context.split(".") if s.strip()][:5]
        key_moments = [
            {"timestamp": ts, "description": f"Scene {i + 1}"}
            for i, ts in enumerate((timestamps or [])[:3])
        ]
        return VideoSummary(
            brief=brief,
            detailed=context[:500],
            bullet_points=bullets,
            key_moments=key_moments,
            word_count=len(context.split()),
        )
