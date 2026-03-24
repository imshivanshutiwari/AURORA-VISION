import os
from dataclasses import dataclass, field
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger("aurora.generation.qa")


@dataclass
class QAPair:
    question: str
    answer: str
    confidence: float
    source_modalities: List[str] = field(default_factory=list)
    timestamp_references: List[float] = field(default_factory=list)


class QAGenerator:
    """Generates question-answer pairs from multi-modal video content."""

    QUESTION_TEMPLATES = [
        "What is happening in the video?",
        "Who is speaking in this video?",
        "What text appears on screen?",
        "What is the main activity shown?",
        "What is the overall topic of this video?",
        "What does the narrator say about {}?",
        "At what point does {} occur in the video?",
    ]

    def __init__(self):
        self.llm = None
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            try:
                from langchain_openai import ChatOpenAI

                self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=1024)
            except Exception as e:
                logger.warning(f"Could not init QA LLM: {e}")

    def generate_qa_pairs(
        self,
        transcript: Optional[str] = None,
        ocr_texts: Optional[List[str]] = None,
        action_label: Optional[str] = None,
        n_pairs: int = 5,
    ) -> List[QAPair]:
        """Generate QA pairs from multi-modal video content."""
        context_parts = []
        if transcript:
            context_parts.append(f"Transcript: {transcript[:1000]}")
        if ocr_texts:
            context_parts.append(f"On-screen text: {'; '.join(ocr_texts[:10])}")
        if action_label:
            context_parts.append(f"Action: {action_label}")

        context = "\n".join(context_parts)

        if self.llm and context:
            try:
                from langchain.schema import HumanMessage

                prompt = (
                    f"Generate {n_pairs} diverse question-answer pairs about this video content.\n"
                    f"Format each as Q: ... A: ...\n\n{context}"
                )
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return self._parse_qa_response(response.content)
            except Exception as e:
                logger.warning(f"LLM QA generation failed: {e}")

        return self._template_qa(context, action_label, n_pairs)

    def _parse_qa_response(self, text: str) -> List[QAPair]:
        """Parse LLM response into QAPair objects."""
        pairs = []
        lines = text.strip().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Q:"):
                question = line[2:].strip()
                answer = ""
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("A:"):
                    answer = lines[i + 1][2:].strip()
                    i += 1
                if question and answer:
                    pairs.append(
                        QAPair(
                            question=question,
                            answer=answer,
                            confidence=0.85,
                            source_modalities=["visual", "audio", "ocr"],
                        )
                    )
            i += 1
        return pairs

    def _template_qa(
        self, context: str, action_label: Optional[str], n_pairs: int
    ) -> List[QAPair]:
        """Fallback template-based QA generation."""
        pairs = []
        templates = self.QUESTION_TEMPLATES[:n_pairs]
        for tpl in templates:
            question = tpl.format(action_label or "this event") if "{}" in tpl else tpl
            answer = f"Based on the video content: {context[:100]}..." if context else "Not available."
            pairs.append(
                QAPair(
                    question=question,
                    answer=answer,
                    confidence=0.5,
                    source_modalities=["template"],
                )
            )
        return pairs
