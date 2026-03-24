from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("aurora.generation.prompt_builder")


class PromptBuilder:
    """
    Context-aware prompt builder for multi-modal video Q&A.
    Assembles structured prompts from visual, audio, and OCR modalities.
    """

    SYSTEM_PROMPT = """You are AURORA-VISION, an expert multi-modal video understanding AI.
You analyze video content across three modalities:
- VISUAL: CLIP embeddings, Swin Transformer patch features, TimeSformer temporal features
- AUDIO: Whisper-large-v3 transcription with speaker diarization
- OCR: TrOCR on-screen text extraction with layout analysis

Answer questions with evidence from all relevant modalities.
Always cite your sources (timestamp, on-screen text, speaker ID).
"""

    def build_qa_prompt(
        self,
        query: str,
        transcript: Optional[str] = None,
        ocr_texts: Optional[List[str]] = None,
        action_label: Optional[str] = None,
        speaker_turns: Optional[List[Dict]] = None,
        clip_top_frames: Optional[List[Dict]] = None,
        sentiment: Optional[str] = None,
    ) -> str:
        """Build a structured multi-modal Q&A prompt."""
        sections = [self.SYSTEM_PROMPT, "\n--- VIDEO EVIDENCE ---\n"]

        if action_label:
            sections.append(f"ACTION CLASSIFICATION: {action_label}")

        if clip_top_frames:
            frame_strs = [
                f"Frame {f.get('frame_idx', i)}: sim={f.get('similarity', 0):.3f}"
                for i, f in enumerate(clip_top_frames[:5])
            ]
            sections.append(f"TOP RELEVANT FRAMES:\n" + "\n".join(frame_strs))

        if transcript:
            sections.append(f"\nAUDIO TRANSCRIPT:\n{transcript[:1500]}")

        if speaker_turns:
            turn_strs = [
                f"[{t.get('speaker_id', 'SPK')} @ {t.get('start', 0):.1f}s]: {t.get('text', '')[:80]}"
                for t in speaker_turns[:10]
            ]
            sections.append(f"\nSPEAKER TURNS:\n" + "\n".join(turn_strs))

        if ocr_texts:
            sections.append(f"\nON-SCREEN TEXT:\n" + " | ".join(ocr_texts[:15]))

        if sentiment:
            sections.append(f"\nSPEECH SENTIMENT: {sentiment}")

        sections.append(f"\n--- QUESTION ---\n{query}")
        sections.append("\n--- YOUR ANSWER ---")

        return "\n".join(sections)

    def build_summary_prompt(
        self,
        transcript: Optional[str] = None,
        ocr_texts: Optional[List[str]] = None,
        action_label: Optional[str] = None,
        duration: float = 0.0,
    ) -> str:
        """Build a video summarization prompt."""
        parts = ["Provide a comprehensive multi-modal video summary.\n"]

        if duration:
            parts.append(f"Video duration: {duration:.1f} seconds\n")
        if action_label:
            parts.append(f"Primary action: {action_label}\n")
        if transcript:
            parts.append(f"Full transcript:\n{transcript[:2000]}\n")
        if ocr_texts:
            parts.append(f"On-screen text: {'; '.join(ocr_texts[:20])}\n")

        parts.append("\nGenerate: (1) brief summary, (2) detailed summary, (3) 5 key points")
        return "".join(parts)

    def build_retrieval_prompt(self, query: str, n_candidates: int = 20) -> str:
        """Build a video retrieval ranking prompt."""
        return (
            f"Rank {n_candidates} video candidates by relevance to query: '{query}'\n"
            "For each candidate, score from 0.0 to 1.0 based on multi-modal similarity."
        )
