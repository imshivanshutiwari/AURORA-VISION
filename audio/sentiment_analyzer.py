from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from audio.whisper_transcriber import Segment
from utils.logger import get_logger

logger = get_logger("aurora.audio.sentiment")


@dataclass
class SentimentScore:
    label: str
    score: float
    segment_text: str
    start: float
    end: float


@dataclass
class SentimentResult:
    segments: List[SentimentScore]
    overall_sentiment: str
    overall_score: float
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float


class SentimentAnalyzer:
    """
    Speech sentiment analysis on transcription segments.
    Uses a RoBERTa-based sentiment classifier from HuggingFace.
    """

    def __init__(self, device: Optional[str] = None):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load sentiment analysis pipeline."""
        try:
            from transformers import pipeline as hf_pipeline

            logger.info("Loading sentiment analysis pipeline (cardiffnlp/twitter-roberta-base-sentiment-latest)...")
            self.pipeline = hf_pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1,
                truncation=True,
                max_length=512,
            )
            logger.info("Sentiment pipeline loaded")
        except Exception as e:
            logger.warning(f"Could not load sentiment pipeline: {e}")

    def analyze_segment(self, text: str, start: float = 0.0, end: float = 1.0) -> SentimentScore:
        """Analyze sentiment of a single text segment."""
        if not text.strip():
            return SentimentScore(
                label="neutral", score=0.5, segment_text=text, start=start, end=end
            )

        if self.pipeline is None:
            return self._lexicon_fallback(text, start, end)

        try:
            result = self.pipeline(text[:512])[0]
            return SentimentScore(
                label=result["label"].lower(),
                score=float(result["score"]),
                segment_text=text,
                start=start,
                end=end,
            )
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return self._lexicon_fallback(text, start, end)

    def analyze_transcription(self, segments: List[Segment]) -> SentimentResult:
        """Analyze sentiment across all transcription segments."""
        scored_segments = [
            self.analyze_segment(seg.text, seg.start, seg.end) for seg in segments
        ]

        if not scored_segments:
            return SentimentResult(
                segments=[],
                overall_sentiment="neutral",
                overall_score=0.5,
                positive_ratio=0.0,
                negative_ratio=0.0,
                neutral_ratio=1.0,
            )

        positive = [s for s in scored_segments if "positive" in s.label or "pos" in s.label]
        negative = [s for s in scored_segments if "negative" in s.label or "neg" in s.label]
        neutral = [s for s in scored_segments if "neutral" in s.label]
        total = len(scored_segments)

        pos_ratio = len(positive) / total
        neg_ratio = len(negative) / total
        neu_ratio = len(neutral) / total

        if pos_ratio >= neg_ratio and pos_ratio >= neu_ratio:
            overall = "positive"
            overall_score = pos_ratio
        elif neg_ratio >= pos_ratio and neg_ratio >= neu_ratio:
            overall = "negative"
            overall_score = neg_ratio
        else:
            overall = "neutral"
            overall_score = neu_ratio

        return SentimentResult(
            segments=scored_segments,
            overall_sentiment=overall,
            overall_score=float(overall_score),
            positive_ratio=float(pos_ratio),
            negative_ratio=float(neg_ratio),
            neutral_ratio=float(neu_ratio),
        )

    def _lexicon_fallback(self, text: str, start: float, end: float) -> SentimentScore:
        """Simple lexicon-based fallback sentiment analysis."""
        positive_words = {"good", "great", "excellent", "amazing", "wonderful", "happy", "love"}
        negative_words = {"bad", "terrible", "awful", "horrible", "hate", "sad", "wrong"}

        words = set(text.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)

        if pos_count > neg_count:
            return SentimentScore(label="positive", score=0.7, segment_text=text, start=start, end=end)
        elif neg_count > pos_count:
            return SentimentScore(label="negative", score=0.7, segment_text=text, start=start, end=end)
        else:
            return SentimentScore(label="neutral", score=0.5, segment_text=text, start=start, end=end)
