from dataclasses import dataclass, field
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger("aurora.audio.whisper")


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: List[WordTimestamp] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class TranscriptionResult:
    text: str
    segments: List[Segment]
    words_with_timestamps: List[WordTimestamp]
    language: str
    duration: float = 0.0


class WhisperTranscriber:
    """
    Whisper-large-v3 speech-to-text transcription with word-level timestamps.
    Uses the OpenAI Whisper library for maximum accuracy.
    """

    def __init__(self, device: Optional[str] = None):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load Whisper large-v3 model."""
        import whisper

        logger.info("Loading Whisper large-v3...")
        self.model = whisper.load_model("large-v3", device=self.device)
        logger.info(f"Whisper large-v3 loaded on {self.device}")

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe audio file with word-level timestamps.

        Args:
            audio_path: Path to audio file (WAV/MP3/etc.)

        Returns:
            TranscriptionResult with text, segments, and word timestamps
        """
        logger.info(f"Transcribing: {audio_path}")
        result = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            language=None,
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=0.0,
        )

        segments = []
        all_words = []

        for seg in result.get("segments", []):
            words = []
            for w in seg.get("words", []):
                wt = WordTimestamp(
                    word=w.get("word", ""),
                    start=float(w.get("start", seg["start"])),
                    end=float(w.get("end", seg["end"])),
                    probability=float(w.get("probability", 1.0)),
                )
                words.append(wt)
                all_words.append(wt)

            segments.append(
                Segment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=seg["text"].strip(),
                    words=words,
                    confidence=float(
                        sum(w.probability for w in words) / len(words) if words else 1.0
                    ),
                )
            )

        duration = float(result.get("duration", segments[-1].end if segments else 0.0))

        return TranscriptionResult(
            text=result["text"].strip(),
            segments=segments,
            words_with_timestamps=all_words,
            language=result.get("language", "en"),
            duration=duration,
        )

    def transcribe_with_timestamps(self, audio_path: str) -> List[Segment]:
        """Convenience method returning only segments with timestamps."""
        result = self.transcribe(audio_path)
        return result.segments

    def translate(self, audio_path: str, target: str = "en") -> str:
        """Translate non-English audio to English using Whisper translate task."""
        result = self.model.transcribe(
            audio_path,
            task="translate",
            beam_size=5,
            best_of=5,
            temperature=0.0,
        )
        return result["text"].strip()
