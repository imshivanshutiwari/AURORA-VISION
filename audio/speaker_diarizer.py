import os
from dataclasses import dataclass, field
from typing import List, Optional

from audio.whisper_transcriber import Segment
from utils.logger import get_logger

logger = get_logger("aurora.audio.diarizer")


@dataclass
class Turn:
    start: float
    end: float
    speaker_id: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SpeakerSegment:
    start: float
    end: float
    text: str
    speaker_id: str
    confidence: float = 1.0


@dataclass
class DiarizationResult:
    turns: List[Turn]
    n_speakers: int
    speaker_ids: List[str]


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote.audio speaker-diarization-3.1.
    Assigns each speech segment to a unique speaker identity.
    """

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.environ.get("HUGGINGFACE_TOKEN", "")
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load pyannote speaker diarization pipeline."""
        if not self.hf_token:
            logger.warning(
                "No HuggingFace token found. Speaker diarization requires "
                "access to pyannote/speaker-diarization-3.1. "
                "Set HUGGINGFACE_TOKEN environment variable."
            )
            return

        try:
            from pyannote.audio import Pipeline

            logger.info("Loading pyannote speaker-diarization-3.1...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
            )
            logger.info("pyannote diarization pipeline loaded")
        except Exception as e:
            logger.warning(f"Could not load pyannote pipeline: {e}")

    def diarize(self, audio_path: str) -> DiarizationResult:
        """
        Run speaker diarization on an audio file.

        Returns:
            DiarizationResult with turn-level speaker assignments
        """
        if self.pipeline is None:
            logger.warning("Diarization pipeline not loaded, returning empty result")
            return DiarizationResult(turns=[], n_speakers=0, speaker_ids=[])

        try:
            diarization = self.pipeline(audio_path)
            turns = []
            speaker_ids = set()

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turns.append(
                    Turn(
                        start=turn.start,
                        end=turn.end,
                        speaker_id=str(speaker),
                    )
                )
                speaker_ids.add(str(speaker))

            turns = sorted(turns, key=lambda t: t.start)
            return DiarizationResult(
                turns=turns,
                n_speakers=len(speaker_ids),
                speaker_ids=sorted(speaker_ids),
            )
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return DiarizationResult(turns=[], n_speakers=0, speaker_ids=[])

    def assign_transcription_to_speakers(
        self,
        segments: List[Segment],
        turns: List[Turn],
    ) -> List[SpeakerSegment]:
        """
        Assign speaker labels to transcription segments by time overlap.

        Uses the speaker turn with maximum overlap with each segment.
        """
        speaker_segments = []

        for seg in segments:
            best_speaker = "SPEAKER_00"
            best_overlap = 0.0

            for turn in turns:
                overlap_start = max(seg.start, turn.start)
                overlap_end = min(seg.end, turn.end)
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn.speaker_id

            speaker_segments.append(
                SpeakerSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    speaker_id=best_speaker,
                    confidence=float(best_overlap / max(1e-6, seg.end - seg.start)),
                )
            )

        return speaker_segments
