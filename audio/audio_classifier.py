from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("aurora.audio.classifier")

AUDIO_EVENTS = [
    "speech",
    "music",
    "noise",
    "silence",
    "applause",
    "laughter",
    "crowd",
    "alarm",
    "vehicle",
    "nature",
]


@dataclass
class AudioEvent:
    label: str
    confidence: float
    start_sec: float
    end_sec: float


@dataclass
class AudioClassificationResult:
    events: List[AudioEvent]
    dominant_event: str
    event_distribution: Dict[str, float] = field(default_factory=dict)


class AudioClassifier:
    """
    Audio event classifier using a pretrained PANNs or wav2vec2-based model.
    Classifies audio segments into semantic event categories.
    """

    def __init__(self, device: Optional[str] = None):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self) -> None:
        """Load audio classification model from transformers hub."""
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            model_id = "facebook/wav2vec2-base"
            logger.info(f"Loading audio classifier: {model_id}")
            self.processor = AutoFeatureExtractor.from_pretrained(model_id)
            self.model = AutoModelForAudioClassification.from_pretrained(model_id)
            self.model = self.model.to(self.device).eval()
            logger.info("Audio classifier loaded")
        except Exception as e:
            logger.warning(f"Could not load audio classifier: {e}. Using rule-based fallback.")

    def classify_segment(
        self, audio_array: np.ndarray, sample_rate: int = 16000
    ) -> AudioClassificationResult:
        """
        Classify a raw audio segment.

        Args:
            audio_array: (N,) float32 numpy array
            sample_rate: audio sample rate in Hz

        Returns:
            AudioClassificationResult with event labels
        """
        if self.model is None or self.processor is None:
            return self._rule_based_classify(audio_array, sample_rate)

        try:
            import torch

            inputs = self.processor(
                audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            id2label = self.model.config.id2label

            events = []
            duration = len(audio_array) / sample_rate
            for i, (idx, prob) in enumerate(
                sorted(enumerate(probs), key=lambda x: -x[1])[:5]
            ):
                label = id2label.get(idx, f"event_{idx}")
                events.append(
                    AudioEvent(
                        label=label,
                        confidence=float(prob),
                        start_sec=0.0,
                        end_sec=duration,
                    )
                )

            dominant = events[0].label if events else "unknown"
            dist = {e.label: e.confidence for e in events}
            return AudioClassificationResult(
                events=events, dominant_event=dominant, event_distribution=dist
            )

        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return self._rule_based_classify(audio_array, sample_rate)

    def _rule_based_classify(
        self, audio_array: np.ndarray, sample_rate: int
    ) -> AudioClassificationResult:
        """Rule-based fallback: classify by RMS energy level."""
        rms = float(np.sqrt(np.mean(audio_array ** 2)))
        duration = len(audio_array) / sample_rate

        if rms < 0.01:
            label, conf = "silence", 0.95
        elif rms < 0.1:
            label, conf = "speech", 0.7
        else:
            label, conf = "noise", 0.6

        event = AudioEvent(label=label, confidence=conf, start_sec=0.0, end_sec=duration)
        return AudioClassificationResult(
            events=[event], dominant_event=label, event_distribution={label: conf}
        )

    def classify_file(
        self,
        audio_path: str,
        chunk_duration: float = 10.0,
    ) -> List[AudioEvent]:
        """Classify an entire audio file in chunks."""
        try:
            import librosa

            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            logger.warning(f"Could not load {audio_path}: {e}")
            return []

        chunk_samples = int(chunk_duration * sr)
        events = []

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i: i + chunk_samples]
            if len(chunk) == 0:
                continue
            result = self.classify_segment(chunk, sr)
            start_sec = i / sr
            for ev in result.events[:1]:
                events.append(
                    AudioEvent(
                        label=ev.label,
                        confidence=ev.confidence,
                        start_sec=start_sec,
                        end_sec=start_sec + chunk_duration,
                    )
                )

        return events
