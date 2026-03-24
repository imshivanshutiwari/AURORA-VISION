"""Tests for audio/ modules: WhisperTranscriber, SpeakerDiarizer, AudioClassifier, SentimentAnalyzer."""
import unittest

import numpy as np


class TestWhisperDataclasses(unittest.TestCase):
    """Tests for Whisper transcription dataclasses."""

    def test_word_timestamp_fields(self):
        """WordTimestamp stores word, start, end, probability correctly."""
        from audio.whisper_transcriber import WordTimestamp

        wt = WordTimestamp(word="hello", start=0.0, end=0.5, probability=0.99)
        self.assertEqual(wt.word, "hello")
        self.assertAlmostEqual(wt.start, 0.0)
        self.assertAlmostEqual(wt.end, 0.5)
        self.assertAlmostEqual(wt.probability, 0.99)

    def test_segment_fields(self):
        """Segment stores start, end, text, and words correctly."""
        from audio.whisper_transcriber import Segment, WordTimestamp

        words = [WordTimestamp(word="hi", start=1.0, end=1.3, probability=0.95)]
        seg = Segment(start=1.0, end=2.0, text="hi there", words=words, confidence=0.95)
        self.assertEqual(seg.text, "hi there")
        self.assertEqual(len(seg.words), 1)
        self.assertAlmostEqual(seg.confidence, 0.95)

    def test_transcription_result_fields(self):
        """TranscriptionResult stores all required fields."""
        from audio.whisper_transcriber import Segment, TranscriptionResult, WordTimestamp

        seg = Segment(start=0.0, end=1.0, text="test", words=[])
        result = TranscriptionResult(
            text="test",
            segments=[seg],
            words_with_timestamps=[],
            language="en",
            duration=1.0,
        )
        self.assertEqual(result.text, "test")
        self.assertEqual(result.language, "en")
        self.assertAlmostEqual(result.duration, 1.0)


class TestSpeakerDiarizationDataclasses(unittest.TestCase):
    """Tests for SpeakerDiarizer dataclasses."""

    def test_turn_duration(self):
        """Turn.duration computes end - start correctly."""
        from audio.speaker_diarizer import Turn

        turn = Turn(start=5.0, end=8.5, speaker_id="SPEAKER_00")
        self.assertAlmostEqual(turn.duration, 3.5)

    def test_diarization_result_n_speakers(self):
        """DiarizationResult stores correct speaker count."""
        from audio.speaker_diarizer import DiarizationResult, Turn

        turns = [
            Turn(start=0.0, end=2.0, speaker_id="SPEAKER_00"),
            Turn(start=2.5, end=4.0, speaker_id="SPEAKER_01"),
        ]
        result = DiarizationResult(
            turns=turns,
            n_speakers=2,
            speaker_ids=["SPEAKER_00", "SPEAKER_01"],
        )
        self.assertEqual(result.n_speakers, 2)
        self.assertEqual(len(result.speaker_ids), 2)


class TestAudioClassifierDataclasses(unittest.TestCase):
    """Tests for AudioClassifier dataclasses."""

    def test_audio_event_fields(self):
        """AudioEvent stores label, confidence, start_sec, end_sec."""
        from audio.audio_classifier import AudioEvent

        event = AudioEvent(label="speech", confidence=0.92, start_sec=0.0, end_sec=3.0)
        self.assertEqual(event.label, "speech")
        self.assertAlmostEqual(event.confidence, 0.92)
        self.assertAlmostEqual(event.end_sec, 3.0)

    def test_audio_classification_result_dominant(self):
        """AudioClassificationResult stores dominant_event correctly."""
        from audio.audio_classifier import AudioClassificationResult, AudioEvent

        events = [AudioEvent(label="music", confidence=0.88, start_sec=0.0, end_sec=5.0)]
        result = AudioClassificationResult(
            events=events,
            dominant_event="music",
            event_distribution={"music": 0.88, "speech": 0.12},
        )
        self.assertEqual(result.dominant_event, "music")
        self.assertAlmostEqual(result.event_distribution["music"], 0.88)


class TestSentimentDataclasses(unittest.TestCase):
    """Tests for SentimentAnalyzer dataclasses."""

    def test_sentiment_result_ratios_sum_to_one(self):
        """SentimentResult positive + negative + neutral ratios sum to 1.0."""
        from audio.sentiment_analyzer import SentimentResult

        result = SentimentResult(
            segments=[],
            overall_sentiment="positive",
            overall_score=0.75,
            positive_ratio=0.6,
            negative_ratio=0.1,
            neutral_ratio=0.3,
        )
        total = result.positive_ratio + result.negative_ratio + result.neutral_ratio
        self.assertAlmostEqual(total, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
