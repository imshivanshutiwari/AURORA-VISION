"""Audio agent: Whisper transcription, speaker diarization, event classification, sentiment."""
import time

from agents.state import VideoQAState
from utils.logger import get_logger

logger = get_logger("aurora.agents.audio")


def audio_node(state: VideoQAState) -> VideoQAState:
    """
    Node 3 — Audio:
    1. Whisper-large-v3 transcription
    2. Speaker diarization (pyannote)
    3. Audio event classification
    4. Speech sentiment analysis
    """
    state.setdefault("pipeline_trace", [])
    state["pipeline_trace"].append("audio_node")

    t0 = time.time()
    logger.info("Audio node started")

    audio_path: str = state.get("audio_path", "")

    if not audio_path:
        logger.warning("No audio path, skipping audio processing")
        state["transcription"] = None
        state["speaker_turns"] = []
        state["audio_events"] = []
        state["sentiment_result"] = None
        return state

    try:
        from audio.whisper_transcriber import WhisperTranscriber

        transcriber = WhisperTranscriber()
        transcription = transcriber.transcribe(audio_path)
        state["transcription"] = transcription
        logger.info(
            f"Transcribed {len(transcription.segments)} segments, lang={transcription.language}"
        )
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        state["transcription"] = None

    try:
        from audio.speaker_diarizer import SpeakerDiarizer

        diarizer = SpeakerDiarizer()
        diarization = diarizer.diarize(audio_path)

        if state.get("transcription") and diarization.turns:
            speaker_segs = diarizer.assign_transcription_to_speakers(
                state["transcription"].segments, diarization.turns
            )
            state["speaker_turns"] = speaker_segs
        else:
            state["speaker_turns"] = []

        logger.info(f"Diarization: {diarization.n_speakers} speakers")
    except Exception as e:
        logger.warning(f"Diarization failed: {e}")
        state["speaker_turns"] = []

    try:
        from audio.audio_classifier import AudioClassifier

        classifier = AudioClassifier()
        events = classifier.classify_file(audio_path)
        state["audio_events"] = events
        logger.info(f"Audio events: {len(events)} classified")
    except Exception as e:
        logger.warning(f"Audio classification failed: {e}")
        state["audio_events"] = []

    if state.get("transcription"):
        try:
            from audio.sentiment_analyzer import SentimentAnalyzer

            analyzer = SentimentAnalyzer()
            sentiment = analyzer.analyze_transcription(state["transcription"].segments)
            state["sentiment_result"] = sentiment
            logger.info(f"Sentiment: {sentiment.overall_sentiment} ({sentiment.overall_score:.2f})")
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            state["sentiment_result"] = None

    elapsed = (time.time() - t0) * 1000
    state.setdefault("metadata", {})["audio_latency_ms"] = round(elapsed, 1)
    logger.info(f"Audio node complete in {elapsed:.1f}ms")
    return state
