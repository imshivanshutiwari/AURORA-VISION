from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from utils.logger import get_logger

logger = get_logger("aurora.data.activitynet")

CAPTIONS_URL = (
    "https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.json"
)


@dataclass
class Caption:
    start_time: float
    end_time: float
    sentence: str


@dataclass
class QAPair:
    question: str
    answer: str
    start_time: float
    end_time: float


class ActivityNetFetcher:
    """Fetches real ActivityNet Captions dataset for dense video captioning."""

    CAPTIONS_URL = CAPTIONS_URL

    def __init__(self, cache_dir: str = "data/cache/activitynet"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._annotations: Optional[Dict] = None

    def fetch_annotations(self) -> Dict:
        """Download ActivityNet Captions annotations from Stanford."""
        cache_file = self.cache_dir / "captions.json"

        if cache_file.exists():
            logger.info(f"Loading cached ActivityNet annotations from {cache_file}")
            import json

            with open(cache_file) as f:
                self._annotations = json.load(f)
            return self._annotations

        logger.info(f"Downloading ActivityNet Captions from {self.CAPTIONS_URL}")
        try:
            resp = requests.get(self.CAPTIONS_URL, timeout=120)
            resp.raise_for_status()
            self._annotations = resp.json()

            import json

            with open(cache_file, "w") as f:
                json.dump(self._annotations, f)

            logger.info(
                f"Downloaded ActivityNet: {len(self._annotations)} videos annotated"
            )
        except Exception as e:
            logger.error(f"Failed to download ActivityNet annotations: {e}")
            raise

        return self._annotations

    def fetch_video(self, video_id: str) -> str:
        """Download a video by its ActivityNet ID via yt-dlp."""
        from data.fetchers.youtube_fetcher import YouTubeFetcher

        output_path = self.cache_dir / f"{video_id}.mp4"
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path)

        fetcher = YouTubeFetcher(output_dir=str(self.cache_dir))
        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            path = fetcher.download_video(url)
            return path or ""
        except Exception as e:
            logger.warning(f"Could not fetch video {video_id}: {e}")
            return ""

    def get_temporal_captions(self, video_id: str) -> List[Caption]:
        """Get ordered temporal captions for a specific video."""
        if self._annotations is None:
            self.fetch_annotations()

        video_data = self._annotations.get(video_id, {})
        timestamps = video_data.get("timestamps", [])
        sentences = video_data.get("sentences", [])

        captions = []
        for (start, end), sentence in zip(timestamps, sentences):
            captions.append(Caption(start_time=float(start), end_time=float(end), sentence=sentence))

        return sorted(captions, key=lambda c: c.start_time)

    def build_qa_pairs(self, video_id: str) -> List[QAPair]:
        """Build question-answer pairs from temporal captions."""
        captions = self.get_temporal_captions(video_id)
        qa_pairs = []

        for i, cap in enumerate(captions):
            question = f"What is happening at {cap.start_time:.1f}s in the video?"
            qa_pairs.append(
                QAPair(
                    question=question,
                    answer=cap.sentence,
                    start_time=cap.start_time,
                    end_time=cap.end_time,
                )
            )

            if i > 0:
                prev = captions[i - 1]
                qa_pairs.append(
                    QAPair(
                        question=f"What happens after '{prev.sentence[:40]}...'?",
                        answer=cap.sentence,
                        start_time=cap.start_time,
                        end_time=cap.end_time,
                    )
                )

        return qa_pairs
