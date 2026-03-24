import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

from utils.logger import get_logger

logger = get_logger("aurora.data.howto100m")

HOWTO100M_CSV_URL = (
    "https://www.rocq.inria.fr/cluster-willow/miech/howto100m/HowTo100M.csv"
)


@dataclass
class HowTo100MClip:
    video_id: str
    category: str
    task: str
    url: str
    local_path: Optional[str] = None


class HowTo100MFetcher:
    """Fetches HowTo100M instructional video clips."""

    CSV_URL = HOWTO100M_CSV_URL

    def __init__(self, cache_dir: str = "data/cache/howto100m"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata: Optional[pd.DataFrame] = None

    def download_metadata(self) -> pd.DataFrame:
        """Download HowTo100M video metadata CSV."""
        csv_path = self.cache_dir / "HowTo100M.csv"

        if csv_path.exists():
            logger.info(f"Loading cached HowTo100M metadata from {csv_path}")
            self._metadata = pd.read_csv(csv_path)
            return self._metadata

        logger.info("Downloading HowTo100M metadata...")
        try:
            resp = requests.get(self.CSV_URL, timeout=120, stream=True)
            resp.raise_for_status()
            with open(csv_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            self._metadata = pd.read_csv(csv_path)
            logger.info(f"Downloaded HowTo100M: {len(self._metadata)} videos")
        except Exception as e:
            logger.warning(f"Could not download HowTo100M metadata: {e}")
            self._metadata = pd.DataFrame(
                columns=["video_id", "category", "task", "url"]
            )

        return self._metadata

    def fetch_clip(self, video_id: str) -> str:
        """Download a single HowTo100M clip."""
        from data.fetchers.youtube_fetcher import YouTubeFetcher

        out = self.cache_dir / f"{video_id}.mp4"
        if out.exists() and out.stat().st_size > 0:
            return str(out)

        fetcher = YouTubeFetcher(output_dir=str(self.cache_dir))
        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            return fetcher.download_video(url) or ""
        except Exception as e:
            logger.warning(f"Failed to fetch HowTo100M clip {video_id}: {e}")
            return ""

    def get_clips_by_category(
        self, category: str, n: int = 20
    ) -> List[HowTo100MClip]:
        """Get clips matching a task category."""
        if self._metadata is None:
            self.download_metadata()

        if self._metadata is None or self._metadata.empty:
            return []

        col = "category" if "category" in self._metadata.columns else self._metadata.columns[0]
        mask = self._metadata[col].str.lower().str.contains(category.lower(), na=False)
        subset = self._metadata[mask].head(n)

        clips = []
        for _, row in subset.iterrows():
            clips.append(
                HowTo100MClip(
                    video_id=str(row.get("video_id", "")),
                    category=str(row.get("category", category)),
                    task=str(row.get("task", "")),
                    url=str(row.get("url", "")),
                )
            )
        return clips
