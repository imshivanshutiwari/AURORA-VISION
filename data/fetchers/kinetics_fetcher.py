import io
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm

from utils.logger import get_logger

logger = get_logger("aurora.data.kinetics")

KINETICS_ANNOTATIONS_URL = (
    "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz"
)
KINETICS_CSV_URL = (
    "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400_train.csv"
)


@dataclass
class VideoClip:
    youtube_id: str
    time_start: int
    time_end: int
    label: str
    split: str
    local_path: Optional[str] = None


class KineticsFetcher:
    """Fetches real Kinetics-400 annotations and video clips from DeepMind storage."""

    ANNOTATIONS_URL = KINETICS_ANNOTATIONS_URL
    COLUMNS = ["youtube_id", "time_start", "time_end", "label", "split"]

    def __init__(self, cache_dir: str = "data/cache/kinetics"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._annotations: Optional[pd.DataFrame] = None

    def download_annotations(self) -> pd.DataFrame:
        """Download and parse Kinetics-400 annotation CSV from DeepMind."""
        csv_path = self.cache_dir / "kinetics400_annotations.csv"

        if csv_path.exists():
            logger.info(f"Loading cached Kinetics-400 annotations from {csv_path}")
            self._annotations = pd.read_csv(csv_path)
            return self._annotations

        logger.info("Downloading Kinetics-400 annotations from DeepMind...")

        train_url = (
            "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400_train.csv"
        )
        val_url = (
            "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400_val.csv"
        )
        test_url = (
            "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400_test.csv"
        )

        dfs = []
        for url, split in [(train_url, "train"), (val_url, "val"), (test_url, "test")]:
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))
                df["split"] = split
                dfs.append(df)
                logger.info(f"Downloaded {split} split: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Could not download {split} split from {url}: {e}")

        if dfs:
            self._annotations = pd.concat(dfs, ignore_index=True)
            self._annotations.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(self._annotations)} annotations to {csv_path}")
        else:
            raise RuntimeError("Failed to download any Kinetics-400 annotations")

        return self._annotations

    def fetch_video_clip(self, youtube_id: str, start: int, end: int) -> str:
        """Download and trim a video clip using yt-dlp + ffmpeg."""
        from data.fetchers.youtube_fetcher import YouTubeFetcher

        output_path = self.cache_dir / f"{youtube_id}_{start}_{end}.mp4"
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path)

        fetcher = YouTubeFetcher(output_dir=str(self.cache_dir))
        url = f"https://www.youtube.com/watch?v={youtube_id}"

        try:
            raw_path = fetcher.download_video(url, quality="360p")
            if raw_path and os.path.exists(raw_path):
                import ffmpeg

                (
                    ffmpeg.input(raw_path, ss=start, t=(end - start))
                    .output(str(output_path), c="copy")
                    .overwrite_output()
                    .run(quiet=True)
                )
                logger.info(f"Trimmed clip saved: {output_path}")
                return str(output_path)
        except Exception as e:
            logger.warning(f"Failed to fetch {youtube_id}: {e}")

        return ""

    def fetch_sample(
        self,
        n_per_class: int = 10,
        classes: List[str] = None,
    ) -> List[VideoClip]:
        """Fetch a sample of video clips for given action classes."""
        if classes is None:
            classes = ["cooking", "sports", "music"]

        if self._annotations is None:
            self.download_annotations()

        clips = []
        for cls in classes:
            mask = self._annotations["label"].str.lower().str.contains(
                cls.lower(), na=False
            )
            subset = self._annotations[mask].head(n_per_class)
            for _, row in subset.iterrows():
                clips.append(
                    VideoClip(
                        youtube_id=str(row.get("youtube_id", row.get("id", ""))),
                        time_start=int(row.get("time_start", 0)),
                        time_end=int(row.get("time_end", 10)),
                        label=str(row.get("label", cls)),
                        split=str(row.get("split", "train")),
                    )
                )

        logger.info(f"Fetched {len(clips)} clips across {len(classes)} classes")
        return clips

    def cache_clips(self, path: str = "data/cache/kinetics") -> None:
        """Cache all sampled clips to disk."""
        clips = self.fetch_sample()
        self.cache_dir = Path(path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for clip in tqdm(clips, desc="Caching Kinetics clips"):
            if not clip.local_path:
                clip.local_path = self.fetch_video_clip(
                    clip.youtube_id, clip.time_start, clip.time_end
                )
