import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger("aurora.data.msrvtt")


@dataclass
class MSRVTTVideo:
    video_id: str
    url: str
    start_time: float
    end_time: float
    split: str
    category: int
    captions: List[str] = field(default_factory=list)


class MSRVTTFetcher:
    """Loads and provides access to MSR-VTT dataset (Microsoft Research Video to Text)."""

    def __init__(
        self,
        data_root: str = "data/cache/msrvtt",
        annotations_path: Optional[str] = None,
    ):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.annotations_path = annotations_path
        self._annotations: Optional[Dict] = None
        self._video_df: Optional[pd.DataFrame] = None
        self._sentence_df: Optional[pd.DataFrame] = None

    def load_annotations(self, json_path: str = None) -> pd.DataFrame:
        """Load MSR-VTT JSON annotations into a DataFrame."""
        path = json_path or self.annotations_path or str(
            self.data_root / "train_val_videodatainfo.json"
        )

        if not Path(path).exists():
            logger.warning(
                f"MSR-VTT annotations not found at {path}. "
                "Download from https://github.com/crux82/msr-vtt"
            )
            return pd.DataFrame(
                columns=["video_id", "url", "start_time", "end_time", "split", "category"]
            )

        with open(path) as f:
            data = json.load(f)

        self._annotations = data
        videos = data.get("videos", [])
        sentences = data.get("sentences", [])

        video_records = [
            {
                "video_id": v["video_id"],
                "url": v.get("url", ""),
                "start_time": float(v.get("start time", 0)),
                "end_time": float(v.get("end time", 15)),
                "split": v.get("split", "train"),
                "category": int(v.get("category", 0)),
            }
            for v in videos
        ]

        self._video_df = pd.DataFrame(video_records)

        sentence_records = [
            {"video_id": s["video_id"], "caption": s["caption"]} for s in sentences
        ]
        self._sentence_df = pd.DataFrame(sentence_records)

        logger.info(
            f"Loaded {len(self._video_df)} videos, {len(self._sentence_df)} captions"
        )
        return self._video_df

    def fetch_video(self, video_id: str) -> str:
        """Download a specific MSR-VTT video by ID."""
        from data.fetchers.youtube_fetcher import YouTubeFetcher

        output_path = self.data_root / f"{video_id}.mp4"
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path)

        if self._video_df is None:
            self.load_annotations()

        if self._video_df is None or self._video_df.empty:
            return ""

        row = self._video_df[self._video_df["video_id"] == video_id]
        if row.empty:
            return ""

        url = row.iloc[0]["url"]
        fetcher = YouTubeFetcher(output_dir=str(self.data_root))
        try:
            return fetcher.download_video(url) or ""
        except Exception as e:
            logger.warning(f"Failed to fetch {video_id}: {e}")
            return ""

    def get_descriptions(self, video_id: str) -> List[str]:
        """Get all captions for a video ID."""
        if self._sentence_df is None:
            self.load_annotations()

        if self._sentence_df is None or self._sentence_df.empty:
            return []

        mask = self._sentence_df["video_id"] == video_id
        return self._sentence_df[mask]["caption"].tolist()

    def build_retrieval_benchmark(self) -> pd.DataFrame:
        """Build 1K test pairs with ground truth labels for R@1, R@5, R@10 evaluation."""
        if self._video_df is None:
            self.load_annotations()

        if self._video_df is None or self._video_df.empty:
            return pd.DataFrame(columns=["video_id", "query", "is_match", "split"])

        test_videos = self._video_df[self._video_df["split"] == "test"].head(1000)
        records = []

        for _, video_row in test_videos.iterrows():
            vid_id = video_row["video_id"]
            gt_captions = self.get_descriptions(vid_id)
            if not gt_captions:
                continue

            for cap in gt_captions[:20]:
                records.append(
                    {
                        "video_id": vid_id,
                        "query": cap,
                        "is_match": True,
                        "split": "test",
                    }
                )

        return pd.DataFrame(records)
