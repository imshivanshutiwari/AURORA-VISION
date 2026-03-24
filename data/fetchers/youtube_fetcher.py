import os
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("aurora.data.youtube")


class YouTubeFetcher:
    """Downloads real YouTube videos using yt-dlp."""

    def __init__(self, output_dir: str = "data/cache/youtube"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_video(
        self, url: str, quality: str = "720p", output_dir: str = None
    ) -> str:
        """Download a YouTube video and return the local file path."""
        import yt_dlp

        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        height = quality.replace("p", "") if quality.endswith("p") else "720"

        ydl_opts = {
            "format": f"bestvideo[height<={height}]+bestaudio/best[height<={height}]",
            "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info.get("id", "")
                ext = info.get("ext", "mp4")

                candidate = out_dir / f"{video_id}.mp4"
                if candidate.exists():
                    return str(candidate)

                for f in out_dir.glob(f"{video_id}.*"):
                    return str(f)

                return ""
        except Exception as e:
            logger.warning(f"yt-dlp failed for {url}: {e}")
            return ""

    def extract_info(self, url: str) -> Dict:
        """Extract metadata without downloading."""
        import yt_dlp

        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False) or {}
        except Exception as e:
            logger.warning(f"Could not extract info for {url}: {e}")
            return {}

    def download_playlist(self, playlist_url: str, max_n: int = 20) -> List[str]:
        """Download up to max_n videos from a YouTube playlist."""
        import yt_dlp

        ydl_opts = {
            "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "outtmpl": str(self.output_dir / "%(id)s.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "playlistend": max_n,
        }

        downloaded = []
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(playlist_url, download=True)
                entries = info.get("entries", []) if info else []
                for entry in entries[:max_n]:
                    vid_id = entry.get("id", "")
                    for f in self.output_dir.glob(f"{vid_id}.*"):
                        downloaded.append(str(f))
                        break
        except Exception as e:
            logger.warning(f"Playlist download failed for {playlist_url}: {e}")

        return downloaded
