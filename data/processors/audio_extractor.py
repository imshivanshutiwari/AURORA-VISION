from pathlib import Path
from typing import Optional

import ffmpeg

from utils.logger import get_logger

logger = get_logger("aurora.data.audio_extractor")


class AudioExtractor:
    """Extracts audio from video files using FFmpeg."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def extract(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> str:
        """Extract audio track from video as WAV file."""
        video_path = Path(video_path)
        if output_path is None:
            output_path = str(video_path.with_suffix(".wav"))

        output_path = str(output_path)

        try:
            input_stream = ffmpeg.input(str(video_path), ss=start_sec)
            if end_sec is not None:
                input_stream = ffmpeg.input(
                    str(video_path), ss=start_sec, t=(end_sec - start_sec)
                )

            (
                input_stream.audio.output(
                    output_path,
                    acodec="pcm_s16le",
                    ac=1,
                    ar=str(self.sample_rate),
                )
                .overwrite_output()
                .run(quiet=True)
            )
            logger.info(f"Extracted audio to {output_path}")
        except Exception as e:
            logger.error(f"Audio extraction failed for {video_path}: {e}")
            raise

        return output_path

    def extract_chunks(
        self,
        video_path: str,
        chunk_duration: float = 30.0,
        output_dir: Optional[str] = None,
    ) -> list[str]:
        """Extract audio in chunks of chunk_duration seconds."""
        from data.processors.video_decoder import VideoDecoder

        decoder = VideoDecoder()
        info = decoder.get_video_info(str(video_path))
        duration = info.get("duration", 0.0)

        if output_dir is None:
            output_dir = str(Path(video_path).parent)

        base = Path(video_path).stem
        chunks = []
        start = 0.0
        idx = 0

        while start < duration:
            end = min(start + chunk_duration, duration)
            out_path = str(Path(output_dir) / f"{base}_chunk_{idx:04d}.wav")
            try:
                self.extract(
                    str(video_path),
                    output_path=out_path,
                    start_sec=start,
                    end_sec=end,
                )
                chunks.append(out_path)
            except Exception as e:
                logger.warning(f"Chunk {idx} extraction failed: {e}")
            start = end
            idx += 1

        return chunks
