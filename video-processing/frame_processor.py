"""Sequential frame processing built on :mod:`video_reader`."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from video_reader import VideoReader


class ProcessingStats(TypedDict):
    """Summary of a sequential pass over a video."""

    frames_processed: int
    fps: float
    duration: float


def process_video(video_path: str | Path) -> ProcessingStats:
    """Read a video and process its frames one after another.

    This pass only counts frames. It does not run accident detection,
    object detection, or tracking.

    Args:
        video_path: Path to a local video file.

    Returns:
        Statistics for the processed video: ``frames_processed``, ``fps``,
        and ``duration`` (seconds).

    Raises:
        VideoOpenError: If the video is missing or cannot be opened.
    """
    with VideoReader(video_path) as reader:
        frames_processed = 0
        for _frame in reader.frames():
            frames_processed += 1

        return {
            "frames_processed": frames_processed,
            "fps": reader.metadata.fps,
            "duration": reader.metadata.duration,
        }
