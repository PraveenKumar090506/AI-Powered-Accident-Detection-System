"""Open and read video files sequentially with OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoOpenError(Exception):
    """Raised when a video path is missing or cannot be opened."""


@dataclass(frozen=True)
class VideoMetadata:
    """Basic properties of an opened video file."""

    width: int
    height: int
    fps: float
    total_frame_count: int
    duration: float


class VideoReader:
    """Open a video file, expose metadata, and yield frames in order.

    Use as a context manager so the capture is always released::

        with VideoReader("clip.mp4") as reader:
            for frame in reader.frames():
                ...
    """

    def __init__(self, video_path: str | Path) -> None:
        """Open ``video_path`` with OpenCV and load metadata.

        Args:
            video_path: Path to a local video file.

        Raises:
            VideoOpenError: If the path is missing, not a file, or OpenCV
                cannot open the video.
        """
        self.video_path = Path(video_path)
        self._capture: cv2.VideoCapture | None = None
        self._validate_path()
        self._open()
        self.metadata = self._read_metadata()

    def _validate_path(self) -> None:
        if not self.video_path.exists():
            raise VideoOpenError(f"Video file not found: {self.video_path}")
        if not self.video_path.is_file():
            raise VideoOpenError(f"Video path is not a file: {self.video_path}")

    def _open(self) -> None:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            capture.release()
            raise VideoOpenError(
                f"Unable to open video file: {self.video_path}"
            )
        self._capture = capture

    def _require_capture(self) -> cv2.VideoCapture:
        if self._capture is None:
            raise VideoOpenError(
                f"Video capture is not open: {self.video_path}"
            )
        return self._capture

    def _read_metadata(self) -> VideoMetadata:
        capture = self._require_capture()
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frame_count / fps if fps > 0 else 0.0
        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            total_frame_count=total_frame_count,
            duration=duration,
        )

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames from the start of the video until it ends."""
        capture = self._require_capture()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        """Release the OpenCV video capture if it is still open."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()
