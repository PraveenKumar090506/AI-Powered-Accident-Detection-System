"""Save unmodified video frames when an accident is reported.

This module only writes evidence images. It does not draw overlays,
send alerts, or store location data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2

from accident_verifier import AccidentVerificationResult

# Default folder next to this module: ml-model/evidence/
DEFAULT_EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"


class EvidenceCapture:
    """Write accident evidence frames to disk as PNG files."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        """Create a capture helper.

        Args:
            output_dir: Folder for PNG files. Defaults to ``ml-model/evidence/``.
        """
        self.output_dir = Path(output_dir) if output_dir is not None else DEFAULT_EVIDENCE_DIR

    def save_evidence(
        self,
        frame: Any,
        accident_result: AccidentVerificationResult,
    ) -> Path:
        """Save the given OpenCV frame without drawing on it.

        Args:
            frame: BGR image array from OpenCV.
            accident_result: Verifier result that triggered this capture.

        Returns:
            Absolute path of the written PNG file.

        Raises:
            ValueError: Frame is invalid or the result is not an accident.
            OSError: The image could not be written.
        """
        if not accident_result.accident_suspected:
            raise ValueError("Refusing to save evidence: accident is not suspected")

        self._validate_frame(frame)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        evidence_path = self.output_dir / self._make_filename()
        try:
            written = cv2.imwrite(str(evidence_path), frame)
        except Exception as exc:
            raise OSError(f"Could not write evidence image: {evidence_path}") from exc

        if not written or not evidence_path.is_file():
            raise OSError(f"Could not write evidence image: {evidence_path}")

        return evidence_path.resolve()

    def _validate_frame(self, frame: Any) -> None:
        """Reject missing or empty images before writing."""
        if frame is None:
            raise ValueError("Cannot save evidence: frame is None")
        if not hasattr(frame, "size") or not hasattr(frame, "shape"):
            raise ValueError("Cannot save evidence: frame is not an image array")
        if frame.size == 0 or len(frame.shape) < 2:
            raise ValueError("Cannot save evidence: frame is empty")

    @staticmethod
    def _make_filename() -> str:
        """Build a unique PNG name from UTC time and a short UUID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = uuid4().hex[:8]
        return f"evidence_{timestamp}_{unique_id}.png"
