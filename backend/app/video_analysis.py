"""Run the existing ML pipeline on an uploaded video file.

``ml-model`` is not a Python package name, so this module adds that
folder to ``sys.path`` and imports the tracker, verifier, and evidence
capture classes directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_MODEL_DIR = PROJECT_ROOT / "ml-model"

if str(ML_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(ML_MODEL_DIR))

from accident_verifier import AccidentVerifier
from evidence_capture import EvidenceCapture
from vehicle_detector import VehicleDetector
from vehicle_tracker import VehicleTracker

# Reuse one YOLO detector across requests; each video still gets a fresh tracker.
_detector: VehicleDetector | None = None


class InvalidVideoError(ValueError):
    """The file could not be opened as a video."""


def _get_detector() -> VehicleDetector:
    global _detector
    if _detector is None:
        _detector = VehicleDetector()
    return _detector


def analyze_video_file(video_path: Path) -> dict[str, Any]:
    """Process a video until an accident is confirmed, or the file ends."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise InvalidVideoError(f"Could not open video: {video_path}")

    tracker = VehicleTracker(detector=_get_detector())
    tracker.reset()
    verifier = AccidentVerifier()
    evidence = EvidenceCapture()

    frames_processed = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frames_processed += 1
            tracked_vehicles = tracker.update(frame)
            result = verifier.update(tracked_vehicles)

            if not result.accident_suspected:
                continue

            evidence_path = evidence.save_evidence(frame, result)
            return {
                "status": "completed",
                "accident_suspected": True,
                "confidence": result.confidence,
                "involved_track_ids": list(result.involved_track_ids),
                "reasons": list(result.reasons),
                "evidence_path": str(evidence_path),
                "frames_processed": frames_processed,
            }
    finally:
        capture.release()

    if frames_processed == 0:
        raise InvalidVideoError(f"Could not read frames from video: {video_path}")

    return {
        "status": "completed",
        "accident_suspected": False,
        "frames_processed": frames_processed,
        "evidence_path": None,
    }
