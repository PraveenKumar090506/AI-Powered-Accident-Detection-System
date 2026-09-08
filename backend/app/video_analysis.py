"""Run the ML pipeline on an uploaded video file and generate an annotated video.

``ml-model`` is not a Python package name, so this module adds that
folder to ``sys.path`` and imports the tracker, verifier, and evidence
capture classes directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_MODEL_DIR = PROJECT_ROOT / "ml-model"
PROCESSED_DIR = PROJECT_ROOT / "uploads" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

if str(ML_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(ML_MODEL_DIR))

from accident_verifier import AccidentVerificationResult, AccidentVerifier
from evidence_capture import EvidenceCapture
from vehicle_detector import VehicleDetector
from vehicle_tracker import TrackedVehicle, VehicleTracker

# Reuse one YOLO detector across requests; each video still gets a fresh tracker.
_detector: VehicleDetector | None = None


class InvalidVideoError(ValueError):
    """The file could not be opened as a video."""


def _get_detector() -> VehicleDetector:
    global _detector
    if _detector is None:
        _detector = VehicleDetector()
    return _detector


def _create_video_writer(out_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create a VideoWriter preferring avc1 (H.264) for HTML5 browser playback, with mp4v fallback."""
    fourcc_avc1 = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc_avc1, fps, (width, height))
    if writer.isOpened():
        return writer
    fourcc_mp4v = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc_mp4v, fps, (width, height))


def _annotate_frame(
    frame: np.ndarray,
    tracked_vehicles: Sequence[TrackedVehicle],
    suspicious_track_ids: set[int],
    confirmed_accident: bool,
    involved_track_ids: set[int],
    confidence: float,
    track_history: dict[int, list[tuple[int, int]]],
    frame_idx: int,
) -> np.ndarray:
    """Draw vehicle bounding boxes, class, ID, confidence, movement trails, and accident indicators."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw movement / tracking history trails first (under bounding boxes)
    for vehicle in tracked_vehicles:
        tid = vehicle.track_id
        history = track_history.get(tid, [])
        if len(history) > 1:
            if confirmed_accident and tid in involved_track_ids:
                trail_color = (0, 0, 240)
            elif tid in suspicious_track_ids:
                trail_color = (0, 165, 255)
            else:
                trail_color = (0, 220, 180)  # Bright cyan/green

            for idx in range(1, len(history)):
                pt1 = history[idx - 1]
                pt2 = history[idx]
                ratio = idx / len(history)
                thickness = max(1, int(3 * ratio))
                cv2.line(annotated, pt1, pt2, trail_color, thickness, cv2.LINE_AA)

        # Draw center point
        cx, cy = int(vehicle.center[0]), int(vehicle.center[1])
        cv2.circle(annotated, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

        # Draw velocity / direction vector arrow
        if vehicle.movement.distance > 1.8:
            arrow_scale = min(35.0, max(12.0, vehicle.movement.distance * 2.2))
            norm = max(vehicle.movement.distance, 0.001)
            arr_dx = int((vehicle.movement.dx / norm) * arrow_scale)
            arr_dy = int((vehicle.movement.dy / norm) * arrow_scale)
            cv2.arrowedLine(
                annotated,
                (cx, cy),
                (cx + arr_dx, cy + arr_dy),
                (0, 255, 255),
                2,
                tipLength=0.35,
            )

    # Draw tracked vehicle bounding boxes and labels
    for vehicle in tracked_vehicles:
        tid = vehicle.track_id
        x1, y1, x2, y2 = vehicle.bbox
        conf_pct = int(vehicle.confidence * 100)
        cname = vehicle.class_name.capitalize()
        dist_str = f"{vehicle.movement.distance:.1f}px"

        if confirmed_accident and tid in involved_track_ids:
            # Confirmed accident state (Red)
            color = (0, 0, 230)
            thickness = 3
            label = f"{cname} | ID {tid} | {conf_pct}% [ACCIDENT DETECTED]"
        elif tid in suspicious_track_ids:
            # Suspicious interaction (Amber / Orange)
            color = (0, 140, 255)
            thickness = 2
            label = f"{cname} | ID {tid} | {conf_pct}% [SUSPICIOUS]"
        else:
            # Normal tracked vehicle (Green)
            color = (40, 200, 40)
            thickness = 2
            label = f"{cname} | ID {tid} | {conf_pct}% | {dist_str}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Corner accents for sleek modern HUD look
        corner_len = min(14, max(6, (x2 - x1) // 5))
        cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), thickness + 1)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), thickness + 1)
        cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), thickness + 1)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), thickness + 1)

        # Draw label badge
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.46
        font_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

        lbl_y2 = max(y1, text_h + 8)
        lbl_y1 = lbl_y2 - text_h - 6
        lbl_x1 = max(0, x1)
        lbl_x2 = min(w, x1 + text_w + 10)

        cv2.rectangle(annotated, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (lbl_x1 + 5, lbl_y2 - 4),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    # Top notification / HUD banners
    if confirmed_accident:
        cv2.rectangle(annotated, (0, 0), (w, 44), (0, 0, 220), -1)
        inv_str = ", ".join(str(tid) for tid in sorted(involved_track_ids)) or "Multiple"
        alert_text = f"ACCIDENT DETECTED - Confidence: {int(confidence * 100)}% - Involved IDs: {inv_str}"
        cv2.putText(
            annotated,
            alert_text,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    elif suspicious_track_ids:
        cv2.rectangle(annotated, (0, 0), (w, 38), (0, 140, 255), -1)
        susp_str = ", ".join(str(tid) for tid in sorted(suspicious_track_ids))
        warn_text = f"SUSPICIOUS: POTENTIAL ACCIDENT DETECTED (Tracks: {susp_str})"
        cv2.putText(
            annotated,
            warn_text,
            (16, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        # Subtle HUD top bar showing active tracking process
        cv2.rectangle(annotated, (0, 0), (w, 32), (20, 26, 34), -1)
        cv2.line(annotated, (0, 32), (w, 32), (37, 99, 235), 1)
        hud_text = f"AI DETECTION & TRACKING ACTIVE | Vehicles: {len(tracked_vehicles)} | Frame: {frame_idx}"
        cv2.putText(
            annotated,
            hud_text,
            (16, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 230, 245),
            1,
            cv2.LINE_AA,
        )

    return annotated


def analyze_video_file(
    video_path: Path,
    original_video_url: str | None = None,
) -> dict[str, Any]:
    """Process a video, generate an annotated output video, and detect accidents."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise InvalidVideoError(f"Could not open video: {video_path}")

    tracker = VehicleTracker(detector=_get_detector())
    tracker.reset()
    verifier = AccidentVerifier()
    evidence = EvidenceCapture()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_filename = f"annotated_{uuid4().hex[:12]}.mp4"
    out_path = PROCESSED_DIR / out_filename

    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    if fps <= 0 or fps > 60:
        fps = 20.0

    writer: cv2.VideoWriter | None = None
    frames_processed = 0
    confirmed_accident = False
    best_result: AccidentVerificationResult | None = None
    saved_evidence_path: Path | None = None
    all_reasons: list[str] = []
    all_involved: set[int] = set()

    # Track position history for movement trails
    track_history: dict[int, list[tuple[int, int]]] = {}

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frames_processed += 1
            if writer is None:
                h, w = frame.shape[:2]
                writer = _create_video_writer(out_path, fps, w, h)

            tracked_vehicles = tracker.update(frame)

            # Update movement history for active tracks
            current_track_ids = {v.track_id for v in tracked_vehicles}
            for v in tracked_vehicles:
                cx, cy = int(v.center[0]), int(v.center[1])
                if v.track_id not in track_history:
                    track_history[v.track_id] = []
                track_history[v.track_id].append((cx, cy))
                if len(track_history[v.track_id]) > 25:
                    track_history[v.track_id].pop(0)

            # Prune old tracks
            for tid in list(track_history):
                if tid not in current_track_ids:
                    track_history[tid].pop(0)
                    if not track_history[tid]:
                        del track_history[tid]

            # Check suspicious pairs (before confirmation threshold)
            assessments = verifier._assess_pairs(tracked_vehicles)
            suspicious_track_ids = {
                tid for a in assessments if a.suspicious for tid in a.track_ids
            }

            result = verifier.update(tracked_vehicles)

            if result.accident_suspected:
                confirmed_accident = True
                if best_result is None or result.confidence >= best_result.confidence:
                    best_result = result
                    all_involved.update(result.involved_track_ids)
                    for r in result.reasons:
                        if r not in all_reasons:
                            all_reasons.append(r)
                if saved_evidence_path is None:
                    saved_evidence_path = evidence.save_evidence(frame, result)

            conf_val = best_result.confidence if best_result is not None else 0.0
            annotated_frame = _annotate_frame(
                frame=frame,
                tracked_vehicles=tracked_vehicles,
                suspicious_track_ids=suspicious_track_ids,
                confirmed_accident=confirmed_accident,
                involved_track_ids=all_involved if confirmed_accident else set(),
                confidence=conf_val,
                track_history=track_history,
                frame_idx=frames_processed,
            )

            if writer is not None:
                writer.write(annotated_frame)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frames_processed == 0:
        raise InvalidVideoError(f"Could not read frames from video: {video_path}")

    if confirmed_accident and best_result is not None:
        return {
            "status": "completed",
            "accident_suspected": True,
            "confidence": best_result.confidence,
            "involved_track_ids": sorted(all_involved) if all_involved else list(best_result.involved_track_ids),
            "reasons": all_reasons if all_reasons else list(best_result.reasons),
            "evidence_path": str(saved_evidence_path) if saved_evidence_path else None,
            "frames_processed": frames_processed,
            "processed_video_url": f"/processed/{out_filename}",
            "processed_video_path": str(out_path),
            "original_video_url": original_video_url,
        }

    return {
        "status": "completed",
        "accident_suspected": False,
        "confidence": 0.0,
        "involved_track_ids": [],
        "reasons": [],
        "frames_processed": frames_processed,
        "evidence_path": None,
        "processed_video_url": f"/processed/{out_filename}",
        "processed_video_path": str(out_path),
        "original_video_url": original_video_url,
    }
