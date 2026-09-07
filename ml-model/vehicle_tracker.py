"""Track vehicles across consecutive video frames using YOLO tracking.

This module assigns a persistent track ID to each vehicle and reports how
far it moved since the previous frame. It does not decide whether an
accident occurred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from vehicle_detector import DEFAULT_MODEL, VEHICLE_CLASSES, VehicleDetector


@dataclass(frozen=True)
class VehicleMovement:
    """Change in position of a tracked vehicle since the last frame."""

    dx: float
    dy: float
    distance: float


@dataclass(frozen=True)
class TrackedVehicle:
    """A vehicle with a stable ID in the current frame."""

    track_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    movement: VehicleMovement


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _movement_from(
    previous_center: tuple[float, float] | None,
    current_center: tuple[float, float],
) -> VehicleMovement:
    if previous_center is None:
        return VehicleMovement(dx=0.0, dy=0.0, distance=0.0)

    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    return VehicleMovement(dx=dx, dy=dy, distance=float(np.hypot(dx, dy)))


class VehicleTracker:
    """Follow vehicles from frame to frame using a pretrained YOLO tracker.

    Frames must be passed in time order through :meth:`update`. Call
    :meth:`reset` before starting a new video so track IDs do not carry
    over from a previous sequence.

    The tracker reuses :class:`VehicleDetector` for the YOLO model and the
    vehicle class filter. Detection-only inference is left unchanged.
    """

    def __init__(
        self,
        detector: VehicleDetector | None = None,
        model_path: str = DEFAULT_MODEL,
        confidence_threshold: float = 0.25,
        vehicle_classes: Sequence[str] | None = None,
    ) -> None:
        """Create a tracker backed by a pretrained YOLO model.

        Args:
            detector: Existing detector to reuse. If omitted, one is
                created with ``model_path``, ``confidence_threshold``, and
                ``vehicle_classes``.
            model_path: Ultralytics model name or path used when no
                detector is provided.
            confidence_threshold: Minimum score to keep a tracked box.
            vehicle_classes: Class names to track. Defaults to car,
                motorcycle, bus, truck, and bicycle.
        """
        self.detector = detector or VehicleDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            vehicle_classes=vehicle_classes,
        )
        self.vehicle_classes = frozenset(
            vehicle_classes or self.detector.vehicle_classes or VEHICLE_CLASSES
        )
        self._previous_centers: dict[int, tuple[float, float]] = {}
        self._class_ids = self._resolve_class_ids()

    def reset(self) -> None:
        """Clear movement history and YOLO tracker state for a new video."""
        self._previous_centers.clear()
        if hasattr(self.detector.model, "predictor") and self.detector.model.predictor is not None:
            self.detector.model.predictor.trackers = []

    def update(self, frame: np.ndarray) -> list[TrackedVehicle]:
        """Track vehicles in the next video frame.

        Args:
            frame: BGR image array, as produced by OpenCV.

        Returns:
            Tracked vehicles in this frame, each with a persistent ID,
            class, confidence, bounding box, center point, and movement
            relative to the previous frame.
        """
        results = self.detector.model.track(
            source=frame,
            persist=True,
            conf=self.detector.confidence_threshold,
            classes=self._class_ids,
            verbose=False,
        )
        tracked = self._parse_tracks(results[0])
        self._previous_centers = {
            vehicle.track_id: vehicle.center for vehicle in tracked
        }
        return tracked

    def _resolve_class_ids(self) -> list[int]:
        name_to_id = {name: class_id for class_id, name in self.detector.model.names.items()}
        return [
            name_to_id[name]
            for name in self.vehicle_classes
            if name in name_to_id
        ]

    def _parse_tracks(self, result) -> list[TrackedVehicle]:
        tracked: list[TrackedVehicle] = []
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return tracked

        for box in boxes:
            if box.id is None:
                continue

            class_name = result.names[int(box.cls[0])]
            if class_name not in self.vehicle_classes:
                continue

            track_id = int(box.id[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = (int(x1), int(y1), int(x2), int(y2))
            center = _bbox_center(bbox)

            tracked.append(
                TrackedVehicle(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=float(box.conf[0]),
                    bbox=bbox,
                    center=center,
                    movement=_movement_from(self._previous_centers.get(track_id), center),
                )
            )

        return tracked
