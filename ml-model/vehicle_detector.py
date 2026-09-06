"""Detect vehicles in a video frame using a pretrained YOLO model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = frozenset({"car", "motorcycle", "bus", "truck", "bicycle"})
DEFAULT_MODEL = "yolov8n.pt"


@dataclass(frozen=True)
class VehicleDetection:
    """A single vehicle found in a frame."""

    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


class VehicleDetector:
    """Run pretrained YOLO inference and keep only vehicle detections.

    This module does not classify accidents. It only reports vehicles
    present in a single image or video frame.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        confidence_threshold: float = 0.25,
        vehicle_classes: Sequence[str] | None = None,
    ) -> None:
        """Load a pretrained YOLO model.

        Args:
            model_path: Ultralytics model name or path. Defaults to the
                small pretrained COCO model ``yolov8n.pt``.
            confidence_threshold: Minimum detection score to keep.
            vehicle_classes: Class names to keep. Defaults to car,
                motorcycle, bus, truck, and bicycle.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.vehicle_classes = frozenset(vehicle_classes or VEHICLE_CLASSES)

    def detect(self, frame: np.ndarray) -> list[VehicleDetection]:
        """Detect vehicles in one image or video frame.

        Args:
            frame: BGR image array, as produced by OpenCV.

        Returns:
            Vehicle detections with class name, confidence, and bounding
            box ``(x1, y1, x2, y2)`` in pixel coordinates.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False,
        )
        return self._parse_results(results[0])

    def _parse_results(self, result) -> list[VehicleDetection]:
        detections: list[VehicleDetection] = []
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            class_name = result.names[int(box.cls[0])]
            if class_name not in self.vehicle_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                VehicleDetection(
                    class_name=class_name,
                    confidence=float(box.conf[0]),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        return detections
