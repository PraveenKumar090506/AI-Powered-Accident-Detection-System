"""Simple test for VehicleDetector on a sample image with vehicles."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vehicle_detector import VehicleDetector

TEST_IMAGE_PATH = SCRIPT_DIR / "test_vehicles.jpg"
SAMPLE_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"


def ensure_test_image() -> Path:
    """Use a local sample image, or download a small photo with a bus."""
    if TEST_IMAGE_PATH.exists():
        return TEST_IMAGE_PATH

    print(f"Downloading sample image to {TEST_IMAGE_PATH} ...")
    urllib.request.urlretrieve(SAMPLE_IMAGE_URL, TEST_IMAGE_PATH)
    return TEST_IMAGE_PATH


def load_image(image_path: Path):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Could not read test image: {image_path}")
    return frame


def main() -> None:
    image_path = ensure_test_image()
    frame = load_image(image_path)

    detector = VehicleDetector()
    detections = detector.detect(frame)

    print(f"Test image: {image_path}")
    print()

    for index, vehicle in enumerate(detections, start=1):
        print(f"Vehicle {index}")
        print(f"  class name  : {vehicle.class_name}")
        print(f"  confidence  : {vehicle.confidence:.4f}")
        print(f"  bounding box: {vehicle.bbox}")
        print()

    print(f"Total vehicles detected: {len(detections)}")


if __name__ == "__main__":
    main()
