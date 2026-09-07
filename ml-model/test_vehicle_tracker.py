"""Simple test for VehicleTracker using a still image as consecutive frames."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vehicle_tracker import VehicleTracker

TEST_IMAGE_PATH = SCRIPT_DIR / "test_vehicles.jpg"
NUM_TEST_FRAMES = 5


def load_image(image_path: Path):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Could not read test image: {image_path}")
    return frame


def main() -> None:
    frame = load_image(TEST_IMAGE_PATH)
    tracker = VehicleTracker()

    print(f"Test image: {TEST_IMAGE_PATH}")
    print(f"Processing the same image for {NUM_TEST_FRAMES} consecutive frames")
    print()

    for frame_number in range(1, NUM_TEST_FRAMES + 1):
        tracked_vehicles = tracker.update(frame)

        print(f"Frame {frame_number}")
        for vehicle in tracked_vehicles:
            print(f"  track ID           : {vehicle.track_id}")
            print(f"  class name         : {vehicle.class_name}")
            print(f"  confidence         : {vehicle.confidence:.4f}")
            print(f"  center             : {vehicle.center}")
            print(f"  movement distance  : {vehicle.movement.distance:.4f}")
            print()

        print(f"  Total tracked vehicles: {len(tracked_vehicles)}")
        print()


if __name__ == "__main__":
    main()
