"""Simple moving-video test for VehicleTracker."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vehicle_tracker import VehicleTracker

SAMPLE_VIDEO_PATH = SCRIPT_DIR / "test_tracking.mp4"
SAMPLE_VIDEO_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/anpr-demo-video.mp4"
)


def ensure_sample_video() -> Path:
    """Use a local sample video, or download the official Ultralytics vehicle clip."""
    if SAMPLE_VIDEO_PATH.exists():
        return SAMPLE_VIDEO_PATH

    print(f"Downloading sample tracking video to {SAMPLE_VIDEO_PATH} ...")
    urllib.request.urlretrieve(SAMPLE_VIDEO_URL, SAMPLE_VIDEO_PATH)
    return SAMPLE_VIDEO_PATH


def resolve_video_path() -> Path:
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python test_tracking_video.py [video_path]")
    if len(sys.argv) == 2:
        return Path(sys.argv[1])
    return ensure_sample_video()


def open_video(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    return capture


def main() -> None:
    video_path = resolve_video_path()
    capture = open_video(video_path)
    tracker = VehicleTracker()

    print(f"Test video: {video_path}")
    print()

    frames_processed = 0
    max_tracked = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frames_processed += 1
        tracked_vehicles = tracker.update(frame)
        max_tracked = max(max_tracked, len(tracked_vehicles))

        print(f"Frame {frames_processed}")
        for vehicle in tracked_vehicles:
            print(f"  track_id           : {vehicle.track_id}")
            print(f"  class_name         : {vehicle.class_name}")
            print(f"  confidence         : {vehicle.confidence:.4f}")
            print(f"  movement distance  : {vehicle.movement.distance:.4f}")
            print()

        print(f"  Total tracked vehicles: {len(tracked_vehicles)}")
        print()

    capture.release()

    print(f"Total frames processed: {frames_processed}")
    print(f"Maximum simultaneously tracked vehicles: {max_tracked}")


if __name__ == "__main__":
    main()
