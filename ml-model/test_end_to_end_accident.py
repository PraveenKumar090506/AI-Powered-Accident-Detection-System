"""End-to-end test: video → tracking → accident verification."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from accident_verifier import AccidentVerifier, AccidentVerificationResult
from vehicle_tracker import VehicleTracker

# Same moving-vehicle clip used by test_tracking_video.py.
SAMPLE_VIDEO_PATH = SCRIPT_DIR / "test_tracking.mp4"
SAMPLE_VIDEO_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/anpr-demo-video.mp4"
)
PROGRESS_EVERY_N_FRAMES = 30


def ensure_sample_video() -> Path:
    """Use a local sample video, or download the official Ultralytics vehicle clip."""
    if SAMPLE_VIDEO_PATH.exists():
        return SAMPLE_VIDEO_PATH

    print(f"Downloading sample tracking video to {SAMPLE_VIDEO_PATH} ...")
    urllib.request.urlretrieve(SAMPLE_VIDEO_URL, SAMPLE_VIDEO_PATH)
    return SAMPLE_VIDEO_PATH


def resolve_video_path() -> Path:
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python test_end_to_end_accident.py [video_path]")
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
    verifier = AccidentVerifier()

    print(f"Test video: {video_path}")
    print("Pipeline: video → VehicleTracker → AccidentVerifier")
    print()

    frames_processed = 0
    max_tracked = 0
    best_result: AccidentVerificationResult | None = None

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frames_processed += 1
        tracked_vehicles = tracker.update(frame)
        max_tracked = max(max_tracked, len(tracked_vehicles))
        result = verifier.update(tracked_vehicles)

        if result.accident_suspected:
            if best_result is None or result.confidence >= best_result.confidence:
                best_result = result

        if frames_processed % PROGRESS_EVERY_N_FRAMES == 0:
            print(
                f"Frame {frames_processed}: "
                f"tracked={len(tracked_vehicles)}, "
                f"accident_suspected={result.accident_suspected}, "
                f"confidence={result.confidence:.3f}"
            )

    capture.release()

    print()
    print(f"Total frames processed: {frames_processed}")
    print(f"Maximum vehicles tracked simultaneously: {max_tracked}")

    if best_result is None:
        print("Whether an accident was suspected: False")
        print("Involved track IDs: n/a")
        print("Accident confidence/score: n/a")
        print("Reasons: n/a")
        print("No accident detected in test video.")
        return

    print("Whether an accident was suspected: True")
    print(f"Involved track IDs: {best_result.involved_track_ids}")
    print(f"Accident confidence/score: {best_result.confidence:.3f}")
    if best_result.reasons:
        print("Reasons:")
        for reason in best_result.reasons:
            print(f"  - {reason}")
    else:
        print("Reasons: n/a")


if __name__ == "__main__":
    main()
