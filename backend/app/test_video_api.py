"""End-to-end test for POST /analyze-video with test_tracking.mp4."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app, PROJECT_ROOT

client = TestClient(app)

SAMPLE_VIDEO_PATH = PROJECT_ROOT / "ml-model" / "test_tracking.mp4"


def test_invalid_extension_rejected():
    files = {"file": ("test.txt", b"hello world", "text/plain")}
    response = client.post("/analyze-video", files=files)
    assert response.status_code == 400
    assert "Invalid uploaded file" in response.json().get("detail", "")


def test_empty_file_rejected():
    files = {"file": ("test.mp4", b"", "video/mp4")}
    response = client.post("/analyze-video", files=files)
    assert response.status_code == 400
    assert "empty" in response.json().get("detail", "").lower()


def test_analyze_video_normal_tracking_video():
    assert SAMPLE_VIDEO_PATH.is_file(), f"Sample video missing: {SAMPLE_VIDEO_PATH}"
    video_bytes = SAMPLE_VIDEO_PATH.read_bytes()

    files = {"file": ("test_tracking.mp4", video_bytes, "video/mp4")}
    response = client.post("/analyze-video", files=files)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["status"] == "completed"
    assert data["accident_suspected"] is False
    assert data["confidence"] == 0.0
    assert data["frames_processed"] == 168
    assert data["involved_track_ids"] == []

    # Verify processed video URL and file existence
    proc_url = data.get("processed_video_url")
    assert proc_url and proc_url.startswith("/processed/"), f"Unexpected proc_url: {proc_url}"
    proc_filename = proc_url.replace("/processed/", "")
    proc_path = PROJECT_ROOT / "uploads" / "processed" / proc_filename
    assert proc_path.is_file(), f"Processed video file does not exist: {proc_path}"

    # Verify that the generated video is readable and valid
    cap = cv2.VideoCapture(str(proc_path))
    assert cap.isOpened(), f"Cannot open processed video: {proc_path}"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None, "Failed to read first frame from processed video"
    assert frame_count == 168

    # Verify original video URL and file existence
    orig_url = data.get("original_video_url")
    assert orig_url and orig_url.startswith("/original/"), f"Unexpected orig_url: {orig_url}"
    orig_filename = orig_url.replace("/original/", "")
    orig_path = PROJECT_ROOT / "uploads" / "original" / orig_filename
    assert orig_path.is_file(), f"Original video file does not exist: {orig_path}"
