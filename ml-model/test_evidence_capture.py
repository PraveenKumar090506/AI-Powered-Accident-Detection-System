"""Simple PASS/FAIL test for EvidenceCapture."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from accident_verifier import AccidentVerificationResult
from evidence_capture import EvidenceCapture

TEST_OUTPUT_DIR = SCRIPT_DIR / "evidence" / "_test_evidence_capture"


def check(name: str, condition: bool, detail: str = "") -> bool:
    suffix = f" - {detail}" if detail else ""
    if condition:
        print(f"PASS: {name}{suffix}")
        return True
    print(f"FAIL: {name}{suffix}")
    return False


def make_sample_frame() -> np.ndarray:
    """Build a small synthetic BGR frame (not from a video)."""
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :] = (40, 80, 160)
    return frame


def make_sample_result() -> AccidentVerificationResult:
    return AccidentVerificationResult(
        accident_suspected=True,
        confidence=0.72,
        involved_track_ids=(1, 2),
        reasons=("synthetic overlap for evidence test",),
    )


def main() -> int:
    print("Evidence capture tests")
    print()

    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)

    outcomes: list[bool] = []
    saved_path: Path | None = None

    try:
        capture = EvidenceCapture(output_dir=TEST_OUTPUT_DIR)
        frame = make_sample_frame()
        result = make_sample_result()
        saved_path = capture.save_evidence(frame, result)

        outcomes.append(
            check(
                "save_evidence returns a path",
                saved_path is not None,
            )
        )
        outcomes.append(
            check(
                "evidence PNG file exists",
                saved_path is not None and saved_path.is_file(),
                str(saved_path),
            )
        )
        outcomes.append(
            check(
                "saved file is a PNG",
                saved_path is not None and saved_path.suffix.lower() == ".png",
                saved_path.suffix if saved_path else "missing",
            )
        )

        loaded = cv2.imread(str(saved_path)) if saved_path else None
        outcomes.append(
            check(
                "saved PNG can be read back",
                loaded is not None and loaded.size > 0,
            )
        )
    except Exception as exc:
        outcomes.append(check("save_evidence completes without error", False, str(exc)))
    finally:
        if TEST_OUTPUT_DIR.exists():
            shutil.rmtree(TEST_OUTPUT_DIR)
        cleaned = not TEST_OUTPUT_DIR.exists()
        outcomes.append(check("test evidence cleaned up", cleaned, str(TEST_OUTPUT_DIR)))

    passed = sum(outcomes)
    total = len(outcomes)
    print()
    print(f"Summary: {passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
