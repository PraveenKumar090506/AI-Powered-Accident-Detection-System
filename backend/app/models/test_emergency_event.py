"""PASS/FAIL tests for the EmergencyEvent data model."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from emergency_event import EmergencyEvent


def check(name: str, condition: bool, detail: str = "") -> bool:
    suffix = f" - {detail}" if detail else ""
    if condition:
        print(f"PASS: {name}{suffix}")
        return True
    print(f"FAIL: {name}{suffix}")
    return False


def test_valid_event_creation() -> bool:
    event = EmergencyEvent(
        accident_suspected=True,
        confidence=0.81,
        involved_track_ids=[1, 4],
        reasons=["overlap confirmed across frames"],
        status="detected",
    )
    return check(
        "valid event creation",
        event.accident_suspected is True
        and event.confidence == 0.81
        and event.involved_track_ids == [1, 4]
        and event.status == "detected",
    )


def test_automatic_event_id() -> bool:
    event = EmergencyEvent(confidence=0.5)
    return check(
        "automatic event_id",
        isinstance(event.event_id, str) and len(event.event_id) > 0,
        event.event_id,
    )


def test_automatic_detected_at() -> bool:
    before = datetime.now(timezone.utc)
    event = EmergencyEvent(confidence=0.5)
    after = datetime.now(timezone.utc)
    detected_at = event.detected_at
    if detected_at.tzinfo is None:
        detected_at = detected_at.replace(tzinfo=timezone.utc)
    in_range = before <= detected_at <= after
    return check(
        "automatic detected_at",
        in_range,
        str(event.detected_at),
    )


def test_confidence_validation() -> bool:
    too_low_rejected = False
    too_high_rejected = False
    try:
        EmergencyEvent(confidence=-0.1)
    except ValidationError:
        too_low_rejected = True
    try:
        EmergencyEvent(confidence=1.2)
    except ValidationError:
        too_high_rejected = True
    valid = EmergencyEvent(confidence=0.0)
    also_valid = EmergencyEvent(confidence=1.0)
    return check(
        "confidence validation",
        too_low_rejected
        and too_high_rejected
        and valid.confidence == 0.0
        and also_valid.confidence == 1.0,
    )


def test_latitude_validation() -> bool:
    rejected = False
    try:
        EmergencyEvent(confidence=0.4, latitude=95.0)
    except ValidationError:
        rejected = True
    valid = EmergencyEvent(confidence=0.4, latitude=12.97)
    return check(
        "latitude validation",
        rejected and valid.latitude == 12.97,
    )


def test_longitude_validation() -> bool:
    rejected = False
    try:
        EmergencyEvent(confidence=0.4, longitude=200.0)
    except ValidationError:
        rejected = True
    valid = EmergencyEvent(confidence=0.4, longitude=77.59)
    return check(
        "longitude validation",
        rejected and valid.longitude == 77.59,
    )


def test_optional_evidence_and_location() -> bool:
    event = EmergencyEvent(confidence=0.6)
    with_values = EmergencyEvent(
        confidence=0.6,
        evidence_path="ml-model/evidence/example.png",
        latitude=12.97,
        longitude=77.59,
    )
    omitted = (
        event.evidence_path is None
        and event.latitude is None
        and event.longitude is None
    )
    provided = (
        with_values.evidence_path == "ml-model/evidence/example.png"
        and with_values.latitude == 12.97
        and with_values.longitude == 77.59
    )
    return check("optional evidence/location fields", omitted and provided)


def main() -> int:
    print("EmergencyEvent model tests")
    print()
    outcomes = [
        test_valid_event_creation(),
        test_automatic_event_id(),
        test_automatic_detected_at(),
        test_confidence_validation(),
        test_latitude_validation(),
        test_longitude_validation(),
        test_optional_evidence_and_location(),
    ]
    passed = sum(outcomes)
    total = len(outcomes)
    print()
    print(f"Summary: {passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
