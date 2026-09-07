"""PASS/FAIL tests for the mock EmergencyAlertService."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
SERVICES_DIR = Path(__file__).resolve().parent
for path in (str(APP_DIR), str(SERVICES_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from emergency_alert import AlertRecipient, EmergencyAlertService
from models.emergency_event import EmergencyEvent


def check(name: str, condition: bool, detail: str = "") -> bool:
    suffix = f" - {detail}" if detail else ""
    if condition:
        print(f"PASS: {name}{suffix}")
        return True
    print(f"FAIL: {name}{suffix}")
    return False


def _sample_setup() -> tuple[EmergencyAlertService, EmergencyEvent]:
    recipients = [
        AlertRecipient(
            recipient_id="police-1",
            recipient_type="police",
            name="Mock Police Station",
            contact="000-POLICE",
        ),
        AlertRecipient(
            recipient_id="ambulance-1",
            recipient_type="ambulance",
            name="Mock Ambulance Unit",
            contact="000-AMBULANCE",
        ),
    ]
    event = EmergencyEvent(
        event_id="evt-test-001",
        accident_suspected=True,
        confidence=0.91,
        involved_track_ids=[3, 7],
        reasons=["overlap confirmed", "sudden stop"],
        evidence_path="ml-model/evidence/clip-001.png",
        latitude=12.9716,
        longitude=77.5946,
        detected_at=datetime(2026, 9, 7, 17, 30, tzinfo=timezone.utc),
        status="detected",
    )
    return EmergencyAlertService(recipients), event


def main() -> int:
    print("EmergencyAlertService mock tests")
    print()

    service, event = _sample_setup()
    result = service.send_alert(event)
    notified_types = {item.recipient_type for item in result.recipients_notified}
    notified_ids = {item.recipient_id for item in result.recipients_notified}

    outcomes = [
        check(
            "both recipients included",
            result.number_of_recipients == 2
            and notified_types == {"police", "ambulance"}
            and notified_ids == {"police-1", "ambulance-1"},
        ),
        check(
            "event_id included",
            result.event_id == event.event_id and event.event_id in result.message,
            result.event_id,
        ),
        check(
            "confidence included",
            str(event.confidence) in result.message,
            str(event.confidence),
        ),
        check(
            "location included",
            str(event.latitude) in result.message and str(event.longitude) in result.message,
            f"{event.latitude}, {event.longitude}",
        ),
        check(
            "evidence path included",
            event.evidence_path is not None and event.evidence_path in result.message,
            event.evidence_path or "",
        ),
        check(
            "status indicates successful mock delivery",
            result.status == "mock_delivered",
            result.status,
        ),
    ]

    passed = sum(outcomes)
    total = len(outcomes)
    print()
    print(f"Summary: {passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
