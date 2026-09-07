"""PASS/FAIL API tests for POST /emergency-alert and GET /health."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def check(name: str, condition: bool, detail: str = "") -> bool:
    suffix = f" - {detail}" if detail else ""
    if condition:
        print(f"PASS: {name}{suffix}")
        return True
    print(f"FAIL: {name}{suffix}")
    return False


def test_health() -> bool:
    response = client.get("/health")
    body = response.json()
    return check(
        "GET /health still works",
        response.status_code == 200
        and body.get("status") == "ok"
        and "Accident Detection API is running" in body.get("message", ""),
        str(body),
    )


def test_confirmed_accident_sends_mock_alert() -> bool:
    payload = {
        "event_id": "evt-api-001",
        "accident_suspected": True,
        "confidence": 0.88,
        "involved_track_ids": [2, 9],
        "reasons": ["overlap confirmed"],
        "evidence_path": "ml-model/evidence/clip-api.png",
        "latitude": 12.97,
        "longitude": 77.59,
        "status": "detected",
    }
    response = client.post("/emergency-alert", json=payload)
    body = response.json()
    recipients = body.get("recipients_notified") or []
    types = {item.get("recipient_type") for item in recipients}
    ok = (
        response.status_code == 200
        and body.get("event_id") == "evt-api-001"
        and body.get("status") == "mock_delivered"
        and body.get("number_of_recipients") == 2
        and types == {"police", "ambulance"}
        and "evt-api-001" in (body.get("message") or "")
        and "0.88" in (body.get("message") or "")
    )
    return check("confirmed accident -> mock alert returned", ok, str(body))


def test_non_accident_does_not_send_alert() -> bool:
    payload = {
        "event_id": "evt-api-002",
        "accident_suspected": False,
        "confidence": 0.12,
        "involved_track_ids": [],
        "reasons": [],
        "status": "detected",
    }
    response = client.post("/emergency-alert", json=payload)
    body = response.json()
    ok = (
        response.status_code == 200
        and body.get("event_id") == "evt-api-002"
        and body.get("status") == "not_sent"
        and body.get("number_of_recipients") == 0
        and body.get("recipients_notified") == []
        and "no alert sent" in (body.get("message") or "").lower()
    )
    return check("non-accident -> alert is not sent", ok, str(body))


def main() -> int:
    print("Emergency alert API tests")
    print()
    outcomes = [
        test_health(),
        test_confirmed_accident_sends_mock_alert(),
        test_non_accident_does_not_send_alert(),
    ]
    passed = sum(outcomes)
    total = len(outcomes)
    print()
    print(f"Summary: {passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
