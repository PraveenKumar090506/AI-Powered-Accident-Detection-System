"""Mock emergency alert service. Does not contact real emergency services."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

try:
    from app.models.emergency_event import EmergencyEvent
except ImportError:
    from models.emergency_event import EmergencyEvent

RecipientType = Literal["police", "ambulance"]
MOCK_ALERT_STATUS = "mock_delivered"


class AlertRecipient(BaseModel):
    """Mock alert recipient. Contact is stored only; never used to send messages."""

    recipient_id: str
    recipient_type: RecipientType
    name: str
    contact: str


class AlertResult(BaseModel):
    """Structured result of a simulated alert delivery."""

    event_id: str
    number_of_recipients: int = Field(ge=0)
    recipients_notified: list[AlertRecipient]
    message: str
    status: str


class EmergencyAlertService:
    """Simulates sending an accident alert to police and ambulance recipients.

    This is mock-only. It never sends SMS, email, phone calls, WhatsApp
    messages, or contacts real emergency services.
    """

    def __init__(self, recipients: list[AlertRecipient]) -> None:
        self.recipients = list(recipients)

    def _build_message(self, event: EmergencyEvent) -> str:
        if event.latitude is not None and event.longitude is not None:
            location = f"{event.latitude}, {event.longitude}"
        elif event.latitude is not None:
            location = f"lat={event.latitude}, lon=unavailable"
        elif event.longitude is not None:
            location = f"lat=unavailable, lon={event.longitude}"
        else:
            location = "unavailable"

        vehicle_ids = ", ".join(str(track_id) for track_id in event.involved_track_ids) or "none"
        reasons = "; ".join(event.reasons) or "none"
        evidence = event.evidence_path or "unavailable"

        return (
            "MOCK EMERGENCY ALERT - not sent to real services\n"
            f"event_id: {event.event_id}\n"
            f"accident_confidence: {event.confidence}\n"
            f"location: {location}\n"
            f"detected_at: {event.detected_at.isoformat()}\n"
            f"involved_vehicle_ids: {vehicle_ids}\n"
            f"reasons: {reasons}\n"
            f"evidence_path: {evidence}"
        )

    def send_alert(self, event: EmergencyEvent) -> AlertResult:
        """Build an alert message and simulate delivery to configured recipients."""
        message = self._build_message(event)
        print(
            f"[MOCK ALERT] Simulated delivery of event {event.event_id} "
            f"to {len(self.recipients)} recipient(s). No real notification sent."
        )
        print(message)
        return AlertResult(
            event_id=event.event_id,
            number_of_recipients=len(self.recipients),
            recipients_notified=list(self.recipients),
            message=message,
            status=MOCK_ALERT_STATUS,
        )
