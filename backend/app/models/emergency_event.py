"""Pydantic data model for a confirmed accident emergency event."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

# Simple MVP lifecycle for an emergency event.
EmergencyStatus = Literal["detected", "alert_pending", "alert_sent", "resolved"]


class EmergencyEvent(BaseModel):
    """In-memory emergency event. This is a data model only."""

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    accident_suspected: bool = True
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    involved_track_ids: list[int] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    evidence_path: str | None = None
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: EmergencyStatus = "detected"
