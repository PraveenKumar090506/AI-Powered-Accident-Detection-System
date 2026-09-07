from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from .models.emergency_event import EmergencyEvent
from .services.emergency_alert import AlertRecipient, EmergencyAlertService
from .services.location_service import LocationProvider
from .video_analysis import InvalidVideoError, PROJECT_ROOT, analyze_video_file

DEFAULT_MOCK_RECIPIENTS = [
    AlertRecipient(
        recipient_id="police-mock-1",
        recipient_type="police",
        name="Mock Police Station",
        contact="000-POLICE",
    ),
    AlertRecipient(
        recipient_id="ambulance-mock-1",
        recipient_type="ambulance",
        name="Mock Ambulance Unit",
        contact="000-AMBULANCE",
    ),
]

UPLOADS_DIR = PROJECT_ROOT / "uploads"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

app = FastAPI(
    title="Accident Detection API",
    description="AI-Powered Accident Detection and Emergency Alert System",
    version="0.1.0",
)


@app.get("/")
def welcome():
    return {
        "message": "Welcome to the AI-Powered Accident Detection and Emergency Alert System API"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Accident Detection API is running",
    }


@app.post("/emergency-alert")
def emergency_alert(event: EmergencyEvent):
    """Simulate a mock emergency alert. Does not contact real services."""
    if not event.accident_suspected:
        return {
            "event_id": event.event_id,
            "number_of_recipients": 0,
            "recipients_notified": [],
            "message": "No alert sent because accident_suspected is false.",
            "status": "not_sent",
        }

    location = LocationProvider().get_location()
    event = event.model_copy(
        update={
            "latitude": event.latitude
            if event.latitude is not None
            else location["latitude"],
            "longitude": event.longitude
            if event.longitude is not None
            else location["longitude"],
        }
    )

    service = EmergencyAlertService(DEFAULT_MOCK_RECIPIENTS)
    result = service.send_alert(event)
    response = result.model_dump()
    response["latitude"] = event.latitude
    response["longitude"] = event.longitude
    return response


@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """Upload a video, run tracking + verification, and return the result."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded file")

    suffix = Path(file.filename).suffix.lower()
    if suffix and suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Invalid uploaded file: expected a video",
        )

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = UPLOADS_DIR / f"{uuid4().hex}{suffix or '.mp4'}"

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        temp_path.write_bytes(contents)

        try:
            return analyze_video_file(temp_path)
        except InvalidVideoError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Video processing failed: {exc}",
            ) from exc
    finally:
        await file.close()
        temp_path.unlink(missing_ok=True)
