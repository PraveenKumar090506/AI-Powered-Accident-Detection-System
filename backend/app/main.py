from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from .video_analysis import InvalidVideoError, PROJECT_ROOT, analyze_video_file

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
