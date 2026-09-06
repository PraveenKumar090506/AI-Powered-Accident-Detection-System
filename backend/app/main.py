from fastapi import FastAPI

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
