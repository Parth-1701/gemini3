import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import AnalyzeRequest, LabReport, UploadResponse
from . import gemini


ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".webm",
    ".mkv",
    ".m4v"
}


app = FastAPI(
    title="Virtual Lab Analysis Engine API",
    description="Backend for Gemini-powered lab video analysis.",
    version="1.0.0"
)


cors_origins = os.getenv(
    "BACKEND_CORS_ORIGINS",
    "http://localhost:3000"
)

origins = [
    origin.strip()
    for origin in cors_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Virtual Lab Analysis Engine API is running.",
        "docs": "/docs",
        "health": "/healthz"
    }


@app.get("/healthz")
def healthz():
    return {
        "status": "ok"
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    filename = file.filename or ""
    extension = os.path.splitext(filename)[1].lower()
    content_type = file.content_type or ""

    is_valid_video = (
        content_type.startswith("video/")
        or extension in ALLOWED_VIDEO_EXTENSIONS
    )

    if not is_valid_video:
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid video file."
        )

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension or ".mp4"
        ) as tmp_file:
            tmp_path = tmp_file.name
            shutil.copyfileobj(file.file, tmp_file)

        return gemini.upload_file_to_gemini(
            local_path=tmp_path,
            display_name=filename or "lab-video.mp4",
            mime_type=content_type or "video/mp4"
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Video upload failed: {str(exc)}"
        ) from exc

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/analyze", response_model=LabReport)
async def analyze_video(request: AnalyzeRequest):
    try:
        return gemini.generate_lab_report(
            file_uri=request.file_uri,
            mime_type=request.mime_type
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {str(exc)}"
        ) from exc
