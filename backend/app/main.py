from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from google.genai import errors as genai_errors

from app.pipeline.agent import run_pipeline
from app.schemas import AnalyzeResponse

app = FastAPI(title="Discern")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if content_type not in SUPPORTED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        # run_pipeline is synchronous/blocking (Gemini calls, OpenCV, YOLO inference);
        # run it off the event loop so other requests (e.g. health polling) aren't stalled.
        return await run_in_threadpool(run_pipeline, image_bytes, content_type)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except genai_errors.ClientError as e:
        if e.code == 429:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota exhausted (free tier allows a limited number of requests "
                    "per day for this model). Wait for the quota to reset, switch GEMINI_MODEL in "
                    "backend/.env to a model with more quota, or upgrade the API key's billing plan."
                ),
            )
        raise HTTPException(status_code=502, detail=f"Gemini rejected the request: {e}")
