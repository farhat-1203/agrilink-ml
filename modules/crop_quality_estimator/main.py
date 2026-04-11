"""
main.py - FastAPI Application for Crop Quality Estimator

Endpoints:
    POST /analyze-image    - accepts multipart file upload
    POST /analyze-b64      - accepts base64 image in JSON body
    GET  /health           - liveness probe
    GET  /classes          - list known disease classes
    GET  /docs             - Swagger UI

Run from crop_quality_estimator directory:
    uvicorn main:app --reload --port 8000
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add current directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import traceback

from modules.inference import get_engine
from config import CHECKPOINT


# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE = os.getenv("DEVICE", "cpu")          # set DEVICE=cuda for GPU
MODEL  = os.getenv("MODEL_PATH", str(CHECKPOINT))


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """Request body for the base64 endpoint."""
    image_b64 : str = Field(
        ...,
        description=(
            "Base64-encoded image. Accepts JPEG, PNG, WebP. "
            "Data-URI prefix (data:image/jpeg;base64,...) is automatically stripped."
        ),
        example="<base64 string here>",
    )
    crop_hint : Optional[str] = Field(
        None,
        description="Optional crop type (e.g. 'tomato'). Logged but not used by the model.",
        example="tomato",
    )


class DiseaseEntry(BaseModel):
    label      : str
    confidence : float  # %


class AnalyzeResponse(BaseModel):
    # ── Core outputs ──────────────────────────────────────────────────────────
    quality_grade      : str    # A / B / C
    grade_label        : str    # Premium / Standard / Below Standard
    shelf_life_days    : int
    urgency_level      : str    # low / medium / high
    freshness_score    : float  # 0–100
    # ── Disease ───────────────────────────────────────────────────────────────
    is_diseased        : bool
    disease_label      : str
    disease_confidence : float  # %
    top3_predictions   : List[DiseaseEntry]
    # ── Signals ───────────────────────────────────────────────────────────────
    freshness_signals  : Dict[str, float]
    image_stats        : Dict[str, float]
    # ── Recommendation ────────────────────────────────────────────────────────
    market_tier        : str
    summary            : str
    # ── Meta ──────────────────────────────────────────────────────────────────
    latency_ms         : float
    mode               : str    # nn+heuristic | heuristic_only


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "AgriLink AI - Crop Quality Estimator",
    description = (
        "Single-photo crop quality analysis powered by MobileNetV3 disease "
        "detection + OpenCV freshness heuristics.\n\n"
        "**What it returns per image:**\n"
        "- Quality grade (A / B / C)\n"
        "- Shelf life estimate (days)\n"
        "- Urgency level (low / medium / high)\n"
        "- Detected disease + confidence\n"
        "- Freshness score 0-100\n"
        "- 6 raw CV signals (saturation, greenness, browning, sharpness, etc.)\n"
        "- Market recommendation"
    ),
    version = "1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Startup — pre-load model ───────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Load the model into memory before the first request arrives.
    Falls back to heuristic-only mode if no checkpoint file exists yet.
    """
    print("[startup] Loading inference engine...")
    get_engine(checkpoint_path=MODEL, device=DEVICE)
    print("[startup] Ready.")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", tags=["system"], summary="Liveness probe")
def health():
    eng = get_engine(MODEL, DEVICE)
    return {
        "status"         : "ok",
        "model_loaded"   : eng.model is not None,
        "classes"        : len(eng.class_names),
        "inference_mode" : "nn+heuristic" if eng.model else "heuristic_only",
        "device"         : str(eng.device),
    }


@app.get("/classes", tags=["system"], summary="List all known disease classes")
def list_classes():
    eng = get_engine(MODEL, DEVICE)
    return {
        "num_classes" : len(eng.class_names),
        "classes"     : eng.class_names,
    }


@app.post(
    "/analyze-image",
    response_model = AnalyzeResponse,
    tags           = ["prediction"],
    summary        = "Analyze crop image (file upload)",
)
async def analyze_image(
    file      : UploadFile = File(..., description="Crop photo — JPEG, PNG, or WebP"),
    crop_hint : str        = Form("",  description="Optional crop type hint"),
):
    """
    Analyze crop quality from a **multipart file upload**.
    
    Returns quality grade (A/B/C), shelf life, disease detection,
    freshness score (0-100), and market recommendation.

    ```bash
    curl -X POST http://localhost:8000/analyze-image \\
         -F "file=@tomato.jpg;type=image/jpeg" \\
         -F "crop_hint=tomato"
    ```
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")
    try:
        data   = await file.read()
        result = get_engine(MODEL, DEVICE).predict(data)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Image analysis failed: {e}\n{traceback.format_exc()}",
        )


@app.post(
    "/analyze-b64",
    response_model = AnalyzeResponse,
    tags           = ["prediction"],
    summary        = "Analyze crop image (base64)",
    responses      = {
        200: {"description": "Quality report for the submitted image"},
        422: {"description": "Invalid or unreadable image"},
    },
)
def analyze_b64(req: AnalyzeRequest):
    """
    Analyze crop quality from a **base64-encoded image**.

    ```python
    import requests, base64

    with open("tomato.jpg", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    resp = requests.post(
        "http://localhost:8000/analyze-b64",
        json={"image_b64": b64, "crop_hint": "tomato"}
    )
    print(resp.json())
    ```
    """
    try:
        result = get_engine(MODEL, DEVICE).predict(req.image_b64)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Image analysis failed: {e}\n{traceback.format_exc()}",
        )


# ── Run directly ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = int(os.getenv("PORT", 8000)),
        reload  = True,
    )
