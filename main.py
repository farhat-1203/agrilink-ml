"""
AgriLink AI — ML Microservice
FastAPI application wiring all 5 ML modules into a single deployable service.

Run:
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /predict-price
    POST /forecast
    POST /analyze-image
    POST /parse-voice
    POST /match-buyers
    GET  /health
    GET  /docs  (auto-generated Swagger UI)
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback


from utils.schemas import (
    PricePredictRequest, DemandForecastRequest, ImageAnalyzeRequest,
    ParseVoiceRequest, MatchBuyersRequest,
)
from modules.price_predictor  import predict_price
from modules.demand_forecaster import forecast_demand
# from modules.image_analyzer    import analyze_image, analyze_image_b64
from modules.nlp_parser        import parse_voice_input
from modules.buyer_matcher     import match_buyers
from modules.stt_module        import get_speech_to_text_module
from modules.stt_module import get_speech_to_text_module

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AgriLink AI — ML Service",
    description=(
        "ML microservice for AgriLink: price prediction, demand forecasting, "
        "image quality analysis, voice NLP parsing, and buyer matching."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health():
    return {
        "status": "ok",
        "service": "AgriLink AI ML",
        "version": "1.0.0",
        "modules": ["price_predictor", "demand_forecaster",
                    "image_analyzer", "nlp_parser", "buyer_matcher"],
    }


# ─── STEP 2: Price Prediction ─────────────────────────────────────────────────
@app.post("/predict-price", tags=["price"])
def api_predict_price(req: PricePredictRequest):
    """
    Predict price per kg for a crop listing.

    **Example request:**
    ```json
    {
      "crop": "tomato",
      "season": "kharif",
      "market": "Nashik",
      "quality_grade": "A",
      "quantity_quintals": 5,
      "rainfall_mm": 80,
      "days_to_market": 1
    }
    ```
    """
    try:
        result = predict_price(
            crop=req.crop, season=req.season, market=req.market,
            quality_grade=req.quality_grade,
            quantity_quintals=req.quantity_quintals,
            rainfall_mm=req.rainfall_mm,
            days_to_market=req.days_to_market,
        )
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Price model not trained yet. Run: python scripts/train_price_model.py"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── STEP 3: Demand Forecasting ───────────────────────────────────────────────
@app.post("/forecast", tags=["demand"])
def api_forecast(req: DemandForecastRequest):
    """
    Forecast crop demand for the next N days (default 7).

    **Example request:**
    ```json
    { "crop": "tomato", "steps": 7 }
    ```
    """
    try:
        return forecast_demand(crop=req.crop, steps=req.steps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── STEP 4: Image Quality Analysis ──────────────────────────────────────────
# @app.post("/analyze-image", tags=["quality"])
# def api_analyze_image(req: ImageAnalyzeRequest):
#     """
#     Analyze crop image for quality grade, freshness, and shelf life.
#     Accepts base64-encoded image string.

#     **Example request:**
#     ```json
#     {
#       "image_b64": "<base64 string>",
#       "crop": "tomato"
#     }
#     ```
#     """
#     try:
#         return analyze_image_b64(req.image_b64, req.crop)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")


# @app.post("/analyze-image-upload", tags=["quality"])
# async def api_analyze_image_upload(
#     file: UploadFile = File(...),
#     crop: str = Form("generic")
# ):
#     """
#     Analyze crop image quality via multipart file upload (mobile-friendly).
#     """
#     try:
#         image_data = await file.read()
#         return analyze_image(image_data, crop)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")


# ─── STEP 5: NLP Voice Parser ─────────────────────────────────────────────────
@app.post("/parse-voice", tags=["nlp"])
def api_parse_voice(req: ParseVoiceRequest):
    """
    Convert a farmer's voice utterance (after STT) into a structured crop listing.

    **Example request:**
    ```json
    { "text": "2 quintal tomatoes fresh selling tomorrow nashik" }
    ```
    """
    try:
        return parse_voice_input(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── STEP 5B: Voice Audio → STT → NLP ────────────────────────────────────────
@app.post("/voice-audio", tags=["nlp"])
async def api_voice_audio(file: UploadFile = File(...)):
    """
    Upload audio → convert to text → extract crop details
    
    **Example usage:**
    Upload a WAV/audio file and get both transcription and parsed crop details.
    """
    try:
        audio_bytes = await file.read()
        
        # STT
        stt = get_speech_to_text_module()
        text = await stt.transcribe(audio_bytes)
        
        # NLP parsing
        parsed = parse_voice_input(text)
        
        return {
            "transcription": text,
            "parsed": parsed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── STEP 6: Buyer Matching ───────────────────────────────────────────────────
@app.post("/match-buyers", tags=["matching"])
def api_match_buyers(req: MatchBuyersRequest):
    """
    Find and rank best buyers for a farmer's crop listing.

    **Example request:**
    ```json
    {
      "crop": "tomato",
      "quantity_kg": 200,
      "farmer_lat": 19.99,
      "farmer_lon": 73.78,
      "market_price_per_kg": 18.5,
      "top_n": 5
    }
    ```
    """
    try:
        return match_buyers(
            crop=req.crop, quantity_kg=req.quantity_kg,
            farmer_lat=req.farmer_lat, farmer_lon=req.farmer_lon,
            market_price_per_kg=req.market_price_per_kg,
            top_n=req.top_n,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Combined Pipeline: Voice → Price → Match ─────────────────────────────────
@app.post("/pipeline/voice-to-match", tags=["pipeline"])
def api_voice_to_match(req: ParseVoiceRequest):
    """
    End-to-end pipeline for voice-first farmers:
    1. Parse voice text → structured listing
    2. Predict market price
    3. Find best buyers

    Single API call for the full flow.
    """
    try:
        # Step 1: Parse voice
        parsed = parse_voice_input(req.text)
        if not parsed["crop"] or not parsed["quantity_kg"]:
            return {
                "status": "incomplete_parse",
                "parsed": parsed,
                "message": f"Could not extract: {parsed['missing_fields']}. Please repeat more clearly."
            }

        # Step 2: Predict price
        price_result = predict_price(
            crop=parsed["crop"],
            season="kharif",      # default; could be inferred from date
            market=parsed["target_market"] or "Nashik",
            quality_grade=parsed["quality_grade"] or "B",
            quantity_quintals=(parsed["quantity_kg"] or 100) / 100,
            rainfall_mm=80.0,
            days_to_market=1,
        )

        # Step 3: Match buyers (use farmer default location if GPS not in text)
        buyer_result = match_buyers(
            crop=parsed["crop"],
            quantity_kg=parsed["quantity_kg"],
            farmer_lat=19.5,    # default Nashik region
            farmer_lon=73.8,
            market_price_per_kg=price_result["predicted_price_per_kg"],
            top_n=3,
        )

        return {
            "status":        "success",
            "parsed_listing": parsed,
            "price_analysis": price_result,
            "top_buyers":     buyer_result["matches"],
            "summary": (
                f"Your {parsed['crop']} ({parsed['quantity_display']}) is worth "
                f"approximately ₹{price_result['predicted_price_per_kg']}/kg. "
                f"Best buyer: {buyer_result['best_match']['name']} "
                f"offering ₹{buyer_result['best_match']['offered_price_per_kg']}/kg."
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
