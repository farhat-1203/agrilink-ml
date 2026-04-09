"""Pydantic v2 request / response models for AgriLink AI ML API."""
from pydantic import BaseModel, Field
from typing import Optional

# ── Price Prediction ──────────────────────────────────────────────────────────
class PricePredictRequest(BaseModel):
    crop:              str   = Field(..., example="tomato")
    season:            str   = Field(..., example="kharif")
    market:            str   = Field(..., example="Nashik")
    quality_grade:     str   = Field("B", example="A")
    quantity_quintals: float = Field(..., example=5.0)
    rainfall_mm:       float = Field(80.0, example=80.0)
    days_to_market:    int   = Field(1, example=1)

# ── Demand Forecast ───────────────────────────────────────────────────────────
class DemandForecastRequest(BaseModel):
    crop:  str = Field(..., example="tomato")
    steps: int = Field(7, ge=1, le=30)

# ── Image Quality ─────────────────────────────────────────────────────────────
class ImageAnalyzeRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image (JPEG/PNG/WebP)")
    crop:      str = Field("generic", example="tomato")

# ── NLP Parser ────────────────────────────────────────────────────────────────
class ParseVoiceRequest(BaseModel):
    text: str = Field(..., example="2 quintal tomatoes fresh selling tomorrow")

# ── Buyer Matching ────────────────────────────────────────────────────────────
class MatchBuyersRequest(BaseModel):
    crop:                str   = Field(..., example="tomato")
    quantity_kg:         float = Field(..., example=200.0)
    farmer_lat:          float = Field(..., example=19.99)
    farmer_lon:          float = Field(..., example=73.78)
    market_price_per_kg: float = Field(..., example=18.5)
    top_n:               int   = Field(5, ge=1, le=20)
