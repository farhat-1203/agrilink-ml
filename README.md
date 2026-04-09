# AgriLink AI — ML Microservice

Production-quality, hackathon-friendly ML backend for a farmer-buyer marketplace targeting rural India (Maharashtra/Hindi-speaking regions).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data + train models (run once)
python scripts/train_all.py

# 3. Start the API server
uvicorn main:app --reload --port 8000

# 4. Open interactive docs
open http://localhost:8000/docs
```

---

## Folder Structure

```
agrilink-ml/
├── main.py                    ← FastAPI app (all endpoints wired here)
├── requirements.txt
│
├── modules/                   ← ML module implementations
│   ├── price_predictor.py     ← RandomForest price model
│   ├── demand_forecaster.py   ← Holt-Winters time-series forecast
│   ├── image_analyzer.py      ← OpenCV/PIL crop quality analysis
│   ├── nlp_parser.py          ← Regex + vocab voice→JSON parser
│   └── buyer_matcher.py       ← Multi-factor buyer scoring & ranking
│
├── utils/
│   └── schemas.py             ← Pydantic request/response models
│
├── scripts/
│   ├── generate_data.py       ← Synthetic dataset generation
│   ├── train_price_model.py   ← Train price RandomForest
│   └── train_all.py           ← One-shot: generate + train everything
│
├── data/
│   ├── crop_prices.csv        ← 3,000-row price training dataset
│   ├── demand_series.csv      ← 3-year daily demand time series
│   └── buyers.json            ← 10 sample buyer profiles
│
└── saved_models/
    ├── price_model.pkl        ← Trained RandomForest
    └── price_encoders.pkl     ← LabelEncoders for categorical features
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |
| POST | `/predict-price` | Predict crop price per kg |
| POST | `/forecast` | 7-day demand forecast |
| POST | `/analyze-image` | Crop quality from base64 image |
| POST | `/analyze-image-upload` | Crop quality from file upload |
| POST | `/parse-voice` | Voice utterance → structured JSON |
| POST | `/match-buyers` | Rank best buyers for a listing |
| POST | `/pipeline/voice-to-match` | End-to-end: voice → price → buyers |

---

## Request / Response Examples

### POST /predict-price
```json
// Request
{
  "crop": "tomato",
  "season": "kharif",
  "market": "Nashik",
  "quality_grade": "A",
  "quantity_quintals": 5,
  "rainfall_mm": 80,
  "days_to_market": 1
}

// Response
{
  "predicted_price_per_kg": 21.64,
  "price_range": { "low": 19.04, "high": 25.31 },
  "confidence": 0.71,
  "currency": "INR"
}
```

### POST /forecast
```json
// Request
{ "crop": "tomato", "steps": 7 }

// Response
{
  "crop": "tomato",
  "trend": "stable",
  "forecast_method": "Holt-Winters Exponential Smoothing",
  "daily_forecast": [
    { "date": "2026-04-07", "day": "Tuesday", "forecast_quintals": 593.2 },
    ...
  ],
  "chart_data": {
    "labels": ["Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"],
    "values": [593.2, 589.3, 588.7, 538.1, 591.3, 588.0, 584.5],
    "unit": "quintals"
  }
}
```

### POST /analyze-image
```json
// Request
{ "image_b64": "<base64 string>", "crop": "tomato" }

// Response
{
  "quality_grade": "A",
  "grade_label": "Premium",
  "freshness_score": 87.2,
  "shelf_life_days": 5,
  "recommendation": "Excellent quality. Suitable for premium market / direct export."
}
```

### POST /parse-voice
```json
// Request
{ "text": "2 quintal tomatoes fresh selling tomorrow nashik" }

// Response
{
  "crop": "tomato",
  "quantity_kg": 200.0,
  "quantity_display": "2.0 quintal",
  "quality_grade": "A",
  "available_date": "2026-04-07",
  "target_market": "Nashik",
  "asking_price_per_kg": null,
  "confidence": 1.0,
  "missing_fields": []
}
```

### POST /match-buyers
```json
// Request
{
  "crop": "tomato",
  "quantity_kg": 200,
  "farmer_lat": 19.99,
  "farmer_lon": 73.78,
  "market_price_per_kg": 18.5,
  "top_n": 3
}

// Response
{
  "total_buyers_evaluated": 10,
  "matches": [
    {
      "name": "Mumbai Fresh Mart",
      "distance_km": 36.1,
      "offered_price_per_kg": 18.68,
      "match_score": 56.96,
      "match_label": "Good",
      "payment_days": 7,
      "score_breakdown": {
        "price_score": 18.5,
        "quantity_score": 25.0,
        "distance_score": 11.5,
        "demand_score": 15.0,
        "total": 56.96
      }
    }
  ]
}
```

### POST /pipeline/voice-to-match  ← Star endpoint for demo
```json
// Request
{ "text": "3 quintal fresh onion nashik tomorrow price 22 rupees" }

// Response
{
  "status": "success",
  "parsed_listing": { "crop": "onion", "quantity_kg": 300, ... },
  "price_analysis": { "predicted_price_per_kg": 22.11, ... },
  "top_buyers": [ ... ],
  "summary": "Your onion (3.0 quintal) is worth approx ₹22.11/kg. Best buyer: Nashik Mandi Co. offering ₹26.39/kg."
}
```

---

## Frontend Integration (JavaScript)

```javascript
const BASE_URL = "http://localhost:8000";

// Price prediction
async function predictPrice(crop, season, market, qty) {
  const res = await fetch(`${BASE_URL}/predict-price`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ crop, season, market,
      quality_grade: "B", quantity_quintals: qty,
      rainfall_mm: 80, days_to_market: 1 })
  });
  return res.json();
}

// Demand chart data (ready for Chart.js / Recharts)
async function getDemandForecast(crop) {
  const res = await fetch(`${BASE_URL}/forecast`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ crop, steps: 7 })
  });
  const data = await res.json();
  return data.chart_data;   // { labels, values, unit }
}

// Image quality (mobile camera)
async function analyzeImage(base64Image, crop) {
  const res = await fetch(`${BASE_URL}/analyze-image`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_b64: base64Image, crop })
  });
  return res.json();
}

// Voice → full pipeline (best for demo!)
async function voiceToMatch(transcribedText) {
  const res = await fetch(`${BASE_URL}/pipeline/voice-to-match`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: transcribedText })
  });
  return res.json();
}
```

---

## ML Model Details

| Module | Algorithm | Accuracy | Notes |
|--------|-----------|----------|-------|
| Price Predictor | RandomForest (200 trees) | MAE ₹2.80, R²=0.84 | Trained on 3,000 samples |
| Demand Forecast | Holt-Winters ETS | Weekly seasonality | 3-year daily history |
| Image Quality | OpenCV HSV heuristic | Rule-based | Plug-in ONNX slot available |
| NLP Parser | Regex + vocabulary | conf 0.8 avg | Hindi/Marathi vocab included |
| Buyer Matcher | Scoring function | N/A | 4-factor weighted ranking |

---

## Extending for Production

- **Real price data**: Replace `data/crop_prices.csv` with Agmarknet / eNAM API data
- **Image model**: Drop an ONNX model into `saved_models/` and call `onnxruntime` in `image_analyzer.py`
- **Hindi NLP**: Plug in `IndicNLP` or `Bhashini API` for better Devanagari parsing  
- **Offline support**: Bundle models with `joblib` + SQLite; sync on reconnect
- **Prophet**: Swap `ExponentialSmoothing` for `prophet` for better holiday/festival handling
