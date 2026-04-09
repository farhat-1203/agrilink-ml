"""
AgriLink AI — Voice-to-Structured-Data NLP Parser
Converts farmer speech (after STT) into a structured listing JSON.

Examples:
  "2 quintal tomatoes fresh selling tomorrow"
  "मेरे पास 5 क्विंटल प्याज है"
  "50 kilo cabbage grade A, Nashik market"
  "3 bag potato available today price 20 rupees"

Approach: Rule-based regex + vocabulary lookup (no GPU, runs on 2G phone data)
Fallback:  LLM prompt template (when an API key is available)
"""
import re
from datetime import date, timedelta
from typing import Optional


# ─── Vocabulary Tables ────────────────────────────────────────────────────────

CROP_VOCAB = {
    # English
    "tomato": "tomato", "tomatoes": "tomato",
    "onion": "onion", "onions": "onion",
    "potato": "potato", "potatoes": "potato",
    "cabbage": "cabbage", "carrot": "carrot", "carrots": "carrot",
    "wheat": "wheat", "rice": "rice", "sugarcane": "sugarcane",
    # Hindi transliteration
    "tamatar": "tomato", "pyaz": "onion", "aloo": "potato",
    "gobhi": "cabbage", "gajar": "carrot", "gehun": "wheat",
    "chawal": "rice", "ganna": "sugarcane",
    # Marathi transliteration
    "tomato": "tomato", "kanda": "onion", "batata": "potato",
    "kobi": "cabbage", "gajar": "carrot",
}

UNIT_VOCAB = {
    # Quintals → kg factor
    "quintal": 100, "quintals": 100, "qtl": 100, "q": 100,
    "क्विंटल": 100, "kvinthal": 100,
    # Kg
    "kg": 1, "kilo": 1, "kilos": 1, "kilogram": 1,
    # Bag (approx 50 kg)
    "bag": 50, "bags": 50, "bori": 50,
    # Ton
    "ton": 1000, "tonne": 1000,
}

QUALITY_VOCAB = {
    "premium": "A", "fresh": "A", "first": "A", "grade a": "A", "grade-a": "A",
    "good": "B", "standard": "B", "medium": "B", "grade b": "B",
    "mixed": "C", "reject": "C", "third": "C", "grade c": "C",
}

TIME_VOCAB = {
    "today":     0,  "aaj": 0,
    "tomorrow":  1,  "kal": 1,   "kaल": 1,
    "day after": 2,  "parson": 2,
    "this week": 3,  "week": 5,
    "monday": None, "tuesday": None, "wednesday": None, "thursday": None,
    "friday": None, "saturday": None, "sunday": None,
}

MARKET_VOCAB = {
    "mumbai": "Mumbai", "pune": "Pune", "nashik": "Nashik",
    "nagpur": "Nagpur", "aurangabad": "Aurangabad", "kolhapur": "Kolhapur",
    "mandi": None,   # generic market keyword → leave blank
}


# ─── Parser ───────────────────────────────────────────────────────────────────

def parse_voice_input(text: str) -> dict:
    """
    Parse a free-form farmer utterance into a structured crop listing.

    Returns:
        {
          "crop": str | None,
          "quantity_kg": float | None,
          "quantity_display": str,          # "2 quintal", "50 kg", etc.
          "quality_grade": "A"|"B"|"C"|None,
          "available_date": "YYYY-MM-DD"|None,
          "target_market": str | None,
          "asking_price_per_kg": float | None,
          "raw_text": str,
          "confidence": float,             # 0–1
          "missing_fields": list[str],
          "parsed_tokens": dict            # what was extracted from each field
        }
    """
    raw = text
    text_lower = text.lower().strip()
    tokens: dict = {}
    confidence_hits = 0
    total_fields = 5  # crop, qty, quality, date, market

    # ── 1. Crop ───────────────────────────────────────────────────────────────
    crop = None
    for word, canonical in CROP_VOCAB.items():
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            crop = canonical
            tokens["crop_match"] = word
            confidence_hits += 1
            break

    # ── 2. Quantity + Unit ────────────────────────────────────────────────────
    quantity_kg   = None
    quantity_disp = None

    # Pattern: "2 quintal" / "50 kg" / "३ क्विंटल" (devanagari digits)
    # Devanagari digit map
    deva = str.maketrans("०१२३४५६७८९", "0123456789")
    norm_text = text_lower.translate(deva)

    qty_pattern = r'(\d+(?:\.\d+)?)\s*(' + '|'.join(re.escape(u) for u in UNIT_VOCAB) + r')'
    m = re.search(qty_pattern, norm_text)
    if m:
        qty_raw   = float(m.group(1))
        unit      = m.group(2)
        unit_kg   = UNIT_VOCAB.get(unit, 1)
        quantity_kg   = qty_raw * unit_kg
        quantity_disp = f"{qty_raw} {unit}"
        tokens["quantity"] = {"raw": qty_raw, "unit": unit, "unit_kg": unit_kg}
        confidence_hits += 1
    else:
        # Try bare number (assume quintals)
        m2 = re.search(r'\b(\d+(?:\.\d+)?)\b', norm_text)
        if m2:
            quantity_kg   = float(m2.group(1)) * 100
            quantity_disp = f"{m2.group(1)} quintal (assumed)"
            tokens["quantity"] = {"raw": float(m2.group(1)), "unit": "quintal (assumed)"}

    # ── 3. Quality ────────────────────────────────────────────────────────────
    quality = None
    for phrase, grade in QUALITY_VOCAB.items():
        if phrase in text_lower:
            quality = grade
            tokens["quality_match"] = phrase
            confidence_hits += 1
            break

    # ── 4. Date / Availability ────────────────────────────────────────────────
    avail_date = None
    today = date.today()
    for phrase, delta in TIME_VOCAB.items():
        if phrase in text_lower:
            if delta is not None:
                avail_date = (today + timedelta(days=delta)).isoformat()
            else:
                # Weekday name → find next occurrence
                days_map = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,
                            "friday":4,"saturday":5,"sunday":6}
                target_dow = days_map.get(phrase)
                if target_dow is not None:
                    days_ahead = (target_dow - today.weekday()) % 7 or 7
                    avail_date = (today + timedelta(days=days_ahead)).isoformat()
            tokens["date_match"] = phrase
            confidence_hits += 1
            break

    # ── 5. Market ─────────────────────────────────────────────────────────────
    market = None
    for word, canonical in MARKET_VOCAB.items():
        if word in text_lower and canonical:
            market = canonical
            tokens["market_match"] = word
            confidence_hits += 1
            break

    # ── 6. Price (optional) ───────────────────────────────────────────────────
    price_per_kg = None
    price_pattern = r'(?:price|rate|Rs|₹|rupee[s]?)[\s:]*(\d+(?:\.\d+)?)'
    pm = re.search(price_pattern, text_lower)
    if not pm:
        # Try: "at 20" / "@20" / "20 rupees"
        pm = re.search(r'(?:at|@)\s*(\d+(?:\.\d+)?)', text_lower)
    if not pm:
        pm = re.search(r'(\d+(?:\.\d+)?)\s*(?:rupees?|rs)', text_lower)
    if pm:
        price_per_kg = float(pm.group(1))
        tokens["price"] = price_per_kg

    # ── Confidence & missing fields ───────────────────────────────────────────
    confidence = round(confidence_hits / total_fields, 2)
    missing = []
    if not crop:          missing.append("crop")
    if not quantity_kg:   missing.append("quantity")
    if not quality:       missing.append("quality_grade")
    if not avail_date:    missing.append("available_date")
    if not market:        missing.append("target_market")

    return {
        "crop":                crop,
        "quantity_kg":         round(quantity_kg, 1) if quantity_kg else None,
        "quantity_display":    quantity_disp,
        "quality_grade":       quality,
        "available_date":      avail_date or today.isoformat(),
        "target_market":       market,
        "asking_price_per_kg": price_per_kg,
        "raw_text":            raw,
        "confidence":          confidence,
        "missing_fields":      missing,
        "parsed_tokens":       tokens,
    }


# ─── LLM Fallback Template (for when API key is available) ────────────────────

def build_llm_prompt(text: str) -> str:
    """
    Returns a prompt you can send to Claude/GPT to parse the utterance.
    Use this as fallback when regex confidence < 0.4.
    """
    return f"""You are an agricultural data parser for an Indian farmer platform.
Parse this farmer utterance into structured JSON.

Utterance: "{text}"

Return ONLY valid JSON with these exact keys:
{{
  "crop": "<crop name in English>",
  "quantity_kg": <number or null>,
  "quality_grade": "<A, B, or C>",
  "available_date": "<YYYY-MM-DD>",
  "target_market": "<city name or null>",
  "asking_price_per_kg": <number or null>
}}

Rules:
- 1 quintal = 100 kg
- Quality: A=fresh/premium, B=good, C=mixed/poor
- If date not mentioned, use today's date: {date.today().isoformat()}
- If a field is unclear, use null"""
