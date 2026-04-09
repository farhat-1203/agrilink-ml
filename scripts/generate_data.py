"""
Generate realistic synthetic datasets for AgriLink AI.
Run once before training: python scripts/generate_data.py
"""
import numpy as np
import pandas as pd
import json
import os

SEED = 42
np.random.seed(SEED)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── 1. Crop Price Dataset ────────────────────────────────────────────────────
CROPS      = ["tomato", "onion", "potato", "cabbage", "carrot", "wheat", "rice", "sugarcane"]
SEASONS    = ["kharif", "rabi", "zaid"]
MARKETS    = ["Mumbai", "Pune", "Nashik", "Nagpur", "Aurangabad", "Kolhapur"]
QUALITIES  = ["A", "B", "C"]

BASE_PRICE = {"tomato":18, "onion":22, "potato":16, "cabbage":14,
              "carrot":20, "wheat":25, "rice":30, "sugarcane":12}
SEASON_MULT = {"kharif":1.1, "rabi":0.9, "zaid":1.2}
MARKET_MULT = {"Mumbai":1.3, "Pune":1.2, "Nashik":1.0, "Nagpur":0.95, "Aurangabad":0.9, "Kolhapur":1.1}
QUALITY_MULT = {"A":1.2, "B":1.0, "C":0.8}

rows = []
for _ in range(3000):
    crop    = np.random.choice(CROPS)
    season  = np.random.choice(SEASONS)
    market  = np.random.choice(MARKETS)
    quality = np.random.choice(QUALITIES)
    qty_qtl = np.random.randint(1, 50)
    rainfall = np.random.uniform(30, 200)          # mm
    days_to_market = np.random.randint(0, 5)

    price = (BASE_PRICE[crop]
             * SEASON_MULT[season]
             * MARKET_MULT[market]
             * QUALITY_MULT[quality]
             * (1 + (rainfall - 100) * 0.002)      # rain effect
             * (1 - days_to_market * 0.03)          # freshness decay
             * np.random.uniform(0.85, 1.15))       # noise

    rows.append({
        "crop": crop, "season": season, "market": market,
        "quality_grade": quality, "quantity_quintals": qty_qtl,
        "rainfall_mm": round(rainfall, 1),
        "days_to_market": days_to_market,
        "price_per_kg": round(max(price, 5), 2)
    })

price_df = pd.DataFrame(rows)
price_df.to_csv(f"{BASE}/data/crop_prices.csv", index=False)
print(f"[✓] crop_prices.csv  — {len(price_df)} rows")

# ─── 2. Demand Time-Series Dataset ────────────────────────────────────────────
dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
demand_rows = []
for crop in ["tomato", "onion", "potato"]:
    base_demand = {"tomato": 500, "onion": 700, "potato": 450}[crop]
    for i, d in enumerate(dates):
        trend      = i * 0.05
        seasonality = np.sin(2 * np.pi * d.dayofyear / 365) * base_demand * 0.15
        weekly     = -50 if d.weekday() == 6 else 0   # Sunday dip
        festival   = 200 if d.month in [10, 11] else 0  # Diwali surge
        noise      = np.random.normal(0, base_demand * 0.08)
        demand_rows.append({
            "date": d.strftime("%Y-%m-%d"), "crop": crop,
            "demand_quintals": round(max(base_demand + trend + seasonality + weekly + festival + noise, 50), 1)
        })

demand_df = pd.DataFrame(demand_rows)
demand_df.to_csv(f"{BASE}/data/demand_series.csv", index=False)
print(f"[✓] demand_series.csv — {len(demand_df)} rows")

# ─── 3. Buyers Dataset (JSON) ─────────────────────────────────────────────────
buyers = []
buyer_names = [
    "Ramesh Agro Traders", "Mumbai Fresh Mart", "Pune Wholesale Hub",
    "Nashik Mandi Co.", "Green Valley Exports", "FarmFresh Pvt Ltd",
    "Nagpur Agri Corp", "Maharashtra Kisan Market", "Kolhapur Traders",
    "Sahyadri Produce Co."
]
for i, name in enumerate(buyer_names):
    buyers.append({
        "buyer_id": f"B{i+1:03d}",
        "name": name,
        "location": np.random.choice(MARKETS),
        "lat": round(np.random.uniform(17.5, 20.5), 4),
        "lon": round(np.random.uniform(73.0, 79.5), 4),
        "preferred_crops": list(np.random.choice(CROPS, size=3, replace=False)),
        "min_quantity_qtl": int(np.random.choice([5, 10, 20])),
        "max_quantity_qtl": int(np.random.choice([100, 200, 500])),
        "avg_price_offered": {c: round(BASE_PRICE[c] * np.random.uniform(0.9, 1.2), 2) for c in CROPS},
        "rating": round(np.random.uniform(3.5, 5.0), 1),
        "payment_days": int(np.random.choice([0, 7, 14, 30]))
    })

with open(f"{BASE}/data/buyers.json", "w") as f:
    json.dump(buyers, f, indent=2)
print(f"[✓] buyers.json       — {len(buyers)} buyers")

print("\nAll datasets generated successfully!")
