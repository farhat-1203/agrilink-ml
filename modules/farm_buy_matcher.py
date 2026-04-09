import json, os, math
import numpy as np

WEIGHTS = {
    "distance": 0.25,
    "demand": 0.15,
    "price": 0.35,
    "quantity": 0.25
}

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def calculate_match_score(dist_km, price_offered, market_price, qty_available, buyer_max_qty, is_preferred):
    s_proximity = 100 * math.exp(-dist_km / 100)
    s_demand = 100 if is_preferred else 40
    s_price = min((price_offered / market_price) * 100, 115)

    qty_ratio = qty_available / buyer_max_qty
    s_qty = 100 if 0.7 <= qty_ratio <= 1.3 else (70 if qty_ratio > 1.3 else 40)

    total = (WEIGHTS["distance"] * s_proximity) + \
            (WEIGHTS["demand"] * s_demand) + \
            (WEIGHTS["price"] * s_price) + \
            (WEIGHTS["quantity"] * s_qty)

    return round(total, 2)

class AgriMatchingEngine:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go to root

        with open(os.path.join(BASE_DIR, 'data', 'buyers_real.json')) as f:
            self.buyers = json.load(f)

        with open(os.path.join(BASE_DIR, 'data', 'farmer_listings.json')) as f:
            self.listings = json.load(f)

    def get_best_buyers_for_farmer(self, crop, qty_kg, f_lat, f_lon, current_mkt_price):
        rankings = []
        for b in self.buyers:
            dist = _haversine_km(f_lat, f_lon, b["lat"], b["lon"])
            is_pref = crop.lower() in b["preferred_crops"]

            buyer_offer = current_mkt_price * b["base_margin_factor"]

            score = calculate_match_score(
                dist, buyer_offer, current_mkt_price,
                qty_kg, b["max_quantity_qtl"]*100, is_pref
            )

            rankings.append({
                "buyer_name": b["name"],
                "score": score,
                "offered_price": round(buyer_offer, 2),
                "distance_km": round(dist, 1)
            })

        return sorted(rankings, key=lambda x: x["score"], reverse=True)

    def get_best_farmers_for_buyer(self, buyer_id, current_mkt_prices_dict):
        buyer = next(b for b in self.buyers if b["buyer_id"] == buyer_id)

        rankings = []
        for l in self.listings:
            if l["crop"].lower() in buyer["preferred_crops"]:
                dist = _haversine_km(buyer["lat"], buyer["lon"], l["lat"], l["lon"])

                mkt_p = current_mkt_prices_dict.get(
                    l["crop"].lower(),
                    l["expected_price_per_kg"]
                )

                score = calculate_match_score(
                    dist, mkt_p, l["expected_price_per_kg"],
                    l["quantity_kg"], buyer["max_quantity_qtl"]*100, True
                )

                rankings.append({
                    "farmer_name": l["farmer_name"],
                    "crop": l["crop"],
                    "quantity": l["quantity_kg"],
                    "score": score,
                    "distance_km": round(dist, 1)
                })

        return sorted(rankings, key=lambda x: x["score"], reverse=True)


# ✅ ONLY RUN WHEN DIRECTLY EXECUTED
if __name__ == "__main__":
    engine = AgriMatchingEngine()

    print(engine.get_best_buyers_for_farmer("wheat", 800, 26.46, 79.51, 24.50)[:3])