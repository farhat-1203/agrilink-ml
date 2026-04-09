"""
Buyer matching wrapper for main.py API compatibility.
"""
from modules.farm_buy_matcher import AgriMatchingEngine

# Initialize the matching engine
_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        _engine = AgriMatchingEngine()
    return _engine


def match_buyers(crop: str, quantity_kg: float, farmer_lat: float, 
                 farmer_lon: float, market_price_per_kg: float, top_n: int = 5):
    """
    Find and rank best buyers for a farmer's crop listing.
    
    Args:
        crop: Crop name (e.g., "tomato", "wheat")
        quantity_kg: Quantity available in kg
        farmer_lat: Farmer's latitude
        farmer_lon: Farmer's longitude
        market_price_per_kg: Current market price per kg
        top_n: Number of top matches to return
    
    Returns:
        dict with matches and best_match
    """
    engine = _get_engine()
    
    all_matches = engine.get_best_buyers_for_farmer(
        crop=crop,
        qty_kg=quantity_kg,
        f_lat=farmer_lat,
        f_lon=farmer_lon,
        current_mkt_price=market_price_per_kg
    )
    
    top_matches = all_matches[:top_n]
    
    return {
        "matches": [
            {
                "name": m["buyer_name"],
                "score": m["score"],
                "offered_price_per_kg": m["offered_price"],
                "distance_km": m["distance_km"]
            }
            for m in top_matches
        ],
        "best_match": {
            "name": top_matches[0]["buyer_name"],
            "score": top_matches[0]["score"],
            "offered_price_per_kg": top_matches[0]["offered_price"],
            "distance_km": top_matches[0]["distance_km"]
        } if top_matches else None
    }
