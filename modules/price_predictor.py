import os, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Configuration
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cleaned_agri_prices.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models", "agri_real_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models", "agri_real_encoders.pkl")

# We use geographic and temporal features for real-world accuracy
CATEGORICAL_COLS = ["State", "District Name", "Market Name", "Commodity", "Variety", "Season"]
FEATURE_COLS = ["State", "District Name", "Market Name", "Commodity", "Variety", "Season", "Month"]
TARGET_COL = "price_per_kg"

def get_season(month):
    if month in [6, 7, 8, 9, 10]: return "Kharif"
    elif month in [11, 12, 1, 2, 3]: return "Rabi"
    else: return "Zaid"

def train_on_real_data():
    print("Step 1: Loading and Cleaning Dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing
    # Create Date column from Year + Month
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))

    # Month already exists → no need to extract again
    df['Season'] = df['Month'].apply(get_season)

    # ⚠️ IMPORTANT: your dataset already has price_per_kg
    # So REMOVE this line completely:
    # df['price_per_kg'] = df['Modal Price (Rs./Quintal)'] / 100
    
    # Step 2: Encoding Categorical Data
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Step 3: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Model Training
    print("Step 2: Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=150, 
        max_depth=15, 
        min_samples_leaf=4, 
        n_jobs=-1, 
        random_state=42
    )
    model.fit(X_train, y_train)

    # Step 5: Evaluation
    preds = model.predict(X_test)
    print(f"  Accuracy Results:")
    print(f"  MAE: ₹{mean_absolute_error(y_test, preds):.2f}/kg")
    print(f"  R² Score: {r2_score(y_test, preds):.4f}")

    # Save
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    print(f"Model and Encoders saved successfully.")

# ─── New Prediction Function ──────────────────────────────────────────────────

def predict_price_realtime(state, district, market, commodity, variety, month):
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    
    season = get_season(month)
    
    def safe_encode(enc, val):
        return int(enc.transform([val])[0]) if val in enc.classes_ else 0

    input_data = np.array([[
        safe_encode(encoders["State"], state),
        safe_encode(encoders["District Name"], district),
        safe_encode(encoders["Market Name"], market),
        safe_encode(encoders["Commodity"], commodity),
        safe_encode(encoders["Variety"], variety),
        safe_encode(encoders["Season"], season),
        month
    ]])

    # Calculate Confidence Intervals using tree variance
    tree_preds = np.array([tree.predict(input_data)[0] for tree in model.estimators_])
    
    return {
        "predicted_price": round(np.mean(tree_preds), 2),
        "range": {
            "min": round(np.percentile(tree_preds, 5), 2),
            "max": round(np.percentile(tree_preds, 95), 2)
        },
        "unit": "INR/kg"
    }

# ─── Simplified Prediction Function for Demo ─────────────────────────────────

def predict_price(crop, season, market, quality_grade, quantity_quintals, rainfall_mm, days_to_market):
    """
    Simplified prediction function for the Streamlit demo.
    Uses a mock model since the real model requires specific state/district data.
    """
    # Base prices per crop (₹/kg)
    base_prices = {
        "tomato": 18.5, "onion": 22.0, "potato": 15.0, "cabbage": 12.0,
        "carrot": 25.0, "wheat": 28.0, "rice": 32.0, "sugarcane": 3.5
    }
    
    # Quality multipliers
    quality_mult = {"A": 1.15, "B": 1.0, "C": 0.85}
    
    # Season multipliers
    season_mult = {"kharif": 1.0, "rabi": 1.05, "zaid": 0.95}
    
    # Market multipliers (premium markets)
    market_mult = {
        "Mumbai": 1.12, "Pune": 1.08, "Nashik": 1.0,
        "Nagpur": 0.98, "Aurangabad": 0.96, "Kolhapur": 0.94
    }
    
    base = base_prices.get(crop.lower(), 20.0)
    price = base * quality_mult.get(quality_grade, 1.0) * season_mult.get(season.lower(), 1.0) * market_mult.get(market, 1.0)
    
    # Add some variance based on rainfall and days
    rainfall_factor = 1.0 + (rainfall_mm - 80) * 0.001
    days_factor = 1.0 - (days_to_market * 0.01)
    
    price = price * rainfall_factor * days_factor
    
    # Add random variance for realism
    variance = np.random.uniform(-0.08, 0.08)
    price = price * (1 + variance)
    
    # Calculate confidence and range
    confidence = 0.82 + np.random.uniform(-0.05, 0.05)
    price_range_pct = 0.12
    
    return {
        "predicted_price_per_kg": round(price, 2),
        "price_range": {
            "low": round(price * (1 - price_range_pct), 2),
            "high": round(price * (1 + price_range_pct), 2)
        },
        "confidence": round(confidence, 3),
        "factors": {
            "base_price": round(base, 2),
            "quality_adjustment": quality_grade,
            "season_adjustment": season,
            "market_adjustment": market,
            "rainfall_mm": rainfall_mm,
            "days_to_market": days_to_market
        }
    }

if __name__ == "__main__":
    train_on_real_data()