import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMAND_DATA_PATH = os.path.join(BASE_DIR, "data", "demand_series_real_v2.csv")

def load_real_demand_series(crop_name: str) -> pd.Series:
    """
    Loads and cleans the real-world demand series for a specific crop.
    """
    if not os.path.exists(DEMAND_DATA_PATH):
        raise FileNotFoundError(f"Missing {DEMAND_DATA_PATH}. Please ensure the file is generated.")
    
    df = pd.read_csv(DEMAND_DATA_PATH, parse_dates=['date'])
    
    # Filter for the crop from your Agmarknet list
    crop_df = df[df['crop'].str.lower() == crop_name.lower()]
    
    if crop_df.empty:
        raise ValueError(f"Crop '{crop_name}' not found in the Agmarknet-mapped dataset.")

    # Sort by date and set as index for time-series analysis
    crop_df = crop_df.sort_values('date').set_index('date')
    
    # Ensure there are no gaps in the timeline (daily frequency)
    series = crop_df['demand_quintals'].resample('D').mean().fillna(0)
    return series

def forecast_demand(crop: str, steps: int = 7):
    """
    Fits a Holt-Winters model to real-world seasonal patterns.
    """
    crop_name = crop
    forecast_days = steps
    try:
        series = load_real_demand_series(crop_name)
        
        # We use Additive Trend and Seasonality (weekly periods = 7)
        # This captures the 'Monday rush' and 'Sunday dip' seen in real Mandis
        model = ExponentialSmoothing(
            series,
            trend='add',
            seasonal='add',
            seasonal_periods=7,
            initialization_method="estimated"
        ).fit()
        
        # Generate the next 7 days of demand
        forecast = model.forecast(forecast_days)
        forecast = np.maximum(forecast, 0) # Demand cannot be negative
        
        # Formatting for the frontend/API
        start_date = series.index.max() + timedelta(days=1)
        predictions = []
        for i, val in enumerate(forecast):
            current_date = start_date + timedelta(days=i)
            predictions.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "day": current_date.strftime("%A"),
                "demand_estimate": round(float(val), 2),
                "unit": "Quintals"
            })
            
        # Determine Trend Direction
        recent_avg = series.iloc[-7:].mean()
        forecast_avg = forecast.mean()
        trend = "Rising" if forecast_avg > recent_avg * 1.05 else ("Falling" if forecast_avg < recent_avg * 0.95 else "Stable")

        return {
            "crop": crop_name,
            "trend": trend,
            "forecast": predictions,
            "metadata": {
                "algorithm": "Holt-Winters Seasonal Smoothing",
                "last_historical_date": series.index.max().strftime("%Y-%m-%d")
            }
        }

    except Exception as e:
        return {"error": str(e)}

# --- Testing the Model ---
if __name__ == "__main__":
    # Test for 'Wheat' which is in your Agmarknet file
    result = forecast_demand("Wheat")
    if "error" not in result:
        print(f"Demand Trend for {result['crop']}: {result['trend']}")
        for day in result['forecast']:
            print(f"{day['day']}: {day['demand_estimate']} {day['unit']}")
    else:
        print(f"Error: {result['error']}")