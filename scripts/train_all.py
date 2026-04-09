"""
AgriLink AI — One-shot training script.
Run this once to generate data and train all trainable models.

Usage:
    cd agrilink-ml
    python scripts/train_all.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 50)
print(" AgriLink AI — Training Pipeline")
print("=" * 50)

print("\n[1/2] Generating datasets...")
exec(open("scripts/generate_data.py").read())

print("\n[2/2] Training price prediction model...")
from modules.price_predictor import train_and_save
train_and_save()

print("\n" + "=" * 50)
print(" All done! Start the server with:")
print("   uvicorn main:app --reload --port 8000")
print("=" * 50)
