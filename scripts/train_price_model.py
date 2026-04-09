import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.price_predictor import train_and_save

print("Training price prediction model...")
train_and_save()
print("Done!")
