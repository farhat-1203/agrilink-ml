"""
config.py  ─  Central configuration for Crop Quality Estimator
All paths, hyperparameters, and thresholds live here.
Change values here; every other module imports from here.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
DATA_DIR    = ROOT_DIR / "data"
MODELS_DIR  = ROOT_DIR / "models"
SYNTH_DIR   = DATA_DIR / "plantvillage_synth"   # synthetic dataset (CI / demo)

# Model checkpoint written by trainer, read by inference
CHECKPOINT  = MODELS_DIR / "best_model.pt"
CLASS_NAMES = MODELS_DIR / "class_names.json"

# ── Image settings ────────────────────────────────────────────────────────────
IMG_SIZE    = 224          # MobileNetV3 / EfficientNet-B0 native resolution
IMG_MEAN    = [0.485, 0.456, 0.406]   # ImageNet statistics
IMG_STD     = [0.229, 0.224, 0.225]

# ── Model settings ────────────────────────────────────────────────────────────
# "mobilenet_v3_small"  → ~2.5 M params, ~55 ms / image on CPU
# "efficientnet_b0"     → ~5.3 M params, ~90 ms / image on CPU
BACKBONE    = "mobilenet_v3_small"

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE       = 32
NUM_WORKERS      = 2
VAL_FRACTION     = 0.15    # 15 % of each class held out for validation
EPOCHS           = 30
WARMUP_EPOCHS    = 5       # backbone frozen; only new heads trained
LR               = 1e-3
WEIGHT_DECAY     = 1e-4
DROPOUT          = 0.3
LABEL_SMOOTHING  = 0.1     # reduces overconfidence
AUX_LOSS_WEIGHT  = 0.2     # binary healthy/diseased head weight
EARLY_STOP_PATIENCE = 8    # stop if val_acc doesn't improve for N epochs

# ── Freshness signal weights (must sum to 1.0) ────────────────────────────────
# Each weight was chosen based on how strongly the signal correlates
# with actual freshness in PlantVillage-class images.
FRESHNESS_WEIGHTS = {
    "saturation" : 0.22,   # HSV S — vivid = fresh, dull = aged
    "greenness"  : 0.22,   # LAB b* axis — negative = green = chlorophyll
    "browning"   : 0.20,   # brown pixel fraction — enzymatic oxidation
    "sharpness"  : 0.18,   # Laplacian variance — turgid cells = crisp edges
    "brightness" : 0.10,   # HSV V — very dark or washed-out = bad
    "uniformity" : 0.08,   # hue CoV — spots break colour uniformity
}
assert abs(sum(FRESHNESS_WEIGHTS.values()) - 1.0) < 1e-6, \
    "Freshness weights must sum to 1.0"

# Disease probability penalty on freshness (scales 0-25 pts off a 100-pt score)
DISEASE_FRESHNESS_PENALTY = 0.25

# ── Grading thresholds ────────────────────────────────────────────────────────
# Grade A: clearly healthy, high freshness
# Grade B: mild disease OR moderate freshness
# Grade C: severe disease OR low freshness
GRADE_THRESHOLDS = {
    "A": {"min_freshness": 68.0, "max_disease_prob": 0.25},
    "B": {"min_freshness": 38.0, "max_disease_prob": 0.70},
    # C is the fallback when neither A nor B conditions are met
}

# Shelf-life lookup: (low_days, high_days) per grade × freshness bucket
# freshness bucket:  high ≥ 70,  medium ≥ 40,  low < 40
SHELF_LIFE_TABLE = {
    "A": {"high": (6, 7), "medium": (4, 5), "low": (4, 5)},
    "B": {"high": (3, 4), "medium": (2, 3), "low": (1, 2)},
    "C": {"high": (1, 1), "medium": (0, 1), "low": (0, 0)},
}

# Days to subtract based on detected disease severity
DISEASE_SHELF_PENALTIES = {
    "healthy"           : 0,
    "early_blight"      : 1,
    "late_blight"       : 2,
    "black_rot"         : 2,
    "leaf_mold"         : 1,
    "septoria_leaf_spot": 1,
    "bacterial_spot"    : 2,
    "apple_scab"        : 1,
    "default"           : 1,   # unknown disease
}
