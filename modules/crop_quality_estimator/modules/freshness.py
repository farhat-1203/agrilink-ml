"""
modules/freshness.py  ─  Image-Based Freshness Estimator

Why not a trained ML model for freshness?
  PlantVillage has ONLY disease labels — no freshness / ripeness labels.
  Building a supervised freshness model would require a custom annotated
  dataset that doesn't publicly exist.

  Instead we use physics-based computer vision signals that genuinely
  correlate with freshness across many crop types:

  Signal          │ What it measures              │ Why it works
  ────────────────┼───────────────────────────────┼──────────────────────────────
  Saturation      │ HSV S channel mean            │ Fresh produce has vivid colour;
                  │                               │ ageing causes colour fade
  Greenness       │ LAB b* axis (negative = green)│ Chlorophyll breaks down as
                  │                               │ produce ages → less green
  Browning        │ Fraction of brown pixels      │ Enzymatic oxidation causes
                  │                               │ browning (Maillard reaction)
  Sharpness       │ Laplacian variance            │ Turgid cells → crisp edges;
                  │                               │ water loss → limp → blurry
  Brightness      │ HSV V channel mean            │ Very dark = rotting;
                  │                               │ washed-out = dried / wilted
  Uniformity      │ Hue CoV (saturated pixels)    │ Disease lesions & ageing
                  │                               │ create hue variation

Each signal is independently normalised to [0, 1] and then combined
via a weighted average (weights in config.py).

The disease probability from the NN is applied as a final penalty:
    final_score = raw_score − disease_weight × disease_prob × 100
"""

from __future__ import annotations

import io
import base64
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FRESHNESS_WEIGHTS, DISEASE_FRESHNESS_PENALTY


@dataclass
class FreshnessResult:
    freshness_score: float                      # 0–100 composite score
    signals        : Dict[str, float] = field(default_factory=dict)  # per-signal [0,1]
    image_stats    : Dict[str, float] = field(default_factory=dict)  # raw measurements

    def to_dict(self) -> dict:
        return asdict(self)


class FreshnessEstimator:
    """
    Stateless estimator — call .estimate(rgb_array, disease_prob) per image.
    Thread-safe; holds no mutable state.
    """

    def estimate(
        self,
        rgb: np.ndarray,        # H×W×3 uint8 RGB
        disease_prob: float = 0.0,
    ) -> FreshnessResult:
        """Compute freshness score from raw pixel array + NN disease probability."""
        bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        b_star  = lab[:,:,2] - 128.0   # centre on 0; negative = green, positive = yellow

        # ── Signal 1: Saturation ──────────────────────────────────────────────
        # Mean saturation 0–255; normalise with 200 as "fully fresh" reference
        sat_mean   = float(S.mean())
        saturation = float(np.clip(sat_mean / 200.0, 0, 1))

        # ── Signal 2: Greenness (LAB b* axis) ────────────────────────────────
        # b_star < 0 = blue-green (fresh chlorophyll)
        # b_star > 0 = yellow-orange (carotenoid, senescence)
        # Remap: -50 → 1.0 (very green), +50 → 0.0 (very yellow)
        b_mean    = float(b_star.mean())
        greenness = float(np.clip((-b_mean + 50.0) / 100.0, 0, 1))

        # ── Signal 3: Browning penalty ────────────────────────────────────────
        # Brown pixels: hue 8–28°, moderate saturation, moderate-low value
        brown_mask  = (H >= 8) & (H <= 28) & (S >= 40) & (S < 150) & (V >= 40) & (V < 160)
        brown_frac  = float(brown_mask.sum() / brown_mask.size)
        # Each 1% of brown area reduces score; >20% brown → near zero
        browning    = float(np.clip(1.0 - brown_frac * 5.0, 0, 1))

        # ── Signal 4: Texture sharpness (Laplacian variance) ─────────────────
        # High variance = crisp cell walls = turgid = fresh
        # Low variance  = blurry = limp / wilted
        lap_var   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = float(np.clip(lap_var / 500.0, 0, 1))

        # ── Signal 5: Brightness ─────────────────────────────────────────────
        # Optimal brightness ≈ 160/255 (mid-range).
        # Very dark (rotting) or very bright (dried) both penalised.
        v_mean     = float(V.mean())
        brightness = float(np.clip(1.0 - abs(v_mean - 160.0) / 160.0, 0, 1))

        # ── Signal 6: Hue uniformity ──────────────────────────────────────────
        # Only on pixels with meaningful saturation (S > 30)
        h_valid = H[S > 30]
        if len(h_valid) > 200:
            h_cov       = float(h_valid.std() / (h_valid.mean() + 1e-6))
            uniformity  = float(np.clip(1.0 - h_cov * 0.5, 0, 1))
        else:
            uniformity  = 0.5   # not enough colour information

        signals = {
            "saturation" : round(saturation, 3),
            "greenness"  : round(greenness,  3),
            "browning"   : round(browning,   3),
            "sharpness"  : round(sharpness,  3),
            "brightness" : round(brightness, 3),
            "uniformity" : round(uniformity, 3),
        }

        # ── Weighted combination → raw score ──────────────────────────────────
        raw = sum(signals[k] * FRESHNESS_WEIGHTS[k] for k in FRESHNESS_WEIGHTS) * 100.0

        # ── Disease penalty ───────────────────────────────────────────────────
        # disease_prob=1.0 → subtract up to 25 pts (DISEASE_FRESHNESS_PENALTY)
        penalty     = DISEASE_FRESHNESS_PENALTY * disease_prob * 100.0
        final_score = float(np.clip(raw - penalty, 0.0, 100.0))

        image_stats = {
            "mean_saturation" : round(sat_mean, 1),
            "mean_brightness" : round(v_mean,   1),
            "mean_b_star"     : round(b_mean,   2),
            "brown_fraction"  : round(brown_frac, 4),
            "laplacian_var"   : round(lap_var,  2),
            "disease_penalty" : round(penalty,  2),
        }

        return FreshnessResult(
            freshness_score = round(final_score, 1),
            signals         = signals,
            image_stats     = image_stats,
        )

    # ── Convenience wrappers ──────────────────────────────────────────────────

    def from_bytes(self, image_bytes: bytes, disease_prob: float = 0.0) -> FreshnessResult:
        """Accept raw image bytes (JPEG / PNG / WebP)."""
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil.thumbnail((1024, 1024), Image.LANCZOS)
        return self.estimate(np.array(pil, dtype=np.uint8), disease_prob)

    def from_b64(self, b64_str: str, disease_prob: float = 0.0) -> FreshnessResult:
        """Accept base64 string (with or without data-URI prefix)."""
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        return self.from_bytes(base64.b64decode(b64_str), disease_prob)


# Module-level singleton — instantiate once per process
_estimator = FreshnessEstimator()


def estimate_freshness(rgb: np.ndarray, disease_prob: float = 0.0) -> FreshnessResult:
    """Convenience function using the module-level singleton."""
    return _estimator.estimate(rgb, disease_prob)
