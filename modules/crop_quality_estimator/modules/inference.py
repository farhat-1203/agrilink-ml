"""
modules/inference.py  ─  Production Inference Engine

Wires the three modules together into a single predict() call:
    1. Decode image (bytes / base64 / numpy / PIL)
    2. Run disease classifier (NN forward pass)   ← ~55 ms on CPU
    3. Run freshness estimator (CV heuristics)     ← ~5  ms on CPU
    4. Run decision engine (pure Python)           ← <1  ms
    5. Return structured QualityReport dict

Key design decisions:
    • Model loaded once at startup, cached as a singleton
    • threading.Lock on the model forward pass for safe concurrent requests
    • Falls back to heuristic-only mode if no checkpoint exists
      (so the API can serve requests before training is done)
    • Accepts 4 input formats: bytes, base64 str, numpy array, PIL Image
"""

from __future__ import annotations

import io
import json
import base64
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHECKPOINT, CLASS_NAMES as CLASS_NAMES_PATH, IMG_MEAN, IMG_STD, IMG_SIZE
from .freshness import FreshnessEstimator
from .decision  import DecisionEngine, QualityReport


# Inference transform — must match val_transforms() exactly
_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])


class InferenceEngine:
    """
    Thread-safe inference engine.
    Instantiate once at application startup; re-use for every request.
    """

    def __init__(
        self,
        checkpoint_path : Optional[str] = None,
        device          : str = "cpu",
        heuristic_only  : bool = False,
    ):
        self._lock        = threading.Lock()
        self.device       = torch.device(device)
        self.model        = None
        self.class_names  : List[str] = []
        self.heuristic_only = heuristic_only

        if not heuristic_only:
            path = checkpoint_path or str(CHECKPOINT)
            if Path(path).exists():
                self._load(path)
            else:
                print(f"[inference] Checkpoint not found at {path} — heuristic-only mode")
                self.heuristic_only = True

        self._freshness = FreshnessEstimator()
        self._decision  = DecisionEngine()

        mode = "heuristic-only" if self.heuristic_only else \
               f"NN ({len(self.class_names)} classes) + heuristic"
        print(f"[inference] Mode   : {mode}")
        print(f"[inference] Device : {self.device}")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        image: Union[bytes, str, np.ndarray, Image.Image],
    ) -> dict:
        """
        Full pipeline: image → QualityReport dict.

        Args:
            image: one of —
                bytes       raw JPEG / PNG / WebP bytes
                str         base64-encoded image (data-URI prefix optional)
                np.ndarray  H×W×3 uint8 RGB array
                PIL.Image   PIL Image object

        Returns:
            dict with keys: quality_grade, grade_label, shelf_life_days,
                            urgency_level, freshness_score, is_diseased,
                            disease_label, disease_confidence, top3_predictions,
                            freshness_signals, market_tier, summary,
                            latency_ms, mode
        """
        t0 = time.perf_counter()

        pil_img, rgb_arr = self._decode(image)

        # ── Step 1: Disease classification (NN) ───────────────────────────────
        if self.model is not None and not self.heuristic_only:
            disease_prob, top_label, top_conf, top3 = self._classify(pil_img)
        else:
            disease_prob, top_label, top_conf, top3 = 0.0, "unknown", 0.0, []

        # ── Step 2: Freshness estimation (CV heuristics) ──────────────────────
        fresh = self._freshness.estimate(rgb_arr, disease_prob)

        # ── Step 3: Decision engine ────────────────────────────────────────────
        report = self._decision.decide(
            freshness_score    = fresh.freshness_score,
            disease_prob       = disease_prob,
            disease_label      = top_label,
            disease_confidence = top_conf,
            top3               = top3,
            freshness_signals  = fresh.signals,
        )

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        result = report.to_dict()
        result["latency_ms"]   = latency_ms
        result["mode"]         = "heuristic_only" if self.heuristic_only else "nn+heuristic"
        result["image_stats"]  = fresh.image_stats
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load(self, path: str):
        from modules.model import CropDiseaseModel
        ckpt = torch.load(path, map_location=self.device)
        self.class_names = ckpt["class_names"]
        model = CropDiseaseModel(
            num_classes=ckpt["num_classes"],
            backbone   =ckpt.get("backbone", "mobilenet_v3_small"),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        model.to(self.device)
        self.model = model
        print(f"[inference] Loaded  : {path}  "
              f"({ckpt['num_classes']} classes, val_acc={ckpt.get('val_acc',0)*100:.1f}%)")

    @staticmethod
    def _decode(image):
        """Normalise any input format → (PIL RGB image, H×W×3 uint8 numpy array)."""
        if isinstance(image, np.ndarray):
            pil = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil = image.convert("RGB")
        elif isinstance(image, str):
            if "," in image:
                image = image.split(",", 1)[1]
            pil = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
        elif isinstance(image, bytes):
            pil = Image.open(io.BytesIO(image)).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        pil.thumbnail((1024, 1024), Image.LANCZOS)   # cap resolution for speed
        return pil, np.array(pil, dtype=np.uint8)

    @torch.no_grad()
    def _classify(self, pil_img: Image.Image):
        """Run NN and return (disease_prob, top_label, top_conf, top3_list)."""
        tensor = _TRANSFORM(pil_img).unsqueeze(0).to(self.device)

        with self._lock:
            logits, disease_prob_t = self.model(tensor)

        probs        = torch.softmax(logits, dim=1)[0]
        disease_prob = float(disease_prob_t[0])

        k = min(3, len(self.class_names))
        top_probs, top_idxs = probs.topk(k)

        top3 = [
            {
                "label"      : self.class_names[int(i)],
                "confidence" : round(float(p) * 100, 1),
            }
            for p, i in zip(top_probs.cpu(), top_idxs.cpu())
        ]

        return disease_prob, self.class_names[int(top_idxs[0])], float(top_probs[0]), top3


# ── Module-level lazy singleton ────────────────────────────────────────────────
_instance : Optional[InferenceEngine] = None
_init_lock = threading.Lock()


def get_engine(
    checkpoint_path: Optional[str] = None,
    device         : str = "cpu",
) -> InferenceEngine:
    """
    Returns the global InferenceEngine, creating it on first call.
    Safe to call from multiple threads simultaneously.
    """
    global _instance
    if _instance is None:
        with _init_lock:
            if _instance is None:
                _instance = InferenceEngine(
                    checkpoint_path=checkpoint_path,
                    device=device,
                )
    return _instance
