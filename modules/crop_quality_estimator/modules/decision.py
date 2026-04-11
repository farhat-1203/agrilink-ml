"""
modules/decision.py  ─  Quality Decision Engine

Takes outputs from the disease classifier and freshness estimator and
combines them into a final, structured quality report.

Decision matrix:
  ┌──────────────────────────────┬───────┬─────────┬──────────┐
  │ Condition                    │ Grade │ Shelf   │ Urgency  │
  ├──────────────────────────────┼───────┼─────────┼──────────┤
  │ Freshness ≥ 68 + dis ≤ 0.25 │ A     │ 5–7 d   │ low      │
  │ Freshness ≥ 38 + dis ≤ 0.70 │ B     │ 2–4 d   │ medium   │
  │ Otherwise                    │ C     │ 0–1 d   │ high     │
  └──────────────────────────────┴───────┴─────────┴──────────┘

Shelf life is further reduced by a disease-specific penalty
(e.g. Late Blight → −2 days) scaled by disease_prob.

All thresholds live in config.py so they can be tuned without touching logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    GRADE_THRESHOLDS,
    SHELF_LIFE_TABLE,
    DISEASE_SHELF_PENALTIES,
)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    # ── Core outputs (used by frontend) ──────────────────────────────────────
    quality_grade      : str          # "A" | "B" | "C"
    grade_label        : str          # "Premium" | "Standard" | "Below Standard"
    shelf_life_days    : int
    urgency_level      : str          # "low" | "medium" | "high"
    freshness_score    : float        # 0–100
    # ── Disease info ─────────────────────────────────────────────────────────
    is_diseased        : bool
    disease_label      : str          # e.g. "Tomato — Early Blight"
    disease_confidence : float        # 0–100 (%)
    top3_predictions   : List[Dict]   # [{label, confidence}, ...]
    # ── Signals ──────────────────────────────────────────────────────────────
    freshness_signals  : Dict[str, float]
    # ── Recommendation ───────────────────────────────────────────────────────
    market_tier        : str
    summary            : str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_api_response(self) -> dict:
        """Returns the full dict — identical to to_dict() for clarity."""
        return self.to_dict()


# ── Decision engine ───────────────────────────────────────────────────────────

class DecisionEngine:
    """Combines disease + freshness signals into a QualityReport."""

    def decide(
        self,
        freshness_score    : float,
        disease_prob       : float,
        disease_label      : str,
        disease_confidence : float,        # 0–1 raw (will be converted to %)
        top3               : List[Dict],
        freshness_signals  : Dict[str, float],
    ) -> QualityReport:

        # ── 1. Quality grade ─────────────────────────────────────────────────
        grade = self._grade(freshness_score, disease_prob)

        # ── 2. Freshness bucket (for shelf-life table) ────────────────────────
        bucket = "high" if freshness_score >= 70 else \
                 "medium" if freshness_score >= 40 else "low"

        # ── 3. Shelf life ─────────────────────────────────────────────────────
        shelf = self._shelf_life(grade, bucket, disease_label, disease_prob)

        # ── 4. Urgency ────────────────────────────────────────────────────────
        urgency = self._urgency(grade, shelf, disease_prob)

        # ── 5. Market tier ────────────────────────────────────────────────────
        market = self._market_tier(grade, shelf, disease_prob >= 0.5)

        # ── 6. Summary ────────────────────────────────────────────────────────
        clean_label = self._fmt_label(disease_label)
        summary     = self._summary(
            grade, shelf, urgency, freshness_score,
            disease_prob >= 0.5, clean_label, market
        )

        return QualityReport(
            quality_grade      = grade,
            grade_label        = {"A":"Premium","B":"Standard","C":"Below Standard"}[grade],
            shelf_life_days    = shelf,
            urgency_level      = urgency,
            freshness_score    = freshness_score,
            is_diseased        = disease_prob >= 0.5,
            disease_label      = clean_label,
            disease_confidence = round(disease_confidence * 100, 1),
            top3_predictions   = top3,
            freshness_signals  = freshness_signals,
            market_tier        = market,
            summary            = summary,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _grade(freshness: float, disease_prob: float) -> str:
        a = GRADE_THRESHOLDS["A"]
        b = GRADE_THRESHOLDS["B"]
        if freshness >= a["min_freshness"] and disease_prob <= a["max_disease_prob"]:
            return "A"
        if freshness >= b["min_freshness"] and disease_prob <= b["max_disease_prob"]:
            return "B"
        return "C"

    @staticmethod
    def _shelf_life(grade, bucket, disease_label, disease_prob) -> int:
        lo, hi = SHELF_LIFE_TABLE[grade][bucket]
        base   = (lo + hi) // 2

        # Look up disease-specific penalty
        label_lower = disease_label.lower()
        penalty_days = DISEASE_SHELF_PENALTIES["default"]
        for key, pen in DISEASE_SHELF_PENALTIES.items():
            if key in label_lower:
                penalty_days = pen
                break

        # Scale penalty by actual disease probability
        adj = math.ceil(penalty_days * disease_prob)
        return max(0, base - adj)

    @staticmethod
    def _urgency(grade, shelf, disease_prob) -> str:
        if grade == "C" or shelf <= 1 or disease_prob >= 0.80:
            return "high"
        if grade == "A" and shelf >= 5:
            return "low"
        return "medium"

    @staticmethod
    def _market_tier(grade, shelf, is_diseased) -> str:
        if grade == "A" and not is_diseased:
            return "Premium retail / export"
        if grade == "A" or (grade == "B" and shelf >= 3):
            return "Local wholesale mandi"
        if grade == "B":
            return "Processing / value-added (juice, pickle)"
        return "Immediate processing or compost"

    @staticmethod
    def _fmt_label(label: str) -> str:
        """'Tomato___Early_blight' → 'Tomato — Early Blight'"""
        if "___" in label:
            crop, cond = label.split("___", 1)
            return f"{crop.replace('_',' ')} — {cond.replace('_',' ').title()}"
        return label.replace("_", " ").title()

    @staticmethod
    def _summary(grade, shelf, urgency, freshness, is_diseased, label, market) -> str:
        status = "diseased" if is_diseased else "healthy"
        parts  = [f"Grade {grade} ({status}). Freshness: {freshness:.0f}/100."]
        if is_diseased:
            parts.append(f"Detected: {label}.")
        if shelf == 0:
            parts.append("⚠ Shelf life expired — do not sell.")
        else:
            parts.append(f"Shelf life: ~{shelf} day{'s' if shelf>1 else ''}.")
        parts.append(f"Recommended: {market}.")
        parts.append(
            "🔴 Sell immediately." if urgency == "high" else
            "🟡 Sell within 2 days." if urgency == "medium" else
            "🟢 Normal market cycle."
        )
        return " ".join(parts)


# Module-level singleton
_engine = DecisionEngine()


def make_report(
    freshness_score    : float,
    disease_prob       : float,
    disease_label      : str,
    disease_confidence : float,
    top3               : List[Dict],
    freshness_signals  : Dict[str, float],
) -> QualityReport:
    return _engine.decide(
        freshness_score    = freshness_score,
        disease_prob       = disease_prob,
        disease_label      = disease_label,
        disease_confidence = disease_confidence,
        top3               = top3,
        freshness_signals  = freshness_signals,
    )
