"""
tests/test_all.py  ─  Full test suite for Crop Quality Estimator

Run:
    cd crop_quality_estimator
    python tests/test_all.py

Coverage:
    • dataset.py      — synthetic generation + DataLoader shapes
    • model.py        — forward pass, loss, freeze/unfreeze
    • freshness.py    — signal ranges, disease penalty, input formats
    • decision.py     — all grade combinations, shelf life, urgency
    • inference.py    — all 4 input formats, heuristic fallback, latency
    • End-to-end      — full pipeline from bytes → QualityReport
"""

import io
import sys
import warnings
import time
import base64

warnings.filterwarnings("ignore")
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))

import numpy as np
import torch
from PIL import Image, ImageDraw

PASS = FAIL = 0


def ok(label: str):
    global PASS
    PASS += 1
    print(f"  [PASS] {label}")


def fail(label: str, detail: str = ""):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {label}  {detail}")


def check(label: str, condition: bool, detail: str = ""):
    if condition:
        ok(label)
    else:
        fail(label, detail)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_pil(color=(200, 50, 40), size=300, spots=False) -> Image.Image:
    img  = Image.new("RGB", (size, size), (240, 245, 235))
    draw = ImageDraw.Draw(img)
    draw.ellipse([30, 30, size-30, size-30], fill=color)
    if spots:
        draw.ellipse([80, 80, 110, 110], fill=(65, 42, 22))
        draw.ellipse([160, 145, 188, 172], fill=(65, 42, 22))
    return img


def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Dataset
# ══════════════════════════════════════════════════════════════════════════════
def test_dataset():
    print("\n[1] Dataset")
    from modules.dataset import generate_synthetic_dataset, get_dataloaders

    root = generate_synthetic_dataset("data/test_synth", images_per_class=10)
    check("Synthetic data generated",   __import__("os").path.isdir(root))

    train_dl, val_dl, classes = get_dataloaders(root, batch_size=8, num_workers=0)
    check("Classes discovered",         len(classes) == 13, f"got {len(classes)}")

    batch = next(iter(train_dl))
    check("Image tensor shape",         tuple(batch["image"].shape[1:]) == (3, 224, 224))
    check("Label tensor long",          batch["label"].dtype == torch.long)
    check("is_diseased float",          batch["is_diseased"].dtype == torch.float32)
    check("is_diseased binary values",  set(batch["is_diseased"].tolist()).issubset({0.0, 1.0}))
    check("Train size > val size",      len(train_dl.dataset) > len(val_dl.dataset))

    val_batch = next(iter(val_dl))
    check("Val batch image shape OK",   tuple(val_batch["image"].shape[1:]) == (3, 224, 224))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Model
# ══════════════════════════════════════════════════════════════════════════════
def test_model():
    print("\n[2] Model")
    from modules.model import build_model, CombinedLoss

    for backbone in ["mobilenet_v3_small", "efficientnet_b0"]:
        model = build_model(num_classes=13, backbone=backbone)
        model.eval()

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits, disease_prob = model(x)

        check(f"{backbone} logits shape",   tuple(logits.shape) == (2, 13))
        check(f"{backbone} disease_prob",   tuple(disease_prob.shape) == (2,))
        check(f"{backbone} prob in [0,1]",  bool((disease_prob >= 0).all() and (disease_prob <= 1).all()))
        check(f"{backbone} param count",    int(sum(p.numel() for p in model.parameters())) > 1_000_000)

    # Loss
    model  = build_model(num_classes=13)
    loss_fn = CombinedLoss()
    x      = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits, dp = model(x)
    labels = torch.tensor([0, 5, 12, 3])
    dis    = torch.tensor([0., 1., 1., 0.])
    loss, info = loss_fn(logits, dp, labels, dis)
    check("Loss is scalar",             loss.ndim == 0)
    check("Loss is positive",           float(loss) > 0)
    check("Loss info has ce/bce keys",  "ce" in info and "bce" in info)

    # Freeze / unfreeze
    frozen_params = sum(1 for p in model.features.parameters() if not p.requires_grad)
    model.unfreeze_backbone()
    unfrozen_params = sum(1 for p in model.features.parameters() if not p.requires_grad)
    check("Unfreeze reduces frozen count", unfrozen_params < frozen_params)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Freshness Estimator
# ══════════════════════════════════════════════════════════════════════════════
def test_freshness():
    print("\n[3] Freshness Estimator")
    from modules.freshness import FreshnessEstimator

    est = FreshnessEstimator()

    # Basic: RGB array input
    fresh_img  = np.array(make_pil((210, 40, 32), spots=False))
    rotten_img = np.array(make_pil((115, 78, 52), spots=True))

    r_fresh  = est.estimate(fresh_img,  disease_prob=0.05)
    r_rotten = est.estimate(rotten_img, disease_prob=0.88)

    check("Fresh score > rotten",       r_fresh.freshness_score > r_rotten.freshness_score,
          f"fresh={r_fresh.freshness_score} rotten={r_rotten.freshness_score}")
    check("Freshness score in [0,100]", 0 <= r_fresh.freshness_score <= 100)
    check("6 signals present",          len(r_fresh.signals) == 6)
    check("All signals in [0,1]",       all(0 <= v <= 1 for v in r_fresh.signals.values()))
    check("Image stats present",        len(r_fresh.image_stats) >= 4)

    # Disease penalty
    r_nodis = est.estimate(fresh_img, disease_prob=0.0)
    r_dis   = est.estimate(fresh_img, disease_prob=1.0)
    check("Disease penalty reduces score", r_dis.freshness_score < r_nodis.freshness_score)

    # Bytes input
    r_bytes = est.from_bytes(pil_to_bytes(make_pil()), disease_prob=0.0)
    check("from_bytes works",           0 <= r_bytes.freshness_score <= 100)

    # Base64 input
    b64 = base64.b64encode(pil_to_bytes(make_pil())).decode()
    r_b64 = est.from_b64(b64, disease_prob=0.0)
    check("from_b64 works",             0 <= r_b64.freshness_score <= 100)

    # Known cases
    green_img = np.array(make_pil((50, 170, 50)))
    brown_img = np.array(make_pil((140, 110, 60), spots=True))
    r_green   = est.estimate(green_img, 0.0)
    r_brown   = est.estimate(brown_img, 0.8)
    check("Green crop higher than brown+diseased",
          r_green.freshness_score > r_brown.freshness_score)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Decision Engine
# ══════════════════════════════════════════════════════════════════════════════
def test_decision():
    print("\n[4] Decision Engine")
    from modules.decision import make_report

    # All grade combinations
    scenarios = [
        # (freshness, disease_prob, expected_grade)
        (82.0, 0.05, "A"),
        (55.0, 0.40, "B"),
        (55.0, 0.80, "C"),
        (28.0, 0.10, "C"),
    ]
    for f, dp, expected in scenarios:
        r = make_report(f, dp, "Tomato___healthy", 0.9, [], {})
        check(f"Grade: freshness={f}, dis={dp} → {expected}",
              r.quality_grade == expected,
              f"got {r.quality_grade}")

    # Shelf life decreases with higher disease
    r_low  = make_report(75.0, 0.05, "Tomato___healthy",    0.95, [], {})
    r_high = make_report(75.0, 0.90, "Tomato___Late_blight", 0.90, [], {})
    check("Higher disease → shorter shelf life",
          r_low.shelf_life_days >= r_high.shelf_life_days)

    # Urgency
    r_a = make_report(82.0, 0.05, "Tomato___healthy",    0.95, [], {})
    r_c = make_report(20.0, 0.92, "Potato___Late_blight", 0.92, [], {})
    check("Grade A urgency is low",    r_a.urgency_level == "low")
    check("Grade C urgency is high",   r_c.urgency_level == "high")

    # Required fields
    r = make_report(60.0, 0.30, "Tomato___Early_blight", 0.75, [], {})
    for field in ["quality_grade","grade_label","shelf_life_days","urgency_level",
                  "freshness_score","is_diseased","disease_label","disease_confidence",
                  "market_tier","summary"]:
        check(f"Field '{field}' present", hasattr(r, field))

    # Label formatting
    r = make_report(60.0, 0.3, "Tomato___Early_blight", 0.7, [], {})
    check("Label formatted correctly",  "—" in r.disease_label,
          f"got: {r.disease_label!r}")

    # to_dict works
    d = r.to_dict()
    check("to_dict returns dict",       isinstance(d, dict))
    check("to_dict has all keys",       len(d) >= 10)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Inference Engine
# ══════════════════════════════════════════════════════════════════════════════
def test_inference():
    print("\n[5] Inference Engine")
    from modules.inference import InferenceEngine

    eng = InferenceEngine(checkpoint_path="models/best_model.pt", device="cpu")

    base_img = make_pil()
    img_bytes = pil_to_bytes(base_img)
    img_np    = np.array(base_img)
    img_b64   = base64.b64encode(img_bytes).decode()

    required_keys = [
        "quality_grade", "grade_label", "shelf_life_days", "urgency_level",
        "freshness_score", "is_diseased", "disease_label", "disease_confidence",
        "top3_predictions", "freshness_signals", "market_tier", "summary",
        "latency_ms", "mode", "image_stats",
    ]

    for fmt, inp in [("bytes", img_bytes), ("PIL", base_img),
                     ("numpy", img_np), ("base64", img_b64)]:
        r = eng.predict(inp)
        check(f"{fmt}: all required keys present",
              all(k in r for k in required_keys),
              str([k for k in required_keys if k not in r]))
        check(f"{fmt}: grade is A/B/C",      r["quality_grade"] in {"A","B","C"})
        check(f"{fmt}: freshness in [0,100]", 0 <= r["freshness_score"] <= 100)
        check(f"{fmt}: shelf_life >= 0",     r["shelf_life_days"] >= 0)
        check(f"{fmt}: urgency valid",       r["urgency_level"] in {"low","medium","high"})
        check(f"{fmt}: top3 is list",        isinstance(r["top3_predictions"], list))

    # Heuristic-only fallback
    eng2 = InferenceEngine(heuristic_only=True)
    r2   = eng2.predict(img_bytes)
    check("Heuristic mode returns grade",   r2["quality_grade"] in {"A","B","C"})
    check("Heuristic mode label",           r2["mode"] == "heuristic_only")

    # Latency check (warm calls after first)
    _ = eng.predict(img_bytes)  # warm-up
    t0 = time.perf_counter()
    for _ in range(5):
        eng.predict(img_bytes)
    avg_ms = (time.perf_counter() - t0) / 5 * 1000
    check(f"Avg latency < 300ms (got {avg_ms:.0f}ms)", avg_ms < 300)


# ══════════════════════════════════════════════════════════════════════════════
# 6. End-to-End Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def test_e2e():
    print("\n[6] End-to-End Pipeline")
    from modules.inference import InferenceEngine

    eng = InferenceEngine(checkpoint_path="models/best_model.pt", device="cpu")

    test_cases = [
        ("Fresh vibrant crop",  (210, 40, 32),  False, 0.05),
        ("Stale dull crop",     (145, 88, 65),  False, 0.45),
        ("Rotting with spots",  (115, 78, 52),  True,  0.88),
        ("Healthy green crop",  (50, 170, 50),  False, 0.04),
    ]

    prev_fresh = 200
    results = []
    for label, color, spots, _ in test_cases:
        r = eng.predict(pil_to_bytes(make_pil(color, spots=spots)))
        results.append(r)
        check(f"Pipeline OK: {label}",
              all(k in r for k in ["quality_grade","freshness_score","urgency_level"]))
        prev_fresh = r["freshness_score"]

    # Spot check: fresh should out-score rotting
    r_fresh  = eng.predict(pil_to_bytes(make_pil((210, 40, 32), spots=False)))
    r_rotten = eng.predict(pil_to_bytes(make_pil((115, 78, 52), spots=True)))
    check("Fresh > rotten freshness",
          r_fresh["freshness_score"] > r_rotten["freshness_score"],
          f"fresh={r_fresh['freshness_score']} rotten={r_rotten['freshness_score']}")

    # Summary is non-empty
    check("Summary text is non-empty", len(results[0]["summary"]) > 20)
    # Market tier is non-empty
    check("Market tier is non-empty",  len(results[0]["market_tier"]) > 5)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 62)
    print(" Crop Quality Estimator — Full Test Suite")
    print("=" * 62)

    test_dataset()
    test_model()
    test_freshness()
    test_decision()
    test_inference()
    test_e2e()

    print("\n" + "=" * 62)
    print(f"  RESULTS: {PASS} passed / {FAIL} failed / {PASS+FAIL} total")
    print("=" * 62)

    if FAIL:
        sys.exit(1)
