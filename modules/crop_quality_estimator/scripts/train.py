"""
scripts/train.py  ─  CLI Training Script

Usage — real PlantVillage data:
    python scripts/train.py --data_dir /path/to/PlantVillage --epochs 30

Usage — synthetic data (for CI / demo, no dataset needed):
    python scripts/train.py --synthetic --epochs 8

Usage — EfficientNet:
    python scripts/train.py --data_dir /path/to/PlantVillage \
        --backbone efficientnet_b0 --epochs 40 --lr 5e-4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser(description="Train Crop Quality Estimator")
    p.add_argument("--data_dir",   default=None,
                   help="PlantVillage root directory (class sub-folders inside)")
    p.add_argument("--backbone",   default="mobilenet_v3_small",
                   choices=["mobilenet_v3_small", "efficientnet_b0"])
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--synthetic",  action="store_true",
                   help="Generate and use synthetic data (ignores --data_dir)")
    p.add_argument("--synthetic_ipc", type=int, default=100,
                   help="Synthetic images per class (default 100)")
    args = p.parse_args()

    from modules.dataset import generate_synthetic_dataset
    from modules.trainer import train

    if args.synthetic or args.data_dir is None:
        print("[train] Generating synthetic PlantVillage dataset...")
        data_dir   = generate_synthetic_dataset(
            out_dir="data/plantvillage_synth",
            images_per_class=args.synthetic_ipc,
        )
        args.epochs     = min(args.epochs, 12)
        args.batch_size = min(args.batch_size, 16)
    else:
        data_dir = args.data_dir
        if not Path(data_dir).exists():
            sys.exit(f"[train] ERROR: --data_dir {data_dir!r} does not exist.")

    print(f"\n[train] Data     : {data_dir}")
    print(f"[train] Backbone : {args.backbone}")
    print(f"[train] Epochs   : {args.epochs}")
    print(f"[train] Batch    : {args.batch_size}")
    print(f"[train] LR       : {args.lr}\n")

    history = train(
        data_dir   = data_dir,
        backbone   = args.backbone,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
    )

    best = max(history, key=lambda r: r["val_acc"])
    print("\n" + "="*55)
    print(f"  Best val accuracy : {best['val_acc']*100:.2f}%  (epoch {best['epoch']})")
    print(f"  Checkpoint saved  : models/best_model.pt")
    print("="*55)
    print("\nStart API server:")
    print("  uvicorn api.app:app --reload --port 8000")


if __name__ == "__main__":
    main()
