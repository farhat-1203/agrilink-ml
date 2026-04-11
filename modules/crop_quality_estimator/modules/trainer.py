"""
modules/trainer.py  ─  Full Training Pipeline

Two-phase training strategy:
    Phase 1 — Warm-up (epochs 0 .. WARMUP_EPOCHS-1)
        Backbone is frozen; only the new heads are trained.
        Safe to use a higher LR since the backbone weights are protected.

    Phase 2 — Fine-tuning (remaining epochs)
        Entire network unfrozen and trained end-to-end.
        LR is reduced by 10× to avoid catastrophic forgetting.
        CosineAnnealingLR decays LR smoothly to near-zero by end of training.

Other features:
    • Gradient clipping (max_norm=1.0) — prevents exploding gradients
    • Early stopping on val_acc with configurable patience
    • Best model saved to CHECKPOINT path (from config.py)
    • Per-epoch metrics dict returned for plotting / logging
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CHECKPOINT, CLASS_NAMES as CLASS_NAMES_PATH,
    EPOCHS, WARMUP_EPOCHS, LR, WEIGHT_DECAY,
    AUX_LOSS_WEIGHT, LABEL_SMOOTHING,
    EARLY_STOP_PATIENCE, MODELS_DIR,
)
from .model import CropDiseaseModel, CombinedLoss


class Trainer:
    """
    Encapsulates the complete training loop.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, class_names)
        history = trainer.fit()
    """

    def __init__(
        self,
        model        : CropDiseaseModel,
        train_loader : DataLoader,
        val_loader   : DataLoader,
        class_names  : List[str],
        epochs       : int   = EPOCHS,
        warmup_epochs: int   = WARMUP_EPOCHS,
        lr           : float = LR,
        weight_decay : float = WEIGHT_DECAY,
        patience     : int   = EARLY_STOP_PATIENCE,
        device       : str   = None,
    ):
        self.class_names   = class_names
        self.epochs        = epochs
        self.warmup_epochs = warmup_epochs
        self.patience      = patience

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.criterion = CombinedLoss(
            aux_weight=AUX_LOSS_WEIGHT,
            label_smoothing=LABEL_SMOOTHING,
        )
        self.optimizer = self._make_optimizer(lr, weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        self.best_val_acc  = 0.0
        self.bad_epochs    = 0
        self.history: List[Dict] = []

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        print(f"[trainer] Device      : {self.device}")
        print(f"[trainer] Parameters  : {model.param_count}")
        print(f"[trainer] Train size  : {len(train_loader.dataset)}")
        print(f"[trainer] Val size    : {len(val_loader.dataset)}")
        print(f"[trainer] Classes     : {len(class_names)}")

    def _make_optimizer(self, lr, wd):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=wd,
        )

    # ── Main training loop ────────────────────────────────────────────────────

    def fit(self) -> List[Dict]:
        print("\n" + "="*62)
        print(" Training started")
        print("="*62)

        for epoch in range(self.epochs):
            t0 = time.time()

            # Switch to fine-tuning phase
            if epoch == self.warmup_epochs:
                print(f"\n[trainer] Epoch {epoch+1}: Switching to fine-tuning (backbone unfrozen)")
                self.model.unfreeze_backbone()
                self.optimizer = self._make_optimizer(
                    lr=self.optimizer.param_groups[0]["lr"] * 0.1,
                    wd=WEIGHT_DECAY,
                )
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.epochs - self.warmup_epochs,
                    eta_min=1e-6,
                )

            tr = self._run_epoch(train=True)
            va = self._run_epoch(train=False)
            self.scheduler.step()

            row = {
                "epoch"    : epoch + 1,
                "lr"       : round(self.optimizer.param_groups[0]["lr"], 8),
                "tr_loss"  : round(tr["loss"], 4),
                "tr_acc"   : round(tr["acc"],  4),
                "val_loss" : round(va["loss"], 4),
                "val_acc"  : round(va["acc"],  4),
                "time_s"   : round(time.time() - t0, 1),
            }
            self.history.append(row)
            self._log(row)

            # Checkpoint best model
            if va["acc"] > self.best_val_acc:
                self.best_val_acc = va["acc"]
                self.bad_epochs   = 0
                self._save()
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.patience:
                    print(f"\n[trainer] Early stop — no improvement for {self.patience} epochs")
                    break

        self._save_artifacts()
        return self.history

    # ── Single epoch ──────────────────────────────────────────────────────────

    def _run_epoch(self, train: bool) -> Dict:
        self.model.train(train)
        loader = self.train_loader if train else self.val_loader
        total_loss = correct = total = 0

        ctx = torch.enable_grad if train else torch.no_grad
        with ctx():
            for batch in loader:
                imgs = batch["image"].to(self.device, non_blocking=True)
                lbl  = batch["label"].to(self.device, non_blocking=True)
                dis  = batch["is_diseased"].to(self.device, non_blocking=True)

                logits, disease_prob = self.model(imgs)
                loss, _ = self.criterion(logits, disease_prob, lbl, dis)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                bs = imgs.size(0)
                total_loss += loss.item() * bs
                correct    += (logits.argmax(1) == lbl).sum().item()
                total      += bs

        return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        torch.save({
            "model_state" : self.model.state_dict(),
            "class_names" : self.class_names,
            "num_classes" : self.model.num_classes,
            "backbone"    : self.model.backbone_name,
            "val_acc"     : self.best_val_acc,
        }, CHECKPOINT)

    def _save_artifacts(self):
        with open(CLASS_NAMES_PATH, "w") as f:
            json.dump(self.class_names, f, indent=2)
        with open(MODELS_DIR / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n[trainer] Best val accuracy : {self.best_val_acc*100:.2f}%")
        print(f"[trainer] Checkpoint        : {CHECKPOINT}")

    @staticmethod
    def _log(r: Dict):
        print(
            f"  Ep {r['epoch']:03d} | lr {r['lr']:.1e} | "
            f"loss {r['tr_loss']:.4f}/{r['val_loss']:.4f} | "
            f"acc {r['tr_acc']*100:.1f}%/{r['val_acc']*100:.1f}% | "
            f"{r['time_s']}s"
        )


# ── Top-level convenience function ────────────────────────────────────────────

def train(
    data_dir  : str,
    backbone  : str   = "mobilenet_v3_small",
    epochs    : int   = EPOCHS,
    batch_size: int   = 32,
    lr        : float = LR,
) -> List[Dict]:
    """
    One-call training function.

    Args:
        data_dir  : PlantVillage root directory (or synthetic dir)
        backbone  : "mobilenet_v3_small" | "efficientnet_b0"
        epochs    : maximum training epochs
        batch_size: images per batch
        lr        : initial learning rate

    Returns:
        List of per-epoch metric dicts
    """
    from modules.dataset import get_dataloaders
    from modules.model   import build_model

    train_dl, val_dl, class_names = get_dataloaders(
        data_dir, batch_size=batch_size, num_workers=0,
    )
    model   = build_model(len(class_names), backbone=backbone)
    trainer = Trainer(
        model, train_dl, val_dl, class_names,
        epochs=epochs, warmup_epochs=max(2, epochs//6),
        lr=lr, patience=max(3, epochs//4),
    )
    return trainer.fit()
