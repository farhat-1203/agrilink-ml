"""
modules/model.py  ─  Lightweight Disease Classifier with Dual Heads

Architecture:
    ┌─────────────────────────────────────────────┐
    │  MobileNetV3-Small (or EfficientNet-B0)      │
    │  Pre-trained on ImageNet                      │
    │  Early layers frozen during warm-up           │
    └──────────────────────┬──────────────────────┘
                           │  feature vector
                    ┌──────┴──────┐
                    │  Shared     │
                    │  projection │
                    └──┬──────┬──┘
                       │      │
              ┌────────┘      └──────────────┐
              ▼                              ▼
    disease_head                       binary_head
    Linear → N classes                 Linear → 1 (sigmoid)
    (CrossEntropy)                     (BCE  — healthy vs diseased)
    Main task                          Auxiliary task

Why dual heads?
  PlantVillage labels = specific disease class.
  The binary head learns a cleaner "is anything wrong?" signal
  which feeds directly into the freshness estimator.

CombinedLoss = 0.8 × CrossEntropy(main) + 0.2 × BCE(aux)
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BACKBONE, DROPOUT


class CropDiseaseModel(nn.Module):
    """MobileNetV3-Small or EfficientNet-B0 backbone + dual classification heads."""

    def __init__(
        self,
        num_classes : int,
        backbone    : str   = BACKBONE,
        dropout     : float = DROPOUT,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.backbone_name = backbone

        # ── 1. Load backbone (without pretrained weights — offline friendly) ──
        if backbone == "mobilenet_v3_small":
            try:
                from torchvision.models import MobileNet_V3_Small_Weights
                base = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
                print(f"[model] Loaded ImageNet weights for {backbone}")
            except Exception:
                base = models.mobilenet_v3_small(weights=None)
                print(f"[model] No pretrained weights available — random init for {backbone}")
            self.features  = base.features
            self.avgpool   = base.avgpool
            self._feat_dim = base.classifier[0].in_features  # 576

        elif backbone == "efficientnet_b0":
            try:
                from torchvision.models import EfficientNet_B0_Weights
                base = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
                print(f"[model] Loaded ImageNet weights for {backbone}")
            except Exception:
                base = models.efficientnet_b0(weights=None)
                print(f"[model] No pretrained weights available — random init for {backbone}")
            self.features  = base.features
            self.avgpool   = base.avgpool
            self._feat_dim = base.classifier[1].in_features  # 1280

        else:
            raise ValueError(f"Unsupported backbone: {backbone!r}. "
                             f"Choose 'mobilenet_v3_small' or 'efficientnet_b0'.")

        # ── 2. Freeze first 8 backbone blocks (warm-up phase) ─────────────────
        self._set_backbone_grad(freeze_n=8)

        # ── 3. Shared projection (backbone output → shared embedding) ─────────
        self.shared = nn.Sequential(
            nn.Linear(self._feat_dim, 512),
            nn.Hardswish(),
            nn.Dropout(dropout),
        )

        # ── 4. Disease classification head (N classes, main task) ─────────────
        self.disease_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.Hardswish(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        # ── 5. Binary head: healthy(0) vs diseased(1) ─────────────────────────
        self.binary_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),   # raw logit; sigmoid applied outside
        )

        self._kaiming_init()

    # ── Backbone freeze / unfreeze ────────────────────────────────────────────

    def _set_backbone_grad(self, freeze_n: int):
        """Freeze the first `freeze_n` children of self.features."""
        for i, child in enumerate(self.features.children()):
            requires = (i >= freeze_n)
            for p in child.parameters():
                p.requires_grad = requires

    def unfreeze_backbone(self):
        """Call after warm-up to allow full fine-tuning."""
        for p in self.features.parameters():
            p.requires_grad = True

    def _kaiming_init(self):
        for module in [self.shared, self.disease_head, self.binary_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    nn.init.zeros_(layer.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (B, 3, H, W) normalised image tensor
        Returns:
            logits       : (B, num_classes)  — disease class scores
            disease_prob : (B,)              — P(image is diseased) in [0,1]
        """
        feat = self.features(x)
        feat = self.avgpool(feat).flatten(start_dim=1)
        emb  = self.shared(feat)
        return self.disease_head(emb), torch.sigmoid(self.binary_head(emb)).squeeze(1)

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor) -> dict:
        """Convenience wrapper — returns softmax probs + disease prob."""
        self.eval()
        logits, disease_prob = self(x)
        probs = torch.softmax(logits, dim=1)
        top_p, top_i = probs.topk(min(3, self.num_classes), dim=1)
        return {
            "probs"       : probs,
            "top_probs"   : top_p,
            "top_indices" : top_i,
            "disease_prob": disease_prob,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def param_count(self) -> str:
        total  = sum(p.numel() for p in self.parameters())
        tunable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"{total/1e6:.2f}M total | {tunable/1e6:.2f}M trainable"


class CombinedLoss(nn.Module):
    """
    Multi-task loss for the dual-head model.
        total = (1 - aux_w) * CrossEntropy + aux_w * BCE
    Label smoothing on CrossEntropy reduces overconfidence on noisy labels.
    """

    def __init__(self, aux_weight: float = 0.2, label_smoothing: float = 0.1):
        super().__init__()
        self.ce  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bce = nn.BCELoss()
        self.aux_w = aux_weight

    def forward(
        self,
        logits      : torch.Tensor,   # (B, C)
        disease_prob: torch.Tensor,   # (B,)
        labels      : torch.Tensor,   # (B,) long
        is_diseased : torch.Tensor,   # (B,) float
    ) -> Tuple[torch.Tensor, dict]:
        ce_loss  = self.ce(logits, labels)
        bce_loss = self.bce(disease_prob, is_diseased)
        total    = (1 - self.aux_w) * ce_loss + self.aux_w * bce_loss
        return total, {"ce": ce_loss.item(), "bce": bce_loss.item()}


def build_model(num_classes: int, backbone: str = BACKBONE) -> CropDiseaseModel:
    """Factory — creates model on CPU; move to device in trainer."""
    return CropDiseaseModel(num_classes=num_classes, backbone=backbone)
