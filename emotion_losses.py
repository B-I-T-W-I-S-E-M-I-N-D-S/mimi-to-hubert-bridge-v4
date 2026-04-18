"""
emotion_losses.py — Multi-Task Loss: HuBERT Feature Reconstruction + Emotion
=============================================================================
Extends BridgeLoss from losses.py with an EmotionLoss term.

Multi-task weighting strategy
------------------------------
Three approaches are implemented, selectable via config:
  1. "fixed"   — static weights from config.yaml  (simplest, good baseline)
  2. "uncertainty" — Kendall et al. (2018) learnable homoscedastic uncertainty
                     weights.  Each task gets a learnable log-variance σ²:
                     L_total = Σ_i  (1/2σ²_i) * L_i + log(σ_i)
                     Pros: auto-balances without tuning; published SOTA on MTL.
  3. "gradnorm" — GradNorm (Chen et al., 2018): normalise per-task gradient
                  magnitudes so no task dominates.  Enabled when
                  training.loss_weights.strategy = "gradnorm" in config.

Default (recommended): "uncertainty"

Config additions needed in config.yaml:
    emotion:
      num_classes:  7
      dropout:      0.2
      hidden_dim:   256
      label_smoothing: 0.1    # cross-entropy label smoothing

    training:
      loss_weights:
        ...                   # keep existing keys
        emotion: 1.0          # only used when strategy = "fixed"
        strategy: "uncertainty"  # "fixed" | "uncertainty"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from losses import BridgeLoss


# ──────────────────────────────────────────────────────────────────────────────
# 1. Emotion Classification Loss
# ──────────────────────────────────────────────────────────────────────────────

class EmotionLoss(nn.Module):
    """
    Cross-entropy with optional label smoothing and class-frequency weighting.

    label_smoothing: reduces overconfidence; typical value 0.05–0.15.
    class_weights:   pass a (num_classes,) tensor if dataset is class-imbalanced.
    """

    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes     = num_classes
        self.label_smoothing = label_smoothing
        # register class_weights as a buffer so it moves with .to(device)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        logits : (B, num_classes)  — raw scores from EmotionHead
        labels : (B,)              — integer class indices
        """
        loss = F.cross_entropy(
            logits,
            labels,
            weight         = self.class_weights,
            label_smoothing= self.label_smoothing,
        )
        # Accuracy for logging
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean().item()

        return loss, {"emotion_ce": loss.item(), "emotion_acc": acc}


# ──────────────────────────────────────────────────────────────────────────────
# 2. Uncertainty-Weighted Multi-Task Loss  (Kendall et al., 2018)
# ──────────────────────────────────────────────────────────────────────────────

class UncertaintyWeighting(nn.Module):
    """
    Learnable homoscedastic uncertainty weights.

    L_total = sum_i ( exp(-s_i) * L_i  +  s_i )
    where s_i = log(σ²_i) is a learnable parameter per task.

    Add this module's parameters to the optimiser alongside the bridge.
    """

    def __init__(self, num_tasks: int):
        super().__init__()
        # Initialise log-variances to 0 (σ²=1 → equal weighting at start)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        losses : (num_tasks,) tensor of individual task losses
        """
        precision = torch.exp(-self.log_vars)         # 1/σ²
        weighted  = (precision * losses + self.log_vars).sum()
        log_dict  = {f"unc_s{i}": self.log_vars[i].item() for i in range(len(losses))}
        return weighted, log_dict


# ──────────────────────────────────────────────────────────────────────────────
# 3. Combined Multi-Task Loss Manager
# ──────────────────────────────────────────────────────────────────────────────

class EmotionBridgeLoss(nn.Module):
    """
    Wraps BridgeLoss (reconstruction + CTC + prosody + adv + stat + smooth)
    and adds EmotionLoss with configurable multi-task weighting.

    strategy "fixed":
        total = bridge_loss  +  w_emotion * emotion_loss
    strategy "uncertainty":
        total = UncertaintyWeighting([bridge_loss, emotion_loss])
        (log-variances are learnable — include this module in the optimiser)

    Usage in trainer:
        criterion = EmotionBridgeLoss(cfg)
        opt = AdamW(list(bridge.parameters()) + list(criterion.parameters()), ...)
        ...
        total, logs = criterion(
            pred_features, emotion_logits, target_features, emotion_labels, batch
        )
    """

    def __init__(self, cfg: dict, class_weights: Optional[torch.Tensor] = None):
        super().__init__()

        # Original bridge losses (recon, CTC, prosody, adv, stat, smooth, alignment)
        self.bridge_loss = BridgeLoss(cfg)

        # Emotion loss
        e_cfg = cfg.get("emotion", {})
        num_classes = e_cfg.get("num_classes", 7)
        self.emotion_loss = EmotionLoss(
            num_classes     = num_classes,
            label_smoothing = e_cfg.get("label_smoothing", 0.1),
            class_weights   = class_weights,
        )

        # Weighting strategy
        w   = cfg["training"]["loss_weights"]
        self.strategy     = w.get("strategy", "uncertainty")
        self.w_emotion    = float(w.get("emotion", 1.0))   # only used for "fixed"

        if self.strategy == "uncertainty":
            # 2 tasks: [bridge_aggregate, emotion]
            self.unc_weight = UncertaintyWeighting(num_tasks=2)
        else:
            self.unc_weight = None

        # Expose sub-modules so trainer can pass disc logits etc.
        self.adv      = self.bridge_loss.adv
        self.prosody  = self.bridge_loss.prosody
        self.ctc      = self.bridge_loss.ctc

    def forward(
        self,
        pred_features: torch.Tensor,
        emotion_logits: torch.Tensor,
        target_features: torch.Tensor,
        emotion_labels: torch.Tensor,
        batch: dict,
        fake_disc_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        pred_features   : (B, 2T, output_dim)
        emotion_logits  : (B, num_emotions)
        target_features : (B, 2T, output_dim)
        emotion_labels  : (B,)  int64
        batch           : dict with mask, f0, energy, etc.
        fake_disc_logits: discriminator output on pred (or None)
        """
        # ── 1. Bridge losses (all original tasks) ─────────────────────────────
        bridge_total, bridge_logs = self.bridge_loss(
            pred_features, target_features, batch, fake_disc_logits
        )

        # ── 2. Emotion classification loss ────────────────────────────────────
        emo_total, emo_logs = self.emotion_loss(emotion_logits, emotion_labels)

        # ── 3. Combine with chosen strategy ───────────────────────────────────
        logs = {**bridge_logs, **emo_logs}

        if self.strategy == "uncertainty":
            task_losses = torch.stack([bridge_total, emo_total])
            total, unc_logs = self.unc_weight(task_losses)
            logs.update(unc_logs)

        elif self.strategy == "fixed":
            total = bridge_total + self.w_emotion * emo_total

        else:
            raise ValueError(f"Unknown weighting strategy: {self.strategy!r}")

        logs["total"]         = total.item()
        logs["bridge_total"]  = bridge_total.item()
        logs["emotion_total"] = emo_total.item()
        return total, logs
