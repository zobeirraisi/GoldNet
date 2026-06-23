"""
trainer.py
----------
Gold Jewelry Authentication — Training engine.

Implements:
  - Progressive fine-tuning (3-phase schedule)
  - Discriminative learning rates
  - Weighted cross-entropy loss
  - Adam optimiser with ReduceLROnPlateau
  - Early stopping
  - Checkpoint management
"""

from __future__ import annotations

import os
import time
import copy
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models import GoldAuthModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Training configuration
# ─────────────────────────────────────────────────────────────────────────────

class TrainConfig:
    # Optimiser
    head_lr:       float = 1e-3
    weight_decay:  float = 1e-4
    beta1:         float = 0.9
    beta2:         float = 0.999
    eps:           float = 1e-8

    # Schedule
    max_epochs:    int   = 15
    batch_size:    int   = 32
    lr_patience:   int   = 3
    lr_factor:     float = 0.5
    early_stop_patience: int = 5

    # Progressive fine-tuning phases
    phase1_end:    int   = 3   # epochs 1-3:  head only
    phase2_end:    int   = 6   # epochs 4-6:  last block
    # phase 3:     epochs 7-15: full network

    # Misc
    seed:          int   = 42
    device:        str   = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str  = "checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:

    def __init__(
        self,
        model:        GoldAuthModel,
        loaders:      Dict[str, DataLoader],
        class_weights: Optional[torch.Tensor] = None,
        config:       TrainConfig = None,
        run_name:     str = "run",
    ):
        self.model   = model
        self.loaders = loaders
        self.config  = config or TrainConfig()
        self.run_name = run_name

        cfg = self.config
        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        # Loss
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Checkpoint dir
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # State
        self.best_val_loss  = float("inf")
        self.best_weights   = None
        self.no_improve     = 0
        self.history        = {"train_loss": [], "val_loss": [],
                               "train_acc": [],  "val_acc": []}

    # ── Main train loop ───────────────────────────────────────────────────────

    def train(self) -> Dict:
        cfg     = self.config
        model   = self.model

        for epoch in range(1, cfg.max_epochs + 1):
            phase = self._get_phase(epoch)
            model.set_phase(phase)

            # rebuild optimiser when phase changes (new param groups)
            if epoch == 1 or epoch == cfg.phase1_end + 1 or epoch == cfg.phase2_end + 1:
                self.optimizer, self.scheduler = self._build_optimizer()

            # ── train epoch ──
            t0 = time.time()
            train_loss, train_acc = self._run_epoch("train")
            val_loss,   val_acc   = self._run_epoch("val")
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            self.scheduler.step(val_loss)

            logger.info(
                f"Epoch {epoch:02d}/{cfg.max_epochs} | Phase {phase} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"{elapsed:.1f}s"
            )

            # ── checkpoint & early stopping ──
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_weights  = copy.deepcopy(model.state_dict())
                self.no_improve    = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                self.no_improve += 1
                if self.no_improve >= cfg.early_stop_patience:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

        # restore best weights
        model.load_state_dict(self.best_weights)
        return self.history

    # ── Single epoch ──────────────────────────────────────────────────────────

    def _run_epoch(self, split: str) -> Tuple[float, float]:
        loader   = self.loaders[split]
        model    = self.model
        device   = self.device
        training = (split == "train")

        model.train(training)
        total_loss, correct, n = 0.0, 0, 0

        with torch.set_grad_enabled(training):
            for batch in loader:
                images, labels = batch[0], batch[1]
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss   = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                preds    = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                n       += labels.size(0)
                total_loss += loss.item() * labels.size(0)

        return total_loss / n, correct / n

    # ── Utilities ────────────────────────────────────────────────────────────

    def _get_phase(self, epoch: int) -> int:
        cfg = self.config
        if epoch <= cfg.phase1_end:
            return 1
        elif epoch <= cfg.phase2_end:
            return 2
        return 3

    def _build_optimizer(self) -> Tuple[Adam, ReduceLROnPlateau]:
        cfg    = self.config
        groups = self.model.get_param_groups(cfg.head_lr)
        opt    = Adam(groups, lr=cfg.head_lr,
                      betas=(cfg.beta1, cfg.beta2),
                      eps=cfg.eps,
                      weight_decay=cfg.weight_decay)
        sched  = ReduceLROnPlateau(opt, mode="min",
                                   patience=cfg.lr_patience,
                                   factor=cfg.lr_factor,
                                   verbose=False)
        return opt, sched

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        path = Path(self.config.checkpoint_dir) / f"{self.run_name}_best.pt"
        torch.save({
            "epoch":      epoch,
            "arch":       self.model.arch,
            "state_dict": self.best_weights,
            "val_loss":   val_loss,
        }, path)
        logger.debug(f"Checkpoint saved → {path}")
