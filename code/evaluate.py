"""
evaluate.py
-----------
Gold Jewelry Authentication — Evaluation, calibration, and metrics.

Implements:
  - Full metric suite: Accuracy, Precision, Recall, F1, AUC, MCC, ECE
  - Temperature scaling (post-hoc calibration)
  - Confidence-based abstention with accuracy-coverage curve
  - Per-subtype performance breakdown
  - Cross-domain robustness evaluation
  - OOD confidence analysis
  - Paired t-test with Bonferroni correction for significance testing
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import LBFGS

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Inference utilities
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
    temperature: float = 1.0,
) -> Dict:
    """
    Run model inference and collect logits, probabilities, predictions,
    labels, subtypes, and item IDs.
    """
    model.eval()
    all_logits, all_probs, all_preds = [], [], []
    all_labels, all_subtypes, all_items = [], [], []

    for batch in loader:
        images, labels, subtypes, item_ids = batch
        images = images.to(device)

        logits = model(images)
        if temperature != 1.0:
            logits = logits / temperature

        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_logits.append(logits.cpu())
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.extend(labels.tolist())
        all_subtypes.extend(subtypes)
        all_items.extend(item_ids)

    return {
        "logits":   torch.cat(all_logits),
        "probs":    torch.cat(all_probs),
        "preds":    torch.cat(all_preds).numpy(),
        "labels":   np.array(all_labels),
        "subtypes": all_subtypes,
        "items":    all_items,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric suite
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    labels: np.ndarray,
    preds:  np.ndarray,
    probs:  np.ndarray,
) -> Dict[str, float]:
    """
    Compute full metric suite used in the paper.
    probs: shape (N, 2), column 1 = P(authentic)
    """
    return {
        "accuracy":  accuracy_score(labels, preds) * 100,
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "auc":       roc_auc_score(labels, probs[:, 1]),
        "mcc":       matthews_corrcoef(labels, preds),
    }


def compute_ece(
    labels:    np.ndarray,
    probs:     np.ndarray,
    n_bins:    int = 10,
) -> float:
    """
    Expected Calibration Error with equal-frequency binning.
    ECE = sum_m |B_m|/N * |acc(B_m) - conf(B_m)|
    """
    confidences = probs.max(axis=1)
    preds       = probs.argmax(axis=1)
    correct     = (preds == labels).astype(float)
    n           = len(labels)

    # equal-frequency bins
    sorted_idx  = np.argsort(confidences)
    bins        = np.array_split(sorted_idx, n_bins)

    ece = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        acc  = correct[b].mean()
        conf = confidences[b].mean()
        ece += len(b) / n * abs(acc - conf)

    return ece * 100   # return as percentage


# ─────────────────────────────────────────────────────────────────────────────
# Temperature scaling calibration
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """
    Single-parameter post-hoc calibration via temperature scaling.
    Learns T* by minimising NLL on the validation set.
    Classification accuracy is unchanged; only confidence magnitudes shift.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)

    def fit(
        self,
        model:  nn.Module,
        loader: DataLoader,
        device: torch.device,
        max_iter: int = 50,
    ) -> float:
        """
        Fit temperature on validation loader.
        Returns the learned T* value.
        """
        model.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                images, labels = batch[0].to(device), batch[1]
                logits = model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        logits_all = torch.cat(all_logits)
        labels_all = torch.cat(all_labels)

        self.to("cpu")
        criterion = nn.CrossEntropyLoss()
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            scaled = self(logits_all)
            loss   = criterion(scaled, labels_all)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return self.temperature.item()

    def calibrated_probs(self, logits: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            scaled = self(logits)
            return F.softmax(scaled, dim=1).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Abstention / accuracy-coverage analysis
# ─────────────────────────────────────────────────────────────────────────────

def accuracy_coverage_curve(
    labels:     np.ndarray,
    probs:      np.ndarray,
    n_thresholds: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Compute accuracy vs coverage for a range of confidence thresholds τ.
    Returns arrays of tau, coverage (%), and accuracy (%).
    """
    max_conf = probs.max(axis=1)
    preds    = probs.argmax(axis=1)
    correct  = (preds == labels)

    taus     = np.linspace(0.5, 0.999, n_thresholds)
    coverages, accuracies = [], []

    for tau in taus:
        mask = max_conf >= tau
        cov  = mask.mean() * 100
        if mask.sum() == 0:
            acc = 100.0
        else:
            acc = correct[mask].mean() * 100
        coverages.append(cov)
        accuracies.append(acc)

    return {
        "tau":      taus,
        "coverage": np.array(coverages),
        "accuracy": np.array(accuracies),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-subtype evaluation
# ─────────────────────────────────────────────────────────────────────────────

def per_subtype_metrics(
    labels:   np.ndarray,
    preds:    np.ndarray,
    probs:    np.ndarray,
    subtypes: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute accuracy and F1 broken down by item subtype.
    """
    subtypes = np.array(subtypes)
    results  = {}

    for subtype in np.unique(subtypes):
        mask = subtypes == subtype
        if mask.sum() == 0:
            continue
        results[subtype] = {
            "n":        int(mask.sum()),
            "accuracy": accuracy_score(labels[mask], preds[mask]) * 100,
            "f1":       f1_score(labels[mask], preds[mask], zero_division=0),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Statistical significance testing
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_significance(
    fold_accs: Dict[str, List[float]],
    reference: str = "efficientnet_b0",
    alpha:     float = 0.05,
) -> Dict[str, Dict]:
    """
    Paired t-test with Bonferroni correction comparing each model
    against the reference model across cross-validation folds.

    Parameters
    ----------
    fold_accs : dict of model_name -> list of per-fold accuracies
    reference : the reference model key
    alpha     : family-wise error rate

    Returns
    -------
    dict of model_name -> {t_stat, p_value, p_corrected, significant}
    """
    ref_accs  = fold_accs[reference]
    n_compare = len(fold_accs) - 1
    results   = {}

    for model_name, accs in fold_accs.items():
        if model_name == reference:
            continue
        t_stat, p_val = scipy_stats.ttest_rel(ref_accs, accs)
        p_corrected   = min(p_val * n_compare, 1.0)   # Bonferroni
        results[model_name] = {
            "t_stat":      t_stat,
            "p_value":     p_val,
            "p_corrected": p_corrected,
            "significant": p_corrected < alpha,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Computational efficiency measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_inference_time(
    model:      nn.Module,
    device:     torch.device,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    n_runs:     int = 1000,
    batch_size_throughput: int = 32,
) -> Dict[str, float]:
    """
    Measure GPU/CPU latency (batch=1) and throughput (batch=32).
    Excludes data loading I/O — forward pass only.
    """
    import time

    model.eval()
    dummy_single = torch.randn(input_size).to(device)
    dummy_batch  = torch.randn(batch_size_throughput, *input_size[1:]).to(device)

    # warmup
    with torch.no_grad():
        for _ in range(50):
            model(dummy_single)

    # latency (batch=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy_single)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)   # ms

    latency_mean = np.mean(times)
    latency_std  = np.std(times)

    # throughput (batch=32)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            model(dummy_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed     = time.perf_counter() - t0
    throughput  = (100 * batch_size_throughput) / elapsed   # img/s

    return {
        "latency_mean_ms": latency_mean,
        "latency_std_ms":  latency_std,
        "throughput_img_s": throughput,
    }
