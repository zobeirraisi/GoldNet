"""
figures.py
----------
Gold Jewelry Authentication — All manuscript figures.

Generates publication-quality figures in PDF + PNG format
matching the academic style of MDPI Sensors.

Figures produced:
  Fig 1 — Confusion matrix (EfficientNet-B0, best fold)
  Fig 2 — Reliability diagrams before/after temperature scaling (2×4)
  Fig 3 — Accuracy–coverage trade-off curve
  Fig 4 — Cross-domain robustness grouped bar chart
  Fig 5 — Per-subtype horizontal bar chart
  Fig 6 — Training curves (loss + accuracy)
  Fig 7 — ROC curves for all models
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Optional

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "TeX Gyre Termes",
    "font.size":         14,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.15,
})

COLORS = {
    "authentic":    "#2166ac",
    "counterfeit":  "#d6604d",
    "challenging":  "#f4a742",
    "efficientnet": "#4dac26",
    "resnet":       "#d01c8b",
    "vit":          "#0571b0",
    "before":       "#d6604d",
    "after":        "#4393c3",
    "baseline":     "#555555",
    "star":         "#d6604d",
}


def save(fig: plt.Figure, out_dir: str, name: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    tp: int, fp: int, fn: int, tn: int,
    out_dir: str = "figures",
) -> None:
    cm       = np.array([[tp, fp], [fn, tn]])
    row_sum  = cm.sum(axis=1, keepdims=True)
    cm_norm  = cm / row_sum

    cell_colors = [
        [COLORS["authentic"],   "#fddbc7"],
        ["#fddbc7",             COLORS["authentic"]],
    ]
    labels = [["TP", "FP"], ["FN", "TN"]]

    fig, ax = plt.subplots(figsize=(5.5, 5))

    for i in range(2):
        for j in range(2):
            color   = cell_colors[i][j]
            tc      = "white" if color == COLORS["authentic"] else "#333333"
            count   = cm[i, j]
            pct     = cm_norm[i, j] * 100
            ax.add_patch(plt.Rectangle((j, 1-i), 1, 1,
                                        color=color, alpha=0.85, zorder=1))
            ax.text(j+0.5, 1-i+0.58, str(count),
                    ha="center", va="center", fontsize=30,
                    fontweight="bold", color=tc, zorder=2)
            ax.text(j+0.5, 1-i+0.28, f"{pct:.1f}%",
                    ha="center", va="center", fontsize=12,
                    color=tc, zorder=2)
            ax.text(j+0.5, 1-i+0.10, labels[i][j],
                    ha="center", va="center", fontsize=11,
                    color=tc, alpha=0.8, style="italic", zorder=2)

    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Authentic", "Counterfeit"], fontsize=13)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Counterfeit", "Authentic"], fontsize=13)
    ax.tick_params(length=0)
    ax.set_xlabel("Predicted label", fontsize=14, labelpad=10)
    ax.set_ylabel("True label",      fontsize=14, labelpad=10)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    save(fig, out_dir, "fig1_confusion_matrix")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Reliability Diagrams
# ─────────────────────────────────────────────────────────────────────────────

def plot_reliability_diagrams(
    models_data: List[Dict],
    out_dir:     str = "figures",
) -> None:
    """
    models_data: list of dicts with keys:
      name, ece_before, ece_after, T_star,
      bins_center, acc_before, acc_after
    """
    fig, axes = plt.subplots(2, len(models_data),
                              figsize=(16, 7.5), sharey=True, sharex=True)

    for col, m in enumerate(models_data):
        for row, (acc_vals, color, ece, T_val) in enumerate([
            (m["acc_before"], COLORS["before"], m["ece_before"], None),
            (m["acc_after"],  COLORS["after"],  m["ece_after"],  m["T_star"]),
        ]):
            ax    = axes[row][col]
            width = 0.18
            ax.bar(m["bins_center"], acc_vals, width=width,
                   color=color, alpha=0.75, edgecolor="white",
                   linewidth=0.5, zorder=2)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.5, zorder=3)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_yticks([0, 0.5, 1.0])
            ax.tick_params(labelsize=12)
            ax.set_title(m["name"] if row == 0 else "", fontsize=13, pad=7)
            info = f"ECE = {ece:.2f}%"
            if T_val:
                info += f"\n$T^*$ = {T_val}"
            ax.text(0.05, 0.93, info, transform=ax.transAxes,
                    fontsize=11, va="top", color="#333333")
            if col == 0:
                ax.set_ylabel("Accuracy", fontsize=13)
            if row == 1:
                ax.set_xlabel("Confidence", fontsize=13)

    patch_b = mpatches.Patch(color=COLORS["before"], alpha=0.75,
                              label="Before calibration")
    patch_a = mpatches.Patch(color=COLORS["after"],  alpha=0.75,
                              label="After calibration")
    diag    = Line2D([0], [0], color="black", linestyle="--",
                     alpha=0.5, label="Perfect calibration")
    fig.legend(handles=[patch_b, patch_a, diag], loc="lower center",
               ncol=3, fontsize=13, frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    save(fig, out_dir, "fig2_calibration_diagrams")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Accuracy–Coverage Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_coverage_curve(
    coverage:      np.ndarray,
    accuracy:      np.ndarray,
    taus:          np.ndarray,
    operating_tau: float = 0.90,
    baseline_acc:  float = 96.83,
    out_dir:       str   = "figures",
) -> None:
    # sort ascending by coverage so curve plots left→right
    order    = np.argsort(coverage)
    coverage = coverage[order]
    accuracy = accuracy[order]
    taus     = taus[order]

    op_mask  = np.isclose(taus, operating_tau, atol=0.02)
    op_x     = coverage[op_mask][0]  if op_mask.any() else coverage[len(coverage)//2]
    op_y     = accuracy[op_mask][0]  if op_mask.any() else accuracy[len(accuracy)//2]

    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    ax.fill_between(coverage, accuracy, baseline_acc - 0.4,
                    alpha=0.10, color=COLORS["authentic"])
    ax.plot(coverage, accuracy,
            color=COLORS["authentic"], linewidth=2.2,
            marker="o", markersize=7,
            markerfacecolor="white",
            markeredgecolor=COLORS["authentic"],
            markeredgewidth=2, zorder=4)

    # operating point star
    ax.plot(op_x, op_y, marker="*", markersize=20,
            color=COLORS["star"],
            markeredgecolor="#b5382a",
            markeredgewidth=1, zorder=5)

    # baseline
    ax.axhline(baseline_acc, color=COLORS["baseline"],
               linestyle="--", linewidth=1.2, alpha=0.6)
    ax.text(coverage.max() + 1.5, baseline_acc - 0.13,
            f"{baseline_acc:.2f}%\n(full cov.)",
            fontsize=9.5, va="top", ha="left",
            color=COLORS["baseline"], style="italic")

    # tau labels for key points
    for cov, acc, tau in zip(coverage, accuracy, taus):
        if tau in [0.50, 0.70, 0.80, 0.95, 0.99]:
            dx = 1.0 if cov < 95 else -1.5
            ha = "left" if dx > 0 else "right"
            ax.text(cov + dx, acc + 0.10,
                    f"$\\tau$={tau}", fontsize=9.5,
                    color="#555555", ha=ha)

    # operating point annotation
    ax.annotate(
        f"$\\tau = {operating_tau}$\n{op_y:.2f}% acc. · {op_x:.1f}% cov.",
        xy=(op_x, op_y), xytext=(75, op_y - 0.9),
        fontsize=10.5, color="#b5382a", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#b5382a", lw=1.4),
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff0ee",
                  ec="#d6604d", alpha=0.95),
    )

    ax.set_xlabel("Coverage (%)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_xlim(coverage.min() - 3, coverage.max() + 6)
    ax.set_ylim(accuracy.min() - 0.4, accuracy.max() + 0.4)
    ax.set_xticks([60, 70, 80, 90, 100])
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    save(fig, out_dir, "fig3_coverage_curve")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Cross-Domain Robustness
# ─────────────────────────────────────────────────────────────────────────────

def plot_robustness(
    conditions:   List[str],
    model_accs:   Dict[str, List[float]],
    baseline_acc: float = 96.83,
    out_dir:      str   = "figures",
) -> None:
    model_names  = list(model_accs.keys())
    color_list   = [COLORS["efficientnet"], COLORS["resnet"], COLORS["vit"]]
    x            = np.arange(len(conditions))
    width        = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, (name, accs) in enumerate(model_accs.items()):
        offset = (i - 1) * width
        bars   = ax.bar(x + offset, accs, width,
                        color=color_list[i % len(color_list)],
                        alpha=0.85, edgecolor="white",
                        linewidth=0.5, label=name, zorder=3)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    f"{val:.1f}",
                    ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color="#333333")

    ax.axhline(baseline_acc, color=COLORS["baseline"],
               linestyle="--", linewidth=1.2, alpha=0.6)
    ax.text(x[-1] + 0.55, baseline_acc + 0.15,
            f"{baseline_acc:.2f}%\nbaseline",
            fontsize=9.5, va="bottom", ha="right",
            color=COLORS["baseline"], style="italic")

    ax.set_xlabel("Distribution shift condition", fontsize=13, labelpad=8)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=12)
    ax.set_ylim(85, 100)
    ax.set_yticks([85, 88, 91, 94, 97, 100])
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11, frameon=False, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save(fig, out_dir, "fig4_robustness")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Per-Subtype Performance
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_subtype(
    subtypes:    List[str],
    accuracies:  List[float],
    f1_scores:   List[float],
    samples:     List[int],
    challenging: List[str],
    out_dir:     str = "figures",
) -> None:
    # colour each bar
    bar_colors = []
    for s in subtypes:
        if s in challenging:
            bar_colors.append(COLORS["challenging"])
        elif any(k in s for k in ["24K", "22K", "18K", "14K"]):
            bar_colors.append(COLORS["authentic"])
        else:
            bar_colors.append(COLORS["counterfeit"])

    fig, ax = plt.subplots(figsize=(9, 6.0))
    y    = np.arange(len(subtypes))
    bars = ax.barh(y, accuracies, color=bar_colors, alpha=0.82,
                   edgecolor="white", linewidth=0.5, height=0.55, zorder=3)

    # F1 marker
    for i, f1 in enumerate(f1_scores):
        ax.plot(f1 * 100, i, marker="|", markersize=14,
                markeredgewidth=2.5, color="#222222", zorder=5)

    # value labels
    for bar, acc, n, s in zip(bars, accuracies, samples, subtypes):
        dagger = "†" if n <= 30 else ""
        ax.text(bar.get_width() + 0.2,
                bar.get_y() + bar.get_height() / 2,
                f"{acc:.2f}%{dagger}  (n={n})",
                va="center", ha="left", fontsize=10.5, color="#333333")

    # section divider between authentic and counterfeit
    n_auth = sum(1 for s in subtypes
                 if any(k in s for k in ["24K", "22K", "18K", "14K"]))
    ax.axhline(n_auth - 0.5, color="#aaaaaa",
               linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(subtypes, fontsize=12)
    ax.set_xlabel("Accuracy (%)", fontsize=13, labelpad=8)
    ax.set_xlim(85, 107)
    ax.set_xticks([85, 88, 91, 94, 97, 100])
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.tick_params(labelsize=11)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    auth_p  = mpatches.Patch(color=COLORS["authentic"],   alpha=0.82, label="Authentic")
    cft_p   = mpatches.Patch(color=COLORS["counterfeit"], alpha=0.82, label="Counterfeit")
    chal_p  = mpatches.Patch(color=COLORS["challenging"], alpha=0.82, label="Challenging subtype")
    f1_line = Line2D([0], [0], marker="|", color="#222222", markersize=12,
                     markeredgewidth=2.5, linestyle="None", label="F1-score marker")
    ax.legend(handles=[auth_p, cft_p, chal_p, f1_line],
              fontsize=10.5, frameon=False, loc="lower right")

    plt.tight_layout()
    save(fig, out_dir, "fig5_per_subtype")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Training Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history:  Dict[str, List[float]],
    arch:     str = "EfficientNet-B0",
    out_dir:  str = "figures",
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Loss
    ax1.plot(epochs, history["train_loss"], color=COLORS["authentic"],
             linewidth=2.0, marker="o", markersize=4,
             markerfacecolor="white", label="Train loss")
    ax1.plot(epochs, history["val_loss"], color=COLORS["counterfeit"],
             linewidth=2.0, marker="s", markersize=4,
             markerfacecolor="white", linestyle="--", label="Val loss")
    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("Cross-entropy loss", fontsize=13)
    ax1.legend(fontsize=11, frameon=False)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax1.set_axisbelow(True)

    # Accuracy
    train_acc_pct = [a * 100 for a in history["train_acc"]]
    val_acc_pct   = [a * 100 for a in history["val_acc"]]
    ax2.plot(epochs, train_acc_pct, color=COLORS["authentic"],
             linewidth=2.0, marker="o", markersize=4,
             markerfacecolor="white", label="Train accuracy")
    ax2.plot(epochs, val_acc_pct, color=COLORS["counterfeit"],
             linewidth=2.0, marker="s", markersize=4,
             markerfacecolor="white", linestyle="--", label="Val accuracy")
    ax2.set_xlabel("Epoch", fontsize=13)
    ax2.set_ylabel("Accuracy (%)", fontsize=13)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.legend(fontsize=11, frameon=False)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax2.set_axisbelow(True)

    # Phase boundary lines
    for ax in (ax1, ax2):
        ax.axvline(3.5, color="#bbbbbb", linestyle=":", linewidth=1.0)
        ax.axvline(6.5, color="#bbbbbb", linestyle=":", linewidth=1.0)
        ax.text(2.0,  ax.get_ylim()[0], "Ph.1", fontsize=8,
                color="#999999", ha="center")
        ax.text(5.0,  ax.get_ylim()[0], "Ph.2", fontsize=8,
                color="#999999", ha="center")
        ax.text(10.0, ax.get_ylim()[0], "Ph.3", fontsize=8,
                color="#999999", ha="center")

    plt.tight_layout()
    save(fig, out_dir, "fig6_training_curves")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — ROC Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    roc_data: Dict[str, Dict],
    out_dir:  str = "figures",
) -> None:
    """
    roc_data: dict of model_name -> {fpr, tpr, auc}
    """
    from sklearn.metrics import auc as sk_auc

    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    color_cycle = [
        "#2166ac", "#d6604d", "#4dac26", "#762a83",
        "#e08214", "#1b7837", "#c51b7d", "#f6e8c3",
    ]

    for i, (name, data) in enumerate(roc_data.items()):
        ax.plot(data["fpr"], data["tpr"],
                color=color_cycle[i % len(color_cycle)],
                linewidth=1.8,
                label=f"{name} (AUC={data['auc']:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.4,
            label="Random classifier")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate",  fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(fontsize=9.5, frameon=False, loc="lower right")
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save(fig, out_dir, "fig7_roc_curves")
