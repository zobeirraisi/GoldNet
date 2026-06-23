"""GPU driver: evaluate ONLY the new 2025 comparison models
(iFormer-S, OverLoCK-XT) through the canonical evaluate_models_2 deep-learning
pipeline — i.e. the *exact same* 5-fold protocol, optimiser, schedule and
augmentation as the existing Table 4 entries — and write the numbers to
results/results_new_models.csv.

This does NOT re-run the already-published baselines (ResNet/VGG/.../ViT) and
does NOT touch the main results table.

Usage:
    python run_new_models.py                 # both new models
    python run_new_models.py --only iformers # single model
    python run_new_models.py --smoke         # 1 fold, 2 epochs (sanity only)
"""
import csv
import argparse
from pathlib import Path

import numpy as np

import evaluate_models_2 as E
from evaluate_models_2 import run_dl_model, GoldDataset, set_seed

DATA_DIR = str(Path(__file__).resolve().parent / "gold")
OUT_DIR  = Path(__file__).resolve().parent / "results"

NEW_MODELS = [
    ("iFormer-S",   "iformers"),
    ("OverLoCK-XT", "overlockxt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None,
                    help="run a single model key (iformers | overlockxt)")
    ap.add_argument("--smoke", action="store_true",
                    help="fast sanity run: 1 fold, few epochs, no compile")
    ap.add_argument("--no-compile", dest="no_compile", action="store_true",
                    help="disable torch.compile (use if triton is unavailable)")
    args = ap.parse_args()

    if args.no_compile:
        E.COMPILE = False

    if args.smoke:
        # shrink the protocol so we only check the wiring, not real accuracy.
        # N_FOLDS must stay >=2 (StratifiedKFold requirement).
        E.N_FOLDS     = 2
        E.NUM_EPOCHS  = 2
        E.UNFREEZE_EP = 1
        E.PATIENCE    = 99
        E.COMPILE     = False
        E.BATCH_SIZE  = min(E.BATCH_SIZE, 32)

    set_seed()
    probe      = GoldDataset(DATA_DIR)
    all_labels = np.array(probe.labels)
    all_idx    = np.arange(len(all_labels))
    del probe

    models = NEW_MODELS
    if args.only:
        models = [(d, k) for d, k in NEW_MODELS if k == args.only]
        if not models:
            raise SystemExit(f"--only must be one of "
                             f"{[k for _, k in NEW_MODELS]}")

    results = {}
    for display, key in models:
        mean, std = run_dl_model(display, key, DATA_DIR, all_idx, all_labels)
        results[display] = dict(
            acc_m=mean[0], acc_s=std[0], prec_m=mean[1], prec_s=std[1],
            rec_m=mean[2], rec_s=std[2], f1_m=mean[3], f1_s=std[3])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / ("results_new_models_smoke.csv" if args.smoke
                          else "results_new_models.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "acc_mean", "acc_std", "prec_mean", "prec_std",
                    "rec_mean", "rec_std", "f1_mean", "f1_std"])
        for name, r in results.items():
            w.writerow([name,
                        f"{r['acc_m']:.2f}",  f"{r['acc_s']:.2f}",
                        f"{r['prec_m']:.4f}", f"{r['prec_s']:.4f}",
                        f"{r['rec_m']:.4f}",  f"{r['rec_s']:.4f}",
                        f"{r['f1_m']:.4f}",   f"{r['f1_s']:.4f}"])

    print("\n==== NEW MODEL NUMBERS (5-fold mean +/- std) ====")
    for name, r in results.items():
        print(f"{name:<14} acc={r['acc_m']:.2f}+/-{r['acc_s']:.2f}  "
              f"prec={r['prec_m']:.4f}  rec={r['rec_m']:.4f}  "
              f"f1={r['f1_m']:.4f}+/-{r['f1_s']:.4f}")
    print("saved ->", csv_path)


if __name__ == "__main__":
    main()
