"""CPU-only driver: re-run ONLY the classical SVM baselines (LBP, Haralick)
using the canonical 5-fold pipeline from evaluate_models_2.py, and rewrite
results/results_table2.csv. Does NOT touch the deep-model (GPU) path.

Replicates the classical block of evaluate_models_2.main() verbatim.
"""
from pathlib import Path

from evaluate_models_2 import (
    run_classical_model, _lbp, _haralick, save_results, set_seed,
)

DATA_DIR = str(Path(__file__).resolve().parent / "gold")
OUT_DIR  = Path(__file__).resolve().parent / "results"

set_seed()
results = {}
for name, fn in [("LBP + SVM", _lbp), ("Haralick + SVM", _haralick)]:
    mean, std = run_classical_model(name, fn, DATA_DIR)
    results[name] = dict(
        acc_m=mean[0], acc_s=std[0],
        prec_m=mean[1], prec_s=std[1],
        rec_m=mean[2], rec_s=std[2],
        f1_m=mean[3], f1_s=std[3],
    )

save_results(results, OUT_DIR)

print("\n==== FRESH SVM BASELINE NUMBERS (5-fold mean +/- std) ====")
for name, r in results.items():
    print(f"{name:<16} acc={r['acc_m']:.2f}+/-{r['acc_s']:.2f}  "
          f"prec={r['prec_m']:.4f}+/-{r['prec_s']:.4f}  "
          f"rec={r['rec_m']:.4f}+/-{r['rec_s']:.4f}  "
          f"f1={r['f1_m']:.4f}+/-{r['f1_s']:.4f}")
