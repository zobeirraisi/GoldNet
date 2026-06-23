"""
Gold Jewelry Authentication — Model Comparison  (RTX 4090 Optimised)
=====================================================================
Evaluates 9 models under 5-fold stratified cross-validation:

  Deep Learning : ResNet18 · ResNet50 · VGG16 · DenseNet121
                  EfficientNet-B0 · MobileNet-V2 · ViT-B/16
  Classical ML  : LBP + SVM · Haralick + SVM

Metrics: Accuracy (%) | Precision | Recall | F1-Score  (mean ± std)

RTX 4090 optimisations applied
  • torch.compile()           — fused kernels (PyTorch >= 2.0)
  • torch.amp (bfloat16)      — native Ampere dtype, no overflow risk
  • cudnn.benchmark = True    — fastest cuDNN algorithm auto-select
  • pin_memory + persistent_workers + prefetch_factor
  • Large batch (128)         — fills the 24 GB VRAM comfortably
  • Parallel feature extraction for classical models (joblib)
  • CosineAnnealingWarmRestarts LR schedule
  • zero_grad(set_to_none=True) + gradient clipping

Intermediate CSV + LaTeX are saved after every model finishes,
so no progress is lost if the run is interrupted.

Usage
-----
  pip install torch torchvision timm scikit-learn scikit-image \
              numpy Pillow pandas tqdm joblib
  python evaluate_models.py --data_dir /path/to/gold [--out_dir ./results]

Dataset layout
--------------
  gold/
    authentic/    (1044 images)
    counterfeit/  (1083 images)
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import argparse
import random
import time
import warnings
from datetime import timedelta
from pathlib import Path

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import torchvision.models as models
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import timm
from tqdm import tqdm
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION  (overridable via CLI flags)
# ═══════════════════════════════════════════════════════════════════════════
SEED         = 42
N_FOLDS      = 5
IMG_SIZE     = 224
BATCH_SIZE   = 128        # RTX 4090 24 GB handles all models at BS=128
NUM_EPOCHS   = 100
LR_HEAD      = 1e-3       # head-only phase learning rate
LR_FULL      = 1e-4       # fine-tune phase learning rate
WEIGHT_DECAY = 1e-4
UNFREEZE_EP  = 10         # epoch at which backbone is unfrozen
PATIENCE     = 15         # early-stopping patience (generous for 100 epochs)
NUM_WORKERS  = 8
PREFETCH     = 4
COMPILE      = True       # set False if PyTorch < 2.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark     = True
cudnn.deterministic = False   # allow non-deterministic for speed

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

W = 90   # console ruler width


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════
# PRETTY PRINTING
# ═══════════════════════════════════════════════════════════════════════════
def section(title: str):
    pad = max(0, (W - len(title) - 2) // 2)
    print(f"\n{'═' * W}")
    print(f"{'═' * pad} {title} {'═' * max(0, W - pad - len(title) - 2)}")
    print(f"{'═' * W}")


def rule():
    print(f"  {'─' * (W - 2)}")


def metric_line(label: str, acc: float, prec: float, rec: float, f1: float):
    print(f"  │  {label:<20}  Acc={acc:>7.3f}%  "
          f"Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# DATASET & DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════
train_tf = T.Compose([
    T.Resize((IMG_SIZE + 32, IMG_SIZE + 32),
             interpolation=T.InterpolationMode.BICUBIC),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE),
             interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class GoldDataset(Dataset):
    """Authentic (0) / Counterfeit (1) image dataset."""

    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples: list = []
        self.labels:  list = []

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        for label, folder in enumerate(["authentic", "counterfeit"]):
            p = Path(root) / folder
            if not p.exists():
                raise FileNotFoundError(f"Missing folder: {p}")
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in exts:
                    self.samples.append((str(f), label))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def make_loaders(root: str, train_idx, val_idx):
    train_ds = Subset(GoldDataset(root, train_tf), train_idx)
    val_ds   = Subset(GoldDataset(root, eval_tf),  val_idx)
    kw = dict(num_workers=NUM_WORKERS, pin_memory=True,
              persistent_workers=(NUM_WORKERS > 0),
              prefetch_factor=PREFETCH if NUM_WORKERS > 0 else None)
    return (DataLoader(train_ds, BATCH_SIZE, shuffle=True,  **kw),
            DataLoader(val_ds,   BATCH_SIZE, shuffle=False, **kw))


# ═══════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════════
def _key(name: str) -> str:
    return name.lower().replace("-", "").replace("/", "").replace("_", "")


def build_model(name: str) -> nn.Module:
    k = _key(name)
    if k == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, 2))

    elif k == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, 2))

    elif k == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        m.classifier[6] = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(m.classifier[6].in_features, 2))

    elif k == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        m.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(m.classifier.in_features, 2))

    elif k == "efficientnetb0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(m.classifier[1].in_features, 2))

    elif k == "mobilenetv2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        m.classifier[1] = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(m.classifier[1].in_features, 2))

    elif k == "vitb16":
        m = timm.create_model("vit_base_patch16_224", pretrained=True,
                               num_classes=2)

    elif k == "iformers":
        # iFormer-S (2025) — loaded from code/external/ via models.py helpers.
        from models import (_iformer_factory, _load_external_pretrained,
                            IFORMER_S_CKPT, IFORMER_S_URL)
        m = _iformer_factory()(pretrained=False, num_classes=2)
        _load_external_pretrained(m, IFORMER_S_CKPT, IFORMER_S_URL)

    elif k == "overlockxt":
        # OverLoCK-XT (CVPR 2025); use_ds=False -> single-tensor forward.
        from models import (_overlock_factory, _load_external_pretrained,
                            OVERLOCK_XT_CKPT, OVERLOCK_XT_URL)
        m = _overlock_factory()(pretrained=False, num_classes=2, use_ds=False)
        _load_external_pretrained(m, OVERLOCK_XT_CKPT, OVERLOCK_XT_URL)

    else:
        raise ValueError(f"Unknown model: '{name}'")
    return m


def _head_params(model: nn.Module, key: str):
    if key in ("resnet18", "resnet50"):   return list(model.fc.parameters())
    if key == "vgg16":                    return list(model.classifier[6].parameters())
    if key == "densenet121":              return list(model.classifier.parameters())
    if key == "efficientnetb0":           return list(model.classifier[1].parameters())
    if key == "mobilenetv2":              return list(model.classifier[1].parameters())
    if key == "vitb16":                   return list(model.head.parameters())
    if key == "iformers":                 return list(model.classifier.parameters())
    if key == "overlockxt":               return list(model.head.parameters())
    return list(model.parameters())


def freeze_backbone(model: nn.Module, key: str):
    for p in model.parameters():   p.requires_grad_(False)
    for p in _head_params(model, key): p.requires_grad_(True)


def unfreeze_all(model: nn.Module):
    for p in model.parameters():   p.requires_grad_(True)


# ═══════════════════════════════════════════════════════════════════════════
# TRAIN / VALIDATE
# ═══════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, epoch_bar):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for bi, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(1) == labels).sum().item()
        total         += bs

        epoch_bar.set_postfix(
            batch=f"{bi+1}/{len(loader)}",
            tr_loss=f"{total_loss/total:.4f}",
            tr_acc=f"{total_correct/total*100:.1f}%",
            refresh=False)

    return total_loss / total, total_correct / total * 100


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds_all, labels_all = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        total_loss   += loss.item() * imgs.size(0)
        preds_all.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    y_t = np.array(labels_all)
    y_p = np.array(preds_all)
    return total_loss / len(loader.dataset), y_t, y_p


def compute_metrics(y_true, y_pred):
    return (accuracy_score(y_true, y_pred) * 100,
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred,    zero_division=0),
            f1_score(y_true, y_pred,         zero_division=0))


# ═══════════════════════════════════════════════════════════════════════════
# DEEP LEARNING CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
def run_dl_model(display_name: str, model_key: str,
                 dataset_root: str, all_indices, all_labels):
    section(display_name)
    key = _key(model_key)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(all_indices, all_labels), 1):

        set_seed(SEED + fold_idx)
        print(f"\n  ┌── Fold {fold_idx}/{N_FOLDS}  "
              f"(train={len(train_idx)}  val={len(val_idx)}) ─────────────────────")

        train_dl, val_dl = make_loaders(dataset_root, train_idx, val_idx)

        model = build_model(model_key).to(DEVICE)
        freeze_backbone(model, key)

        if COMPILE and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                print("  │  torch.compile ✓")
            except Exception as e:
                print(f"  │  torch.compile skipped ({e})")

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        scaler    = GradScaler()

        # Phase 1 — head only
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        best_val_loss = float("inf")
        best_val_acc  = 0.0
        patience_cnt  = 0
        best_state    = None
        t_fold        = time.time()

        # ── epoch bar ────────────────────────────────────────────────────────
        epoch_bar = tqdm(
            range(1, NUM_EPOCHS + 1),
            desc=f"  │  Fold {fold_idx}",
            unit="ep",
            ncols=W,
            leave=True,
            bar_format=(
                "  │  {l_bar}{bar}| {n_fmt}/{total_fmt}ep "
                "[{elapsed}<{remaining}] {postfix}"
            ),
        )

        for epoch in epoch_bar:

            # ── backbone unfreeze ────────────────────────────────────────────
            if epoch == UNFREEZE_EP + 1:
                unfreeze_all(model)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=LR_FULL, weight_decay=WEIGHT_DECAY)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=max(1, NUM_EPOCHS - UNFREEZE_EP),
                    T_mult=1, eta_min=1e-7)
                epoch_bar.write(f"  │  ── backbone unfrozen (lr={LR_FULL:.0e}) ──")

            tr_loss, tr_acc = train_one_epoch(
                model, train_dl, optimizer, criterion, scaler, epoch_bar)
            val_loss, y_t, y_p = validate(model, val_dl, criterion)
            scheduler.step(epoch)

            v_acc = accuracy_score(y_t, y_p) * 100
            v_f1  = f1_score(y_t, y_p, zero_division=0)
            lr    = optimizer.param_groups[0]["lr"]

            epoch_bar.set_postfix(
                trL=f"{tr_loss:.4f}", trA=f"{tr_acc:.1f}%",
                vL=f"{val_loss:.4f}",  vA=f"{v_acc:.1f}%",
                F1=f"{v_f1:.4f}",      lr=f"{lr:.1e}",
                refresh=True)

            # early stopping
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_val_acc  = v_acc
                best_state    = {k: v.cpu().clone()
                                 for k, v in model.state_dict().items()}
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    epoch_bar.write(
                        f"  │  Early stop @ epoch {epoch} "
                        f"(no improvement for {PATIENCE} epochs)")
                    break

        # ── fold final eval ──────────────────────────────────────────────────
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        _, y_t, y_p = validate(model, val_dl, criterion)
        m = compute_metrics(y_t, y_p)
        fold_metrics.append(m)

        elapsed = timedelta(seconds=int(time.time() - t_fold))
        metric_line(f"Fold {fold_idx} result", *m)
        print(f"  │  Best val_loss={best_val_loss:.4f}  "
              f"best_val_acc={best_val_acc:.2f}%  time={elapsed}")
        print(f"  └{'─' * (W - 4)}┘")

        del model
        torch.cuda.empty_cache()

    arr  = np.array(fold_metrics)
    mean, std = arr.mean(0), arr.std(0)

    rule()
    print(f"  ✔  {display_name}  ── 5-fold CV complete")
    metric_line("Mean", mean[0], mean[1], mean[2], mean[3])
    print(f"  │  {'Std':<20}  "
          f"±{std[0]:.3f}%    ±{std[1]:.4f}    ±{std[2]:.4f}    ±{std[3]:.4f}")
    rule()
    return mean, std


# ═══════════════════════════════════════════════════════════════════════════
# CLASSICAL ML PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def _lbp(path: str, P: int = 8, R: int = 1, n_bins: int = 256) -> np.ndarray:
    img  = np.array(Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    gray = img_as_ubyte(rgb2gray(img))
    lbp  = local_binary_pattern(gray, P=P, R=R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def _haralick(path: str) -> np.ndarray:
    img  = np.array(Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    gray = img_as_ubyte(rgb2gray(img))
    glcm = graycomatrix(gray, distances=[1, 3],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity",
             "energy", "correlation", "ASM"]
    return np.concatenate(
        [graycoprops(glcm, p).ravel() for p in props]).astype(np.float32)


def run_classical_model(display_name: str, feature_fn,
                        dataset_root: str):
    section(display_name)
    ds     = GoldDataset(dataset_root)
    paths  = [s[0] for s in ds.samples]
    labels = np.array(ds.labels)

    t0 = time.time()
    print(f"  Extracting features in parallel (joblib, n_jobs=-1) …")
    X = np.array(
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(feature_fn)(p)
            for p in tqdm(paths, desc="  Features", ncols=W, unit="img")))
    print(f"  Feature matrix: {X.shape}  "
          f"({timedelta(seconds=int(time.time()-t0))})")

    skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold_idx, (tr, val) in enumerate(skf.split(X, labels), 1):
        sc  = StandardScaler()
        Xtr = sc.fit_transform(X[tr]);   Xval = sc.transform(X[val])
        clf = SVC(kernel="rbf", C=10, gamma="scale", cache_size=2000)
        clf.fit(Xtr, labels[tr])
        m = compute_metrics(labels[val], clf.predict(Xval))
        fold_metrics.append(m)
        metric_line(f"Fold {fold_idx}", *m)

    arr  = np.array(fold_metrics)
    mean, std = arr.mean(0), arr.std(0)
    rule()
    print(f"  ✔  {display_name}  ── 5-fold CV complete")
    metric_line("Mean", mean[0], mean[1], mean[2], mean[3])
    print(f"  │  {'Std':<20}  "
          f"±{std[0]:.3f}%    ±{std[1]:.4f}    ±{std[2]:.4f}    ±{std[3]:.4f}")
    rule()
    return mean, std


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT: CSV + LaTeX  (written after every model)
# ═══════════════════════════════════════════════════════════════════════════
CLASSICAL = {"LBP + SVM", "Haralick + SVM"}


def _rows(results: dict) -> list:
    return [{
        "Model":        n,
        "Accuracy (%)": f"{r['acc_m']:.2f} ± {r['acc_s']:.2f}",
        "Precision":    f"{r['prec_m']:.4f} ± {r['prec_s']:.4f}",
        "Recall":       f"{r['rec_m']:.4f} ± {r['rec_s']:.4f}",
        "F1-Score":     f"{r['f1_m']:.4f} ± {r['f1_s']:.4f}",
    } for n, r in results.items()]


def _best_models(rows: list) -> dict:
    cols = ["Accuracy (%)", "Precision", "Recall", "F1-Score"]
    best = {}
    for c in cols:
        top_name, top_val = None, -1.0
        for r in rows:
            v = float(r[c].split(" ±")[0])
            if v > top_val:
                top_val, top_name = v, r["Model"]
        best[c] = top_name
    return best


def save_csv(results: dict, path: Path):
    pd.DataFrame(_rows(results)).to_csv(path, index=False)
    print(f"  💾 CSV   → {path}")


def save_latex(results: dict, path: Path):
    rows = _rows(results)
    best = _best_models(rows)
    cols = ["Accuracy (%)", "Precision", "Recall", "F1-Score"]

    def cell(r, c):
        v = r[c]
        return (f"$\\mathbf{{{v}}}$" if best.get(c) == r["Model"]
                else f"${v}$")

    lines = [
        r"\begin{table}[H]",
        r"\caption{Five-fold cross-validation results (mean $\pm$ std across folds). "
        r"Bold denotes best performance per metric."
        r"\label{tab:model_comparison}}",
        r"\begin{adjustwidth}{-\extralength}{0cm}",
        r"\begin{tabularx}{\fulllength}{lCCCC}",
        r"\toprule",
        (r"\textbf{Model} & \textbf{Accuracy (\%)} & \textbf{Precision} "
         r"& \textbf{Recall} & \textbf{F1-Score} \\"),
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Classical texture baselines}} \\",
    ]
    for r in [x for x in rows if x["Model"] in CLASSICAL]:
        lines.append(
            f"{r['Model']} & {cell(r,cols[0])} & {cell(r,cols[1])} "
            f"& {cell(r,cols[2])} & {cell(r,cols[3])} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{l}{\textit{Deep learning models}} \\")
    for r in [x for x in rows if x["Model"] not in CLASSICAL]:
        lines.append(
            f"{r['Model']} & {cell(r,cols[0])} & {cell(r,cols[1])} "
            f"& {cell(r,cols[2])} & {cell(r,cols[3])} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}",
              r"\end{adjustwidth}", r"\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  💾 LaTeX → {path}")


def save_results(results: dict, out_path: Path):
    save_csv(results,   out_path / "results_table2.csv")
    save_latex(results, out_path / "table2.tex")


def print_final_table(results: dict):
    section("FINAL RESULTS SUMMARY")
    print(f"  {'Model':<20}  {'Accuracy (%)':>14}  {'Precision':>14}  "
          f"{'Recall':>14}  {'F1-Score':>14}")
    rule()

    best_acc = max(v["acc_m"] for v in results.values())
    best_f1  = max(v["f1_m"]  for v in results.values())

    prev_classical = True
    for name, r in results.items():
        is_dl = name not in CLASSICAL
        if is_dl and prev_classical:
            print(f"\n  ── Deep Learning {'─' * 57}")
            prev_classical = False

        acc_tag = "  ← best" if abs(r["acc_m"] - best_acc) < 0.005 else ""
        f1_tag  = " / best F1" if abs(r["f1_m"]  - best_f1) < 1e-4  else ""
        print(
            f"  {name:<20}  "
            f"{r['acc_m']:>6.2f}±{r['acc_s']:.2f}%    "
            f"{r['prec_m']:.4f}±{r['prec_s']:.4f}    "
            f"{r['rec_m']:.4f}±{r['rec_s']:.4f}    "
            f"{r['f1_m']:.4f}±{r['f1_s']:.4f}"
            f"{acc_tag}{f1_tag}"
        )
    rule()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main(data_dir: str, out_dir: str):
    set_seed()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    section("GOLD AUTHENTICATION — MODEL COMPARISON")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Device       : {DEVICE}  ({gpu})")
    print(f"  Data dir     : {data_dir}")
    print(f"  Output dir   : {out_path.resolve()}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Epochs       : {NUM_EPOCHS}  "
          f"(backbone unfreezes @ ep {UNFREEZE_EP + 1})")
    print(f"  Early stop   : patience = {PATIENCE}")
    print(f"  torch.compile: {COMPILE and hasattr(torch, 'compile')}")
    print(f"  AMP dtype    : bfloat16")

    probe      = GoldDataset(data_dir)
    n_auth     = probe.labels.count(0)
    n_fake     = probe.labels.count(1)
    n_total    = len(probe.labels)
    all_labels = np.array(probe.labels)
    all_idx    = np.arange(n_total)
    print(f"  Dataset      : {n_auth} authentic + {n_fake} counterfeit "
          f"= {n_total} total")
    del probe

    results: dict = {}
    t_total = time.time()

    # ── Classical models ─────────────────────────────────────────────────────
    for name, fn in [("LBP + SVM",      _lbp),
                     ("Haralick + SVM", _haralick)]:
        t0 = time.time()
        mean, std = run_classical_model(name, fn, data_dir)
        results[name] = dict(acc_m=mean[0], acc_s=std[0],
                              prec_m=mean[1], prec_s=std[1],
                              rec_m=mean[2],  rec_s=std[2],
                              f1_m=mean[3],   f1_s=std[3])
        print(f"  ⏱  {name} finished in "
              f"{timedelta(seconds=int(time.time()-t0))}")
        save_results(results, out_path)   # intermediate save

    # ── Deep learning models ─────────────────────────────────────────────────
    for display, key in [
        # ("ResNet18",        "resnet18"),
        ("ResNet50",        "resnet50"),
        ("VGG16",           "vgg16"),
        ("DenseNet121",     "densenet121"),
        ("EfficientNet-B0", "efficientnetb0"),
        ("MobileNet-V2",    "mobilenetv2"),
        ("ViT-B/16",        "vitb16"),
    ]:
        t0 = time.time()
        mean, std = run_dl_model(display, key, data_dir, all_idx, all_labels)
        results[display] = dict(acc_m=mean[0], acc_s=std[0],
                                prec_m=mean[1], prec_s=std[1],
                                rec_m=mean[2],  rec_s=std[2],
                                f1_m=mean[3],   f1_s=std[3])
        print(f"  ⏱  {display} finished in "
              f"{timedelta(seconds=int(time.time()-t0))}")
        save_results(results, out_path)   # intermediate save

    # ── Final summary ────────────────────────────────────────────────────────
    print_final_table(results)
    save_results(results, out_path)
    total = timedelta(seconds=int(time.time() - t_total))
    print(f"\n  Total wall-clock time: {total}")
    section("DONE")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gold authentication — model comparison (RTX 4090 optimised)")
    parser.add_argument("--data_dir",   required=True,
                        help="Path to gold/ (contains authentic/ and counterfeit/)")
    parser.add_argument("--out_dir",    default="./results",
                        help="Output directory for CSV + LaTeX  (default: ./results)")
    parser.add_argument("--epochs",     type=int, default=NUM_EPOCHS,
                        help=f"Training epochs per fold  (default: {NUM_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size  (default: {BATCH_SIZE})")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile  (use with PyTorch < 2.0)")
    args = parser.parse_args()

    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    if args.no_compile:
        COMPILE = False

    main(args.data_dir, args.out_dir)