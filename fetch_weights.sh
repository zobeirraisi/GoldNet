#!/usr/bin/env bash
# fetch_weights.sh - download all model weights needed to reproduce GoldNet / GoldFormer.
#
# Two groups are fetched:
#   1. Upstream ImageNet-1k inits for the 2025 baselines (iFormer-S, OverLoCK-XT).
#   2. The trained GoldNet checkpoints, attached as assets to a GitHub release.
#
# Files are downloaded as raw bytes and later loaded with
# torch.load(..., weights_only=True), which unpickles tensors without executing
# any code embedded in a checkpoint.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/weights"
mkdir -p "$DEST"

# --- helper: resumable, fail-loud download that skips files already present ----
get() {  # get <filename> <url>
  local out="$DEST/$1"
  if [ -s "$out" ]; then
    echo "[skip] $1 already present ($(stat -c%s "$out") bytes)"
    return
  fi
  echo "[get ] $1 <- $2"
  curl -fLsS -C - -o "$out" "$2"
  echo "[ok  ] $1 ($(stat -c%s "$out") bytes)"
}

# --- 1. upstream ImageNet-1k backbone inits (third-party, not redistributed) ---
get "iFormer_s.pth"             "https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_s.pth"
get "overlock_xt_in1k_224.pth"  "https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_xt_in1k_224.pth"

# The iFormer checkpoint contains numpy metadata that PyTorch 2.4.1 refuses to
# unpickle with weights_only=True. Sanitize it into a tensors-only copy:
if [ -s "$DEST/iFormer_s.pth" ] && [ ! -s "$DEST/iFormer_s_clean.pth" ]; then
  python scripts/sanitize_iformer_ckpt.py "$DEST/iFormer_s.pth" "$DEST/iFormer_s_clean.pth" || \
    echo "[warn] run scripts/sanitize_iformer_ckpt.py manually to create iFormer_s_clean.pth"
fi

# --- 2. trained GoldNet checkpoints (hosted on Hugging Face) -------------------
HF_BASE="https://huggingface.co/zobeir/GoldNet/resolve/main/weights"

TRAINED=(
  "GoldFormer_best.pth"
  "Swin_T_best.pth"
  "ViT_B16_best.pth"
  "ResNet101_best.pth"
  "ResNet50_best.pth"
  "ResNet18_best.pth"
  "DenseNet121_best.pth"
  "EfficientNet_B3_best.pth"
  "EfficientNet_B0_best.pth"
  "MobileNet_V2_best.pth"
)
for f in "${TRAINED[@]}"; do
  get "$f" "$HF_BASE/$f"
done

# --- note: natten (required only for OverLoCK-XT) -----------------------------
# natten ships from a separate index, not from PyPI:
#   pip install natten==0.17.4+torch240cu118 -f https://shi-labs.com/natten/wheels
# (the TLS cert on that host may be expired; add -k to curl/pip if needed.)

echo "Done. Weights in: $DEST"
