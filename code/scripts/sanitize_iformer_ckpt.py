"""One-time sanitization of the official iFormer-S ImageNet checkpoint.

The released iFormer_s.pth is a full training checkpoint that stores the tensor
state_dict alongside benign numpy scalars (epoch, best-acc) and an argparse
Namespace. torch 2.4.1's weights_only=True unpickler cannot build those numpy
objects, so we cannot load it safely as-is.

This script performs a SINGLE trusted-source load (weights_only=False) of the
official file, keeps ONLY the tensor state_dict, and re-saves a clean,
tensors-only checkpoint (iFormer_s_clean.pth). From then on the pipeline loads
the clean file with weights_only=True — no pickle code execution ever again.

Run once:
    python scripts/sanitize_iformer_ckpt.py
"""
import os
from collections import OrderedDict

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(HERE, "external", "weights", "iFormer_s.pth")
DST  = os.path.join(HERE, "external", "weights", "iFormer_s_clean.pth")


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"Missing {SRC} — run scripts/fetch_weights.sh first.")
    if os.path.exists(DST):
        print(f"[skip] clean checkpoint already exists: {DST}")
        return

    # One-time, user-authorized trusted load of the official release file.
    ckpt = torch.load(SRC, map_location="cpu", weights_only=False)

    state = ckpt
    for key in ("model", "state_dict", "model_ema"):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            print(f"[info] using nested state under '{key}'")
            break

    clean, dropped = OrderedDict(), []
    for k, v in state.items():
        if torch.is_tensor(v):
            clean[k] = v.detach().cpu().contiguous()
        else:
            dropped.append((k, type(v).__name__))

    if not clean:
        raise SystemExit("No tensors found in checkpoint — aborting.")

    print(f"[ok ] kept {len(clean)} tensors; "
          f"dropped {len(dropped)} non-tensor entries")
    if dropped:
        print(f"       dropped sample: {dropped[:6]}")

    torch.save(clean, DST)
    print(f"[ok ] saved {DST} ({os.path.getsize(DST)} bytes)")

    # Verify the clean file is loadable with the SAFE path.
    verify = torch.load(DST, map_location="cpu", weights_only=True)
    print(f"[ok ] verified weights_only=True load: {len(verify)} tensors")


if __name__ == "__main__":
    main()
