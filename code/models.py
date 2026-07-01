"""
models.py
---------
Gold Jewelry Authentication — All evaluated architectures.

Implements:
  CNN family   : ResNet18, ResNet50, VGG16, DenseNet121, EfficientNet-B0, MobileNet-V2
  Transformer  : ViT-B/16
  Recent (2025): iFormer-S, OverLoCK-XT   (loaded from code/external/, see below)
  Classical    : LBP + SVM,  Haralick + SVM
  Proposed     : GoldFormer (ResNet-50 + Swin-T dual-stream, gated by TAAG)

Each deep model is wrapped in GoldAuthModel which handles:
  - ImageNet pretrained weight loading
  - Binary classification head replacement
  - Progressive fine-tuning phase management (freeze / unfreeze backbone)
"""

from __future__ import annotations

import os
import sys
import logging
import importlib.util

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# External 2025 architectures (iFormer, OverLoCK) — added for the revision (Phase B)
# ---------------------------------------------------------------------------
# These live in code/external/ (cloned from the official repos). We load the
# model definition files directly with importlib because:
#   1. timm.create_model() injects a `pretrained_cfg` kwarg that the iFormer
#      factories reject — calling the factory functions directly avoids this.
#   2. Both repos ship a top-level package literally named `models`, which would
#      collide with THIS file (`models.py`). Loading by file path under unique
#      module names ("iformer_ext", "overlock_ext") sidesteps the clash.
# Imports are lazy: the heavy deps (natten, mmengine) are only touched when an
# iFormer/OverLoCK model is actually requested.
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXT_DIR  = os.path.join(_THIS_DIR, "external")

# Official ImageNet-1k pretrained weights (release assets). These are fetched
# once into code/external/weights/ as raw bytes (see scripts/fetch_weights.sh)
# and loaded locally with torch.load(weights_only=True), i.e. WITHOUT executing
# arbitrary pickle code — only tensors are unpickled.
IFORMER_S_URL   = "https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_s.pth"
OVERLOCK_XT_URL = "https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_xt_in1k_224.pth"

_WEIGHTS_DIR     = os.path.join(_EXT_DIR, "weights")
# iFormer's released .pth bundles numpy metadata that torch 2.4.1 cannot load
# under weights_only=True, so the pipeline reads a sanitized tensors-only copy
# produced once by scripts/sanitize_iformer_ckpt.py.
IFORMER_S_RAW    = os.path.join(_WEIGHTS_DIR, "iFormer_s.pth")
IFORMER_S_CKPT   = os.path.join(_WEIGHTS_DIR, "iFormer_s_clean.pth")
OVERLOCK_XT_CKPT = os.path.join(_WEIGHTS_DIR, "overlock_xt_in1k_224.pth")

_ext_factory_cache: Dict[str, object] = {}


def _load_module_from_file(mod_name: str, file_path: str, extra_syspath: str = None):
    """Import a single .py file under a unique module name (no package clash)."""
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    if extra_syspath and extra_syspath not in sys.path:
        sys.path.insert(0, extra_syspath)
    spec   = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _iformer_factory():
    if "iformer_s" not in _ext_factory_cache:
        d   = os.path.join(_EXT_DIR, "iFormer", "models")
        mod = _load_module_from_file("iformer_ext",
                                     os.path.join(d, "iformer.py"),
                                     extra_syspath=d)
        _ext_factory_cache["iformer_s"] = mod.iFormer_s
    return _ext_factory_cache["iformer_s"]


def _overlock_factory():
    if "overlock_xt" not in _ext_factory_cache:
        d   = os.path.join(_EXT_DIR, "OverLoCK")
        mod = _load_module_from_file("overlock_ext",
                                     os.path.join(d, "models", "overlock.py"),
                                     extra_syspath=os.path.join(d, "models"))
        _ext_factory_cache["overlock_xt"] = mod.overlock_xt
    return _ext_factory_cache["overlock_xt"]


def _safe_torch_load(ckpt_path: str):
    """
    Load a checkpoint WITHOUT executing arbitrary pickle code (weights_only=True).

    Official training checkpoints often wrap the tensor state_dict alongside
    benign metadata: numpy scalars/dtypes (epoch, best-acc) and an
    argparse.Namespace of the training args. weights_only blocks these by
    default. We auto-allowlist ONLY such data-only globals — strictly limited to
    `numpy.*` and `argparse.Namespace`, which reconstruct plain data and cannot
    execute code on unpickling. Any other unexpected global aborts the load.
    """
    import re
    import argparse
    import importlib

    try:
        torch.serialization.add_safe_globals([argparse.Namespace])
    except Exception:
        pass

    def _legacy_alias(real_obj, qualname):
        # numpy 2.x relocated numpy.core.* -> numpy._core.*, so the real object's
        # __module__ no longer matches the legacy path stored in old pickles
        # (torch keys its allowlist by __module__.__name__). Register a thin
        # forwarding alias under the EXACT legacy name so the key matches; it only
        # forwards to numpy's own data reconstruction — no code execution.
        mod_name, _, name = qualname.rpartition(".")

        def _alias(*args, **kwargs):
            return real_obj(*args, **kwargs)

        _alias.__module__ = mod_name
        _alias.__name__ = name
        _alias.__qualname__ = name
        return _alias

    for _ in range(24):
        try:
            return torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001
            m = re.search(r"GLOBAL (\S+) ", str(exc))
            if not m:
                raise
            qualname = m.group(1)
            if not (qualname.startswith("numpy.") or
                    qualname == "argparse.Namespace"):
                # refuse to allowlist anything that isn't provably data-only
                raise
            mod_name, _, attr = qualname.rpartition(".")
            obj = None
            for candidate in (mod_name,
                              mod_name.replace("numpy.core", "numpy._core")):
                try:
                    obj = getattr(importlib.import_module(candidate), attr)
                    break
                except Exception:
                    continue
            if obj is None:
                raise
            torch.serialization.add_safe_globals([_legacy_alias(obj, qualname)])
    return torch.load(ckpt_path, map_location="cpu", weights_only=True)


def _load_external_pretrained(model: nn.Module, ckpt_path: str,
                              url: str = "") -> None:
    """
    Load locally-cached ImageNet-1k weights into a model whose head was built
    for 2 classes. The 1000-class classifier tensors are skipped via
    strict=False (shape mismatch), keeping only transferable backbone weights.
    """
    if not os.path.exists(ckpt_path):
        hint = "Run scripts/fetch_weights.sh"
        if ckpt_path.endswith("iFormer_s_clean.pth"):
            hint += " then scripts/sanitize_iformer_ckpt.py"
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {ckpt_path}\n"
            f"{hint}  (source: {url})")
    ckpt  = _safe_torch_load(ckpt_path)
    state = ckpt
    for key in ("model", "state_dict", "model_ema"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    cleaned = {(k[7:] if k.startswith("module.") else k): v
               for k, v in state.items()}
    # strict=False ignores missing/unexpected keys but NOT shape mismatches, so
    # drop tensors whose shape differs (the 1000-class head vs our 2-class head).
    model_sd = model.state_dict()
    filtered = {k: v for k, v in cleaned.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    skipped = [k for k in cleaned
               if k in model_sd and model_sd[k].shape != cleaned[k].shape]
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    logger.info(f"Pretrained {os.path.basename(ckpt_path)}: {len(filtered)} "
                f"tensors loaded, {len(skipped)} shape-skipped (head), "
                f"{len(unexpected)} unexpected.")


# ─────────────────────────────────────────────────────────────────────────────
# Deep learning wrapper
# ─────────────────────────────────────────────────────────────────────────────

class GoldAuthModel(nn.Module):
    """
    Unified wrapper for all CNN / ViT architectures.
    Replaces the original classification head with a binary
    classifier: Linear(d, 2) with dropout(0.5).
    """

    SUPPORTED = [
        "resnet18", "resnet50", "vgg16",
        "densenet121", "efficientnet_b0", "mobilenet_v2", "vit_b_16",
        "iformer_s", "overlock_xt",
    ]

    def __init__(self, arch: str, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        self.arch = arch
        self._build(arch, pretrained, dropout)

    # ── Construction ─────────────────────────────────────────────────────────

    def _build(self, arch: str, pretrained: bool, dropout: float) -> None:
        weights = "DEFAULT" if pretrained else None

        if arch == "resnet18":
            base = models.resnet18(weights=weights)
            d = base.fc.in_features
            base.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 2))
            self.backbone = base

        elif arch == "resnet50":
            base = models.resnet50(weights=weights)
            d = base.fc.in_features
            base.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 2))
            self.backbone = base

        elif arch == "vgg16":
            base = models.vgg16(weights=weights)
            d = base.classifier[-1].in_features
            base.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 2))
            self.backbone = base

        elif arch == "densenet121":
            base = models.densenet121(weights=weights)
            d = base.classifier.in_features
            base.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 2))
            self.backbone = base

        elif arch == "efficientnet_b0":
            base = models.efficientnet_b0(weights=weights)
            d = base.classifier[-1].in_features
            base.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 2))
            self.backbone = base

        elif arch == "mobilenet_v2":
            base = models.mobilenet_v2(weights=weights)
            d = base.classifier[-1].in_features
            base.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 2))
            self.backbone = base

        elif arch == "vit_b_16":
            # ImageNet-21k pretrained via IMAGENET21K_V1 weights
            base = models.vit_b_16(weights=weights)
            d = base.heads.head.in_features
            base.heads.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 2))
            self.backbone = base

        elif arch == "iformer_s":
            # iFormer-S (2025): hybrid Conv/attention, ~6.2M params.
            # Built directly with a 2-class head (its native BN_Linear classifier);
            # ImageNet backbone weights loaded via strict=False below.
            base = _iformer_factory()(pretrained=False, num_classes=2)
            if pretrained:
                _load_external_pretrained(base, IFORMER_S_CKPT, IFORMER_S_URL)
            self.backbone = base

        elif arch == "overlock_xt":
            # OverLoCK-XT (CVPR 2025): large-kernel ConvNet with top-down context.
            # use_ds=False disables the deep-supervision aux head so forward() always
            # returns a single logits tensor (compatible with the shared Trainer).
            base = _overlock_factory()(pretrained=False, num_classes=2,
                                       use_ds=False)
            if pretrained:
                _load_external_pretrained(base, OVERLOCK_XT_CKPT, OVERLOCK_XT_URL)
            self.backbone = base

        else:
            raise ValueError(f"Unsupported architecture: {arch}. "
                             f"Choose from {self.SUPPORTED}")

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        # OverLoCK returns dict(main=, aux=) when deep supervision is active;
        # we build it with use_ds=False, but guard defensively regardless.
        if isinstance(out, dict):
            return out["main"]
        return out

    # ── Progressive fine-tuning phase management ─────────────────────────────

    def set_phase(self, phase: int) -> None:
        """
        Phase 1 (1-3):  freeze backbone, train head only
        Phase 2 (4-6):  unfreeze last block, discriminative LR
        Phase 3 (7-15): unfreeze full network, discriminative LR schedule
        """
        if phase == 1:
            self._freeze_backbone()
        elif phase == 2:
            self._freeze_backbone()
            self._unfreeze_last_block()
        elif phase == 3:
            self._unfreeze_all()

    def _freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            if not self._is_head_param(name):
                param.requires_grad = False

    def _unfreeze_last_block(self) -> None:
        """Unfreeze the final residual block / transformer layer."""
        last_block_names = {
            "resnet18":        "layer4",
            "resnet50":        "layer4",
            "vgg16":           "features.28",
            "densenet121":     "features.denseblock4",
            "efficientnet_b0": "features.8",
            "mobilenet_v2":    "features.18",
            "vit_b_16":        "encoder.layers.encoder_layer_11",
            "iformer_s":       "stages.3",
            "overlock_xt":     "sub_blocks4",
        }
        target = last_block_names.get(self.arch, "")
        for name, param in self.backbone.named_parameters():
            if target in name or self._is_head_param(name):
                param.requires_grad = True

    def _unfreeze_all(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _is_head_param(self, name: str) -> bool:
        # Arch-specific head module names for the external 2025 models.
        arch_heads = {
            "iformer_s":   ("classifier",),   # Classfier(BN_Linear)
            "overlock_xt": ("head",),         # Conv2d classifier head
        }
        if self.arch in arch_heads:
            return any(h in name for h in arch_heads[self.arch])
        head_keywords = {"fc", "classifier", "heads"}
        return any(kw in name for kw in head_keywords)

    def get_param_groups(
        self, head_lr: float
    ) -> List[Dict]:
        """
        Discriminative learning rate parameter groups for Phase 3:
          - Head:          head_lr
          - Middle layers: 0.1 * head_lr
          - Early layers:  0.01 * head_lr
        """
        head_params, mid_params, early_params = [], [], []

        named = list(self.backbone.named_parameters())
        n = len(named)

        for i, (name, param) in enumerate(named):
            if not param.requires_grad:
                continue
            if self._is_head_param(name):
                head_params.append(param)
            elif i > n * 0.66:
                mid_params.append(param)
            else:
                early_params.append(param)

        return [
            {"params": head_params,  "lr": head_lr},
            {"params": mid_params,   "lr": head_lr * 0.1},
            {"params": early_params, "lr": head_lr * 0.01},
        ]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# GoldFormer: hybrid dual-stream CNN+Swin architecture with TAAG
#
# Does not go through GoldAuthModel: its forward() returns (logits, gamma) —
# gamma is the TAAG gate activation, kept for interpretability — rather than
# bare logits, and its two backbones (ResNet-50 + Swin-T) are fused rather
# than swapped in behind a single `.backbone`. Build with
# `build_model("goldformer")`, then `model.load_state_dict(torch.load(
# "external/weights/GoldFormer_best.pth", weights_only=True))` to reproduce
# the paper's checkpoint.
# ─────────────────────────────────────────────────────────────────────────────

class TextureAwareAttentionGate(nn.Module):
    """
    TAAG — Texture-Aware Attention Gate.

    Uses per-channel RMS texture energy from a Laplacian-of-Gaussian
    depthwise convolution on CNN feature maps to gate each dimension of
    the Swin Transformer feature vector.

    Args:
        cnn_channels  : number of CNN feature map channels (default 2048)
        swin_dim      : Swin feature dimensionality (default 768)
        reduction     : bottleneck reduction ratio in gate MLP (default 16)
    """

    def __init__(self,
                 cnn_channels: int = 2048,
                 swin_dim: int = 768,
                 reduction: int = 16,
                 ablation: str = "full") -> None:
        super().__init__()

        # ── Ablation switch (Reviewer 1 round-2, Comment 3) ────────────────
        # "full"        : LoG prior + per-channel gate + learned residual (default)
        # "no_taag"     : disable gating; pass raw Swin features through
        # "no_log"      : random-init learnable texture conv (no LoG prior)
        # "scalar_gate" : single scalar gate per sample (no per-channel control)
        # "mult_only"   : purely multiplicative gate (no learned residual)
        assert ablation in {"full", "no_taag", "no_log",
                             "scalar_gate", "mult_only"}, ablation
        self.ablation = ablation
        self.swin_dim = swin_dim

        # Depthwise texture filter for texture energy estimation
        self.texture_conv = nn.Conv2d(
            cnn_channels, cnn_channels,
            kernel_size=3, padding=1,
            groups=cnn_channels, bias=False,
        )
        # The LoG prior is the design under test; ablation "no_log" leaves the
        # depthwise conv at its default (random) initialisation instead.
        if ablation != "no_log":
            self._init_log_kernel()

        self.texture_bn = nn.BatchNorm2d(cnn_channels)

        # Two-layer bottleneck MLP: cnn_channels → hidden → gate_dim.
        # A scalar gate emits one value per sample; otherwise one per Swin dim.
        gate_dim = 1 if ablation == "scalar_gate" else swin_dim
        hidden = max(cnn_channels // reduction, 64)
        self.gate_mlp = nn.Sequential(
            nn.Linear(cnn_channels, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, gate_dim, bias=False),
        )

        # Learnable residual scale — initialised to 0 for training stability
        # At init GoldFormer == plain Swin-T; gate grows as alpha learns
        self.alpha = nn.Parameter(torch.zeros(1))

    def _init_log_kernel(self) -> None:
        """
        Initialise the depthwise conv with a 3×3 LoG kernel.
        This approximates the centre-surround response that underlies LBP.
        """
        log_kernel = torch.tensor(
            [[0., -1.,  0.],
             [-1.,  4., -1.],
             [0., -1.,  0.]], dtype=torch.float32
        )
        # Shape: (out_channels, in_channels/groups, kH, kW)
        weight = log_kernel.unsqueeze(0).unsqueeze(0)
        weight = weight.expand(self.texture_conv.weight.shape)
        with torch.no_grad():
            self.texture_conv.weight.copy_(weight)

    def forward(self,
                cnn_feat_map: torch.Tensor,
                swin_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cnn_feat_map : (B, C, H, W)  — spatial CNN feature map
            swin_feat    : (B, D)         — pooled Swin feature vector

        Returns:
            gated_swin   : (B, D)         — texture-gated Swin features
        """
        # ── Ablation: disable gating entirely (raw Swin features) ──────────
        if self.ablation == "no_taag":
            # Return a neutral gamma (0.5) so the gate sparsity loss, whose
            # target is rho=0.5, contributes nothing for this variant.
            gamma = torch.full((swin_feat.size(0), self.swin_dim), 0.5,
                               device=swin_feat.device, dtype=swin_feat.dtype)
            return swin_feat, gamma

        # ── Texture energy estimation ──────────────────────────────────────
        # Depthwise LoG convolution + BN
        tex = self.texture_conv(cnn_feat_map)      # (B, C, H, W)
        tex = self.texture_bn(tex)                 # (B, C, H, W)

        # Per-channel RMS energy: sqrt(mean(response^2)) over spatial dims
        energy = tex.pow(2).mean(dim=[2, 3]).sqrt()  # (B, C)

        # ── Gate computation ───────────────────────────────────────────────
        gate_logits = self.gate_mlp(energy)          # (B, D) or (B, 1)
        gamma       = torch.sigmoid(gate_logits)     # ∈ [0,1]
        if self.ablation == "scalar_gate":
            gamma = gamma.expand(-1, self.swin_dim)  # (B, 1) → (B, D)

        # ── Gated feature ──────────────────────────────────────────────────
        if self.ablation == "mult_only":
            # Purely multiplicative gate, no learned residual scale.
            gated = swin_feat * gamma                # (B, D)
        else:
            # Default: learned residual scale (alpha=0 at init → identity).
            gated = swin_feat + self.alpha * (swin_feat * gamma)  # (B, D)

        return gated, gamma  # return gamma for interpretability


class GoldFormer(nn.Module):
    """
    GoldFormer: Hybrid Dual-Stream Gold Authentication Network.

    Streams:
      CNN   → ResNet-50 (stages 1-2 frozen, 3-4 fine-tuned) → 2048-dim
      Swin  → Swin-T (fully fine-tuned)                      → 768-dim
      TAAG  → gates Swin features with CNN texture energy
      Fuse  → concat + residual projection → 512-dim
      Head  → Linear → 2 logits
    """

    CNN_DIM  = 2048
    SWIN_DIM = 768
    FUSE_DIM = 512
    NUM_CLS  = 2

    def __init__(self,
                 dropout: float = 0.35,
                 taag_reduction: int = 16,
                 ablation: str = "full") -> None:
        super().__init__()
        self.ablation = ablation

        # ── CNN Stream (ResNet-50) ─────────────────────────────────────────
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze stem + stages 1-2; fine-tune stages 3-4
        frozen_layers = [
            resnet.conv1, resnet.bn1,
            resnet.layer1, resnet.layer2,
        ]
        for layer in frozen_layers:
            for p in layer.parameters():
                p.requires_grad = False

        # We keep layer3, layer4, and avgpool; drop the FC head
        self.cnn_stem   = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.cnn_stage1 = resnet.layer1   # frozen
        self.cnn_stage2 = resnet.layer2   # frozen
        self.cnn_stage3 = resnet.layer3   # fine-tuned
        self.cnn_stage4 = resnet.layer4   # fine-tuned → spatial feat map
        self.cnn_gap    = nn.AdaptiveAvgPool2d(1)

        # ── Swin Stream ────────────────────────────────────────────────────
        swin = models.swin_t(
            weights=models.Swin_T_Weights.IMAGENET1K_V1)
        # Remove the Swin classification head
        self.swin_features = swin.features   # patch embed + 4 stages
        self.swin_norm      = swin.norm
        self.swin_pool      = swin.avgpool

        # ── Texture-Aware Attention Gate ───────────────────────────────────
        self.taag = TextureAwareAttentionGate(
            cnn_channels=self.CNN_DIM,
            swin_dim=self.SWIN_DIM,
            reduction=taag_reduction,
            ablation=ablation,
        )

        # ── Cross-Stream Fusion (residual) ─────────────────────────────────
        fuse_in = self.CNN_DIM + self.SWIN_DIM   # 2048 + 768 = 2816
        self.fuse_main = nn.Sequential(
            nn.Linear(fuse_in, self.FUSE_DIM, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.FUSE_DIM),
        )
        # Linear shortcut for residual connection (no bias, no activation)
        self.fuse_shortcut = nn.Linear(fuse_in, self.FUSE_DIM, bias=False)

        # ── Classification Head ────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(self.FUSE_DIM),
            nn.Dropout(dropout),
            nn.Linear(self.FUSE_DIM, self.NUM_CLS),
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _cnn_forward(self,
                     x: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            feat_map : (B, 2048, h, w)  spatial feature map from layer4
            feat_vec : (B, 2048)        globally pooled CNN descriptor
        """
        x = self.cnn_stem(x)
        x = self.cnn_stage1(x)
        x = self.cnn_stage2(x)
        x = self.cnn_stage3(x)
        feat_map = self.cnn_stage4(x)                   # (B, 2048, h, w)
        feat_vec = self.cnn_gap(feat_map).flatten(1)    # (B, 2048)
        return feat_map, feat_vec

    def _swin_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            feat_vec : (B, 768)  globally pooled Swin descriptor
        """
        x = self.swin_features(x)   # (B, H', W', 768) — NHWC layout
        x = self.swin_norm(x)
        # swin_pool expects (B, C, H, W) — permute before, squeeze after
        x = x.permute(0, 3, 1, 2)  # (B, 768, H', W')
        x = self.swin_pool(x)       # (B, 768, 1, 1)
        x = x.flatten(1)            # (B, 768)
        return x

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (B, 3, H, W) input image batch

        Returns:
            logits : (B, 2)   classification logits
            gamma  : (B, 768) TAAG gate activations (for interpretability)
        """
        # ── Dual-stream feature extraction ────────────────────────────────
        cnn_map, cnn_vec = self._cnn_forward(x)     # (B,2048,h,w), (B,2048)
        swin_vec         = self._swin_forward(x)    # (B, 768)

        # ── TAAG gating ────────────────────────────────────────────────────
        gated_swin, gamma = self.taag(cnn_map, swin_vec)  # (B,768), (B,768)

        # ── Cross-stream fusion ────────────────────────────────────────────
        concat = torch.cat([cnn_vec, gated_swin], dim=1)  # (B, 2816)
        h = self.fuse_main(concat) + self.fuse_shortcut(concat)  # (B, 512)

        # ── Classification ─────────────────────────────────────────────────
        logits = self.head(h)   # (B, 2)

        return logits, gamma


# ─────────────────────────────────────────────────────────────────────────────
# Classical texture baselines
# ─────────────────────────────────────────────────────────────────────────────

class LBPSVMClassifier:
    """
    Local Binary Pattern + RBF-SVM baseline.

    LBP: rotation-invariant uniform, radius=3, P=24 sampling points.
    Features: concatenated histograms from 4×4 spatial grid → 416-dim.
    SVM: RBF kernel, C=10, gamma=0.001.
    """

    def __init__(self, radius: int = 3, n_points: int = 24,
                 grid_x: int = 4, grid_y: int = 4,
                 C: float = 10.0, gamma: float = 0.001):
        self.radius   = radius
        self.n_points = n_points
        self.grid_x   = grid_x
        self.grid_y   = grid_y
        self.C        = C
        self.gamma    = gamma
        self.clf      = None
        self.n_bins   = n_points + 2   # uniform LBP bins

    def _extract_features(self, images: list) -> "np.ndarray":
        import numpy as np
        from skimage.feature import local_binary_pattern
        from skimage.color import rgb2gray

        feats = []
        for img in images:
            if isinstance(img, type(None)):
                continue
            gray = rgb2gray(np.array(img)) if img.mode == "RGB" else np.array(img)
            h, w = gray.shape
            gh, gw = h // self.grid_y, w // self.grid_x
            hist_concat = []
            for gy in range(self.grid_y):
                for gx in range(self.grid_x):
                    patch = gray[gy*gh:(gy+1)*gh, gx*gw:(gx+1)*gw]
                    lbp   = local_binary_pattern(
                        patch, self.n_points, self.radius, method="uniform"
                    )
                    hist, _ = np.histogram(lbp.ravel(), bins=self.n_bins,
                                           range=(0, self.n_bins), density=True)
                    hist_concat.append(hist)
            feats.append(np.concatenate(hist_concat))
        return np.array(feats)

    def fit(self, images: list, labels: list) -> None:
        from sklearn.svm import SVC
        X = self._extract_features(images)
        self.clf = SVC(C=self.C, kernel="rbf", gamma=self.gamma,
                       probability=True, random_state=42)
        self.clf.fit(X, labels)

    def predict(self, images: list) -> "np.ndarray":
        X = self._extract_features(images)
        return self.clf.predict(X)

    def predict_proba(self, images: list) -> "np.ndarray":
        X = self._extract_features(images)
        return self.clf.predict_proba(X)


class HaralickSVMClassifier:
    """
    Haralick GLCM features + Linear SVM baseline.

    GLCM offsets: [(1,0), (1,1), (0,1), (-1,1)]
    13 Haralick features per direction, averaged → 13-dim.
    SVM: linear kernel, C=1.
    """

    def __init__(self, C: float = 1.0):
        self.C   = C
        self.clf = None

    def _extract_features(self, images: list) -> "np.ndarray":
        import numpy as np
        from skimage.feature import graycomatrix, graycoprops
        from skimage.color import rgb2gray

        offsets   = [(1, 0), (1, 1), (0, 1), (-1, 1)]
        distances = [1]
        angles    = [d[1] for d in offsets]   # approximate via angle list

        props = ["contrast", "dissimilarity", "homogeneity",
                 "energy", "correlation", "ASM"]

        feats = []
        for img in images:
            gray  = rgb2gray(np.array(img))
            gray8 = (gray * 255).astype(np.uint8)
            glcm  = graycomatrix(gray8, distances=distances,
                                  angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                  levels=256, symmetric=True, normed=True)
            f = []
            for prop in props:
                vals = graycoprops(glcm, prop)
                f.append(vals.mean())
                f.append(vals.std())
            # additional: entropy
            eps = 1e-10
            entropy = -np.sum(glcm * np.log2(glcm + eps), axis=(0, 1)).mean()
            f.append(entropy)
            feats.append(f)
        return np.array(feats)

    def fit(self, images: list, labels: list) -> None:
        from sklearn.svm import SVC
        import numpy as np
        X = self._extract_features(images)
        self.clf = SVC(C=self.C, kernel="linear",
                       probability=True, random_state=42)
        self.clf.fit(X, labels)

    def predict(self, images: list) -> "np.ndarray":
        X = self._extract_features(images)
        return self.clf.predict(X)

    def predict_proba(self, images: list) -> "np.ndarray":
        X = self._extract_features(images)
        return self.clf.predict_proba(X)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(arch: str, pretrained: bool = True):
    """
    Convenience factory. Returns a GoldAuthModel for baseline architectures,
    or a raw GoldFormer module for arch == "goldformer" — GoldFormer's
    forward() returns a (logits, gamma) tuple rather than bare logits, so it
    isn't wrapped in GoldAuthModel's uniform interface.
    """
    if arch == "goldformer":
        return GoldFormer(ablation="full")
    return GoldAuthModel(arch=arch, pretrained=pretrained)


ALL_DEEP_MODELS = [
    "resnet18", "resnet50", "vgg16",
    "densenet121", "efficientnet_b0", "mobilenet_v2", "vit_b_16",
    "iformer_s", "overlock_xt",
]

ALL_CLASSICAL_MODELS = ["lbp_svm", "haralick_svm"]
