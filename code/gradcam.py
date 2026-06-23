"""
gradcam.py
----------
Gold Jewelry Authentication — Grad-CAM interpretability.

Generates Gradient-weighted Class Activation Maps for any
GoldAuthModel architecture, used to validate that learned
representations reflect genuine surface texture characteristics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Target layer registry (last conv layer per architecture)
# ─────────────────────────────────────────────────────────────────────────────

TARGET_LAYERS = {
    "resnet18":        "backbone.layer4.1.conv2",
    "resnet50":        "backbone.layer4.2.conv3",
    "vgg16":           "backbone.features.28",
    "densenet121":     "backbone.features.denseblock4.denselayer16.conv2",
    "efficientnet_b0": "backbone.features.8.0",
    "mobilenet_v2":    "backbone.features.18.conv.0.0",
    "vit_b_16":        "backbone.encoder.layers.encoder_layer_11.ln_1",
}


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Usage
    -----
    gcam = GradCAM(model, arch)
    heatmap = gcam(image_tensor, target_class)
    overlay = gcam.overlay(original_pil_image, heatmap)
    """

    def __init__(self, model: nn.Module, arch: str):
        self.model  = model
        self.arch   = arch
        self._hooks: list = []
        self._grads: Optional[torch.Tensor] = None
        self._acts:  Optional[torch.Tensor] = None

        self._register_hooks(arch)

    def _register_hooks(self, arch: str) -> None:
        layer_name = TARGET_LAYERS.get(arch)
        if layer_name is None:
            raise ValueError(f"No target layer defined for arch='{arch}'")

        # resolve layer by dotted name
        layer = self._get_layer(layer_name)

        def forward_hook(module, input, output):
            self._acts = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._grads = grad_out[0].detach()

        self._hooks.append(layer.register_forward_hook(forward_hook))
        self._hooks.append(layer.register_full_backward_hook(backward_hook))

    def _get_layer(self, dotted_name: str) -> nn.Module:
        parts  = dotted_name.split(".")
        module = self.model
        for p in parts:
            module = getattr(module, p)
        return module

    def __call__(
        self,
        image:        torch.Tensor,
        target_class: Optional[int] = None,
        device:       str = "cpu",
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Parameters
        ----------
        image        : (1, 3, H, W) preprocessed tensor
        target_class : class index to visualise (None → argmax)
        device       : 'cpu' or 'cuda'

        Returns
        -------
        heatmap : (H, W) float32 array in [0, 1]
        """
        self.model.eval()
        image = image.to(device)

        # forward
        self.model.zero_grad()
        logits = self.model(image)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # backward from target class score
        score = logits[0, target_class]
        score.backward()

        # Grad-CAM: global average pool gradients over spatial dims
        grads = self._grads   # (1, C, H', W') or (1, L, d) for ViT
        acts  = self._acts

        if grads is None or acts is None:
            raise RuntimeError("Hooks did not capture gradients/activations. "
                               "Ensure model performed a forward+backward pass.")

        if grads.dim() == 4:
            # CNN: average over H', W'
            weights  = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
            cam      = (weights * acts).sum(dim=1).squeeze()   # (H', W')
        else:
            # ViT: spatial tokens (skip CLS token at index 0)
            weights  = grads[:, 1:, :].mean(dim=2)             # (1, L)
            cam      = (weights.unsqueeze(-1) * acts[:, 1:, :]).sum(dim=2)
            # reshape to grid
            n_patch = cam.shape[1]
            side    = int(n_patch ** 0.5)
            cam     = cam.squeeze().reshape(side, side)

        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)           # ReLU
        if cam.max() > 0:
            cam = cam / cam.max()          # normalise to [0, 1]

        # upsample to input resolution
        h, w = image.shape[2], image.shape[3]
        cam  = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

        return cam.astype(np.float32)

    @staticmethod
    def overlay(
        pil_image,
        heatmap:   np.ndarray,
        alpha:     float = 0.45,
        colormap:  int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Blend original image with Grad-CAM heatmap.

        Returns
        -------
        blended : (H, W, 3) uint8 RGB array
        """
        img_np  = np.array(pil_image.convert("RGB"))
        heat_u8 = (heatmap * 255).astype(np.uint8)
        heat_c  = cv2.applyColorMap(heat_u8, colormap)
        heat_c  = cv2.cvtColor(heat_c, cv2.COLOR_BGR2RGB)
        blended = (alpha * heat_c + (1 - alpha) * img_np).astype(np.uint8)
        return blended

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()
