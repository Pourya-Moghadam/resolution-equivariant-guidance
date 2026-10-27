"""Frequency band splits of a latent-space prediction (component A of REG, and FDG's transform).

A split maps a tensor x of shape (B, C, H, W) to bands whose sum is exactly x, so applying a
different gain per band and summing reconstructs CFG when the gains are equal.

``LaplacianSplit`` is FDG's single-level Laplacian pyramid: low = up(down(blur(x))), high = x - low.
The cutoff is set by ``down``, the decimation factor on the latent grid. FDG fixes it on the grid
(down = 2); REG-A scales it with the resolution ratio so the cutoff stays put in cycles per image.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).to(dtype)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur over the last two dims with reflect padding; x is (B, C, H, W)."""
    if sigma <= 0:
        return x
    k = gaussian_kernel1d(sigma, x.device, x.dtype)
    r = (len(k) - 1) // 2
    c = x.shape[1]
    kh = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    kv = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    # reflect padding needs the pad to be smaller than the dim; fall back to replicate for tiny maps
    mode = "reflect" if r < min(x.shape[-2:]) else "replicate"
    y = F.pad(x, (r, r, 0, 0), mode=mode)
    y = F.conv2d(y, kh, groups=c)
    y = F.pad(y, (0, 0, r, r), mode=mode)
    return F.conv2d(y, kv, groups=c)


class LaplacianSplit:
    """Two bands: low-pass (blur, decimate by ``down``, bilinear upsample) and the residual."""

    names = ("low", "high")

    def __init__(self, down: float = 2.0, sigma: float | None = None):
        if down < 1:
            raise ValueError("down must be >= 1")
        self.down = float(down)
        self.sigma = 0.5 * self.down if sigma is None else sigma  # anti-alias for decimation by `down`

    def lowpass(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if self.down == 1.0:
            return gaussian_blur(x, self.sigma)
        small = (max(1, round(h / self.down)), max(1, round(w / self.down)))
        y = gaussian_blur(x.float(), self.sigma)
        y = F.interpolate(y, size=small, mode="area")
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        return y.to(x.dtype)

    def split(self, x: torch.Tensor) -> list[torch.Tensor]:
        low = self.lowpass(x)
        return [low, x - low]

    def __repr__(self) -> str:
        return f"LaplacianSplit(down={self.down:g}, sigma={self.sigma:g})"
