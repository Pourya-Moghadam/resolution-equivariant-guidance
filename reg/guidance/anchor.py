"""Component (C) of REG: native-resolution anchored guidance.

At resolution ratio r the model is out of distribution, and the pilots (08) show that the coarse
band of its guidance signal Δ = v_cond − v_uncond is what degrades: the native weight collapses the
layout at 4K and twice the weight is needed, while Δ's relative size is unchanged (diagnostics).
The anchor evaluates that band where the model is native:

  1. area-downsample the current latent x_σ by r: down(x_σ) = (1−σ)·down(x0) + σ·ε̄, with ε̄ the
     mean of r² unit-normal noises, so its std is σ/r;
  2. rescale to a valid native sample at a derived, lower noise level:
         c = (1−σ) + σ/r,   x′ = down(x_σ)/c,   σ′ = (σ/r)/c
     (x′ = (1−σ′)·down(x0) + σ′·ε′ exactly, up to the downsample-the-latent approximation);
  3. run cond and uncond at native resolution on x′ at σ′ (1/r² of the tokens, so cheap), giving
     Δ′ = u′_cond − u′_uncond in the model's output units u = x0 − ε;
  4. since x̂0 = x_σ + σ·u, the denoised-image difference is Δ_x0 = σ·Δ, and at native σ′·Δ′.
     The native difference, upsampled, estimates the low band of Δ_x0 at full resolution, so
         Δ_low ≈ up(Δ′) · σ′/σ.
  5. guided prediction with the *native* weight w on every band:
         u = u_uncond + w · (Δ − low_r(Δ) + Δ_low),
     where low_r is the Laplacian low-pass at cutoff r (the band representable at native resolution).

No new hyperparameters: w is the native CFG scale, r the resolution ratio. At r = 1 the term is
plain CFG, bit for bit (the sampler reuses the main passes).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .bands import LaplacianSplit
from .base import Step, Term


def anchor_level(sigma: float, ratio: float) -> tuple[float, float]:
    """(σ′, c) for the downsampled latent at noise level ``sigma`` and resolution ratio ``ratio``."""
    c = (1.0 - sigma) + sigma / ratio
    return (sigma / ratio) / c, c


def downsample(x: torch.Tensor, ratio: int) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=ratio, stride=ratio) if ratio > 1 else x


def upsample(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x.float(), size=size, mode="bilinear", align_corners=False).to(x.dtype)


class AnchorCFGTerm(Term):
    """CFG whose band below the native Nyquist comes from the native-resolution anchor passes."""

    kind, name = "anchor", "anchor_cfg"

    def __init__(self, scale: float, ratio: float, start: float = 0.0, end: float = 1.0):
        super().__init__(scale, start, end)
        if ratio < 1 or abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(f"anchor needs an integer resolution ratio >= 1, got {ratio}")
        self.ratio = int(round(ratio))
        self.split = None  # bands handled here, not by Term.contribution
        self.band_scales = (self.scale,)
        self._low = LaplacianSplit(down=float(self.ratio)) if self.ratio > 1 else None

    def active(self, step: Step) -> bool:
        return self.start < step.frac <= self.end and self.scale != 1.0

    def anchor_input(self, latents: torch.Tensor, sigma: float) -> tuple[torch.Tensor, float]:
        """Native-resolution latent x′ and its noise level σ′ for the anchor passes."""
        sigma_a, c = anchor_level(sigma, self.ratio)
        return downsample(latents, self.ratio) / c, sigma_a

    def contribution(self, preds: dict[str, torch.Tensor], step: Step) -> torch.Tensor:
        delta = preds["cond"] - preds["uncond"]
        if self.ratio == 1:
            return self.scale * delta
        sigma_a, _ = anchor_level(step.sigma, self.ratio)
        delta_anchor = upsample(preds["anchor_cond"] - preds["anchor_uncond"], delta.shape[-2:]) * (sigma_a / step.sigma)
        return self.scale * (delta - self._low.lowpass(delta) + delta_anchor)

    def describe(self) -> dict:
        d = super().describe()
        d["ratio"] = self.ratio
        return d
