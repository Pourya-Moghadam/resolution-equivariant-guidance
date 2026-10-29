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

  ``base=True`` (variant tested 2026-10-28 after the Δ-only anchor failed at 4K, 08) also replaces
  the coarse band of the *base* prediction u_uncond by the native one. With u = (x0 − x_σ)/σ and
  x̂0′ = x′ + σ′·u′ the native denoised estimate, the coarse band of the base becomes
         low(u_uncond) ← (up(x̂0′_uncond) − up(down(x_σ)))/σ = up((1−c)·x′ + σ′·u′_uncond)/σ,
  so the guided prediction's whole sub-native band comes from where the model is native, and the
  weight w only amplifies the native difference. Guidance cannot remove an error the two branches
  share; this variant removes it from the base instead.

  **Pyramid (2026-10-29).** ``levels`` adds intermediate anchor ratios, e.g. ratio 4 with levels (2,) at 4K:
  the band below the 1K Nyquist comes from the 1K pass, the band between the 1K and 2K Nyquists from a
  2K pass (where the model is only 2× out of range instead of 4×), and only the band above the 2K Nyquist
  from the 4K pass — each pass at its own derived noise level. Base and Δ are both assembled this way
  when ``base=True``. Upsample Guidance is the single-level case. Cost: Σ 1/r² extra per pass.

  **Super-native weight.** ``super_scale`` is the CFG weight applied to Δ's bands above the native
  Nyquist (the frequencies the model never saw at training); default = ``scale``. ``super_scale = 1``
  switches guidance off there (a test of whether the 4K speckle is guidance-amplified noise).

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

    def __init__(self, scale: float, ratio: float, start: float = 0.0, end: float = 1.0, base: bool = False,
                 levels=(), super_scale: float | None = None):
        super().__init__(scale, start, end)
        self.anchor_base = bool(base)
        if ratio < 1 or abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(f"anchor needs an integer resolution ratio >= 1, got {ratio}")
        self.ratio = int(round(ratio))
        extra = tuple(int(round(l)) for l in levels)
        if any(l <= 1 or l >= self.ratio or abs(l - round(l)) > 1e-6 for l in levels):
            raise ValueError(f"pyramid levels must be integers in (1, ratio={self.ratio}), got {levels}")
        self.levels = (self.ratio,) + tuple(sorted(set(extra), reverse=True))  # coarsest first
        self.super_scale = self.scale if super_scale is None else float(super_scale)
        self.split = None  # bands handled here, not by Term.contribution
        self.band_scales = (self.scale,)
        self._low = LaplacianSplit(down=float(self.ratio)) if self.ratio > 1 else None
        self._lows = {r: LaplacianSplit(down=float(r)) for r in self.levels if r > 1}

    def key(self, ratio: int, name: str) -> str:
        """preds key of a level's pass: 'anchor_<name>' for the coarsest (native) level, 'anchor<r>_<name>' otherwise."""
        return f"anchor_{name}" if ratio == self.ratio else f"anchor{ratio}_{name}"

    def _bands(self, full: dict[int, torch.Tensor], top: torch.Tensor, gains: dict[int, float] | None = None, top_gain: float = 1.0) -> torch.Tensor:
        """Assemble a full-resolution tensor from per-level estimates: the band below the coarsest Nyquist from the
        coarsest level, each band between two Nyquists from the finer of the two levels, the band above the
        finest anchor level's Nyquist from ``top`` (the full-resolution pass). Optional per-band gains."""
        gains = gains or {}
        lv = self.levels
        # an upsampled level-r estimate is already band-limited to its own Nyquist, so it is used as is and
        # only the coarser Nyquist is cut from it; the single-level case is then the original formula exactly
        out = gains.get(lv[0], 1.0) * full[lv[0]]
        for coarse, fine in zip(lv[:-1], lv[1:]):
            y = full[fine]
            out = out + gains.get(fine, 1.0) * (y - self._lows[coarse].lowpass(y))
        return out + top_gain * (top - self._lows[lv[-1]].lowpass(top))

    def active(self, step: Step) -> bool:
        return self.start < step.frac <= self.end and self.scale != 1.0

    def anchor_input(self, latents: torch.Tensor, sigma: float, ratio: int | None = None) -> tuple[torch.Tensor, float]:
        """Anchor-level latent x′ (downsampled by ``ratio``, default the native ratio) and its noise level σ′."""
        ratio = self.ratio if ratio is None else int(ratio)
        sigma_a, c = anchor_level(sigma, ratio)
        return downsample(latents, ratio) / c, sigma_a

    def base(self, preds: dict[str, torch.Tensor], step: Step, v_base: torch.Tensor) -> torch.Tensor:
        """With ``base=True``: the base prediction with each sub-native band taken from the coarsest pass that
        represents it (single level: the sub-native band from the native pass)."""
        if not self.anchor_base or self.ratio == 1:
            return v_base
        size = v_base.shape[-2:]
        full = {}
        for r in self.levels:
            sigma_a, c = anchor_level(step.sigma, r)
            x_a = preds[self.key(r, "x")]  # the level's input x′ (sampler-provided)
            full[r] = upsample((1.0 - c) * x_a + sigma_a * preds[self.key(r, "uncond")], size) / step.sigma
        return self._bands(full, v_base)

    def contribution(self, preds: dict[str, torch.Tensor], step: Step) -> torch.Tensor:
        delta = preds["cond"] - preds["uncond"]
        if self.ratio == 1:
            return self.scale * delta
        size = delta.shape[-2:]
        full, gains = {}, {}
        for r in self.levels:
            sigma_a, _ = anchor_level(step.sigma, r)
            full[r] = upsample(preds[self.key(r, "cond")] - preds[self.key(r, "uncond")], size) * (sigma_a / step.sigma)
            gains[r] = self.scale if r == self.ratio else self.super_scale  # bands above the native Nyquist get super_scale
        return self._bands(full, delta, gains, top_gain=self.super_scale)

    def describe(self) -> dict:
        d = super().describe()
        d["ratio"] = self.ratio
        d["base"] = self.anchor_base
        if len(self.levels) > 1:
            d["levels"] = self.levels[1:]
        if self.super_scale != self.scale:
            d["super_scale"] = self.super_scale
        return d
