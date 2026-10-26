"""DyPE (Dynamic Position Extrapolation, Issachar et al., ICML 2026) for FLUX-style 2D RoPE.

Our own implementation of the method, written from the authors' MIT-licensed reference
(see 07-dype-reference.md for the constants and where they come from). It is a drop-in for
diffusers' ``FluxPosEmbed``: at native resolution (<= 64x64 patches) it reproduces the stock
embedding exactly (tested), above it the two spatial axes get YaRN or NTK frequency scaling
whose correction thresholds are multiplied by kappa(t) = t^2, t in (0, 1], 1 = pure noise.

Usage with the stock pipeline (schedule convention: pass shift_mu=1.15, see Flux.generate):

    apply_dype(pipe.transformer, method="yarn", dype=True)
    ...
    remove_dype(pipe.transformer)
"""

from __future__ import annotations

import math

import torch
from torch import nn

# YaRN correction ranges in "number of rotations over the native extent" (DyPE reference values)
BETA_0, BETA_1 = 1.25, 0.75  # linear <-> NTK blend
GAMMA_0, GAMMA_1 = 16.0, 2.0  # scaled <-> original-frequency blend (keeps high-frequency dims native)
KAPPA_MIN = 1e-6


def _correction_factor(num_rotations: float, dim: int, base: float, max_pe_len: int) -> float:
    # inverse of the RoPE dim formula: which dim index completes `num_rotations` over max_pe_len
    return (dim * math.log(max_pe_len / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _correction_range(low_rot: float, high_rot: float, dim: int, base: float, max_pe_len: int) -> tuple[int, int]:
    low = math.floor(_correction_factor(low_rot, dim, base, max_pe_len))
    high = math.ceil(_correction_factor(high_rot, dim, base, max_pe_len))
    low, high = max(low, 0), min(high, dim - 1)
    return max(0, low), min(dim // 2, high)


def _ramp(low: int, high: int, n: int, dtype: torch.dtype, device) -> torch.Tensor:
    if low == high:
        high = high + 0.001  # avoid the singularity, as the reference does
    return torch.clamp((torch.arange(n, dtype=dtype, device=device) - low) / (high - low), 0, 1)


def base_freqs(dim: int, theta: float, dtype: torch.dtype, device) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device) / dim))


def yarn_freqs(dim: int, theta: float, scale: float, kappa: float, base_patches: int, dtype: torch.dtype, device):
    """Per-dim angular frequencies (dim/2,) and the attention-temperature factor mscale."""
    kappa = max(kappa, KAPPA_MIN)
    idx = torch.arange(0, dim, 2, dtype=dtype, device=device) / dim
    f_base = 1.0 / (theta**idx)
    f_linear = f_base / scale
    new_base = theta * scale ** (dim / (dim - 2))
    f_ntk = 1.0 / (new_base**idx)

    low, high = _correction_range(BETA_0 * kappa, BETA_1 * kappa, dim, theta, base_patches)
    mask = 1 - _ramp(low, high, dim // 2, dtype, device)
    freqs = f_linear * (1 - mask) + f_ntk * mask

    low, high = _correction_range(GAMMA_0 * kappa, GAMMA_1 * kappa, dim, theta, base_patches)
    mask = 1 - _ramp(low, high, dim // 2, dtype, device)
    freqs = freqs * (1 - mask) + f_base * mask

    mscale = 1.0 if scale <= 1 else 0.1 * math.log(scale) + 1.0
    return freqs, mscale


def ntk_freqs(dim: int, theta: float, scale: float, kappa: float, dtype: torch.dtype, device) -> torch.Tensor:
    kappa = max(kappa, KAPPA_MIN)
    ntk_factor = max(1.0, (scale ** (dim / (dim - 2))) ** (2.0 * kappa))
    return base_freqs(dim, theta * ntk_factor, dtype, device)


class DyPEFluxPosEmbed(nn.Module):
    """Drop-in for diffusers.models.transformers.transformer_flux.FluxPosEmbed.

    ``ids`` is (S, 3): axis 0 is the text/time axis (left untouched), axes 1 and 2 are patch
    rows and columns. Scaling kicks in only when an axis extends past ``base_patches`` (64 =
    1024 px / 16 for FLUX).
    """

    def __init__(self, theta: float, axes_dim: list[int], method: str = "yarn", dype: bool = True, base_patches: int = 64):
        super().__init__()
        if method not in ("base", "ntk", "yarn"):
            raise ValueError(f"method must be base|ntk|yarn, got {method!r}")
        self.theta = theta
        self.axes_dim = list(axes_dim)
        self.method = method
        self.dype = dype and method != "base"
        self.base_patches = base_patches
        self.t = 1.0  # normalized noise level of the current step; 1 = pure noise

    def set_timestep(self, t: float) -> None:
        self.t = float(t)

    @property
    def kappa(self) -> float:
        return self.t**2 if self.dype else 1.0

    def forward(self, ids: torch.Tensor):
        pos = ids.float()
        device = ids.device
        fdtype = torch.float32 if device.type in ("mps", "npu") else torch.float64
        cos_out, sin_out = [], []
        for i, dim in enumerate(self.axes_dim):
            p = pos[:, i].to(fdtype)
            scale = (p.max().item() + 1) / self.base_patches if i > 0 else 1.0
            mscale = 1.0
            if self.method == "yarn" and scale > 1:
                freqs, mscale = yarn_freqs(dim, self.theta, scale, self.kappa, self.base_patches, fdtype, device)
            elif self.method == "ntk" and scale > 1:
                freqs = ntk_freqs(dim, self.theta, scale, self.kappa, fdtype, device)
            else:
                freqs = base_freqs(dim, self.theta, fdtype, device)
            ang = torch.outer(p, freqs)  # (S, dim/2)
            cos_out.append(ang.cos().repeat_interleave(2, dim=1).float() * mscale)
            sin_out.append(ang.sin().repeat_interleave(2, dim=1).float() * mscale)
        return torch.cat(cos_out, dim=-1), torch.cat(sin_out, dim=-1)


class DyPELuminaRope:
    """DyPE for diffusers' ``Lumina2RotaryPosEmbed`` (complex ``freqs_cis``, gathered by position).

    Replaces the module's ``_get_freqs_cis(ids)`` (ids: (B, S, 3) int; axis 0 = caption/time,
    axes 1, 2 = patch rows/cols). Axis 0 keeps the stock gather; the spatial axes get the same
    YaRN/NTK frequencies as the FLUX version, as complex exponentials with ``mscale`` as the
    magnitude (q and k are both multiplied, so attention logits scale by mscale^2, exactly as
    scaling cos and sin does for FLUX).

    Lumina's transformer receives ``timestep = 1 - t/1000`` (1 = clean image), so the noise
    level DyPE wants is ``1 - timestep``; ``set_timestep`` takes the noise level directly.
    """

    def __init__(self, rope: nn.Module, method: str = "yarn", dype: bool = True, base_patches: int = 64):
        if method not in ("base", "ntk", "yarn"):
            raise ValueError(f"method must be base|ntk|yarn, got {method!r}")
        self.rope = rope
        self.theta = float(rope.theta)
        self.axes_dim = list(rope.axes_dim)
        self.method = method
        self.dype = dype and method != "base"
        self.base_patches = base_patches
        self.t = 1.0
        self._stock_get = rope._get_freqs_cis

    def set_timestep(self, t: float) -> None:
        self.t = float(t)

    @property
    def kappa(self) -> float:
        return self.t**2 if self.dype else 1.0

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        fdtype = torch.float64 if device.type != "mps" else torch.float32
        out = []
        for i, dim in enumerate(self.axes_dim):
            table = self.rope.freqs_cis[i].to(device)  # (axes_len, dim/2) complex
            if i == 0:
                index = ids[:, :, i : i + 1].repeat(1, 1, table.shape[-1]).to(torch.int64)
                out.append(torch.gather(table.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
                continue
            pos = ids[:, :, i].to(fdtype)  # (B, S)
            scale = (pos.max().item() + 1) / self.base_patches
            mscale = 1.0
            if self.method == "yarn" and scale > 1:
                freqs, mscale = yarn_freqs(dim, self.theta, scale, self.kappa, self.base_patches, fdtype, device)
            elif self.method == "ntk" and scale > 1:
                freqs = ntk_freqs(dim, self.theta, scale, self.kappa, fdtype, device)
            else:
                index = ids[:, :, i : i + 1].repeat(1, 1, table.shape[-1]).to(torch.int64)
                out.append(torch.gather(table.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
                continue
            ang = pos.unsqueeze(-1) * freqs  # (B, S, dim/2)
            out.append(torch.polar(torch.full_like(ang, mscale), ang).to(table.dtype))
        return torch.cat(out, dim=-1)


def apply_dype(transformer: nn.Module, method: str = "yarn", dype: bool = True, base_patches: int = 64):
    """Enable DyPE on a FLUX (``pos_embed``) or Lumina-2 (``rope_embedder``) transformer.

    Wraps ``forward`` so the per-step timestep reaches the position embedding: FLUX passes
    ``timestep`` in (0, 1] with 1 = pure noise; Lumina passes ``1 - t/1000`` (1 = clean) and is
    inverted here. Idempotent: calling again just reconfigures. Returns the DyPE object.
    """
    if hasattr(transformer, "pos_embed") and type(getattr(transformer, "pos_embed")).__name__ in ("FluxPosEmbed", "DyPEFluxPosEmbed"):
        if not hasattr(transformer, "_dype_orig"):
            transformer._dype_orig = ("flux", transformer.pos_embed, transformer.forward)
        stock = transformer._dype_orig[1]
        pe = DyPEFluxPosEmbed(stock.theta, stock.axes_dim, method=method, dype=dype, base_patches=base_patches)
        transformer.pos_embed = pe
        to_noise_level = lambda ts: float(ts.reshape(-1)[0])  # noqa: E731
    elif hasattr(transformer, "rope_embedder"):  # Lumina-Image 2.0
        if not hasattr(transformer, "_dype_orig"):
            transformer._dype_orig = ("lumina", transformer.rope_embedder._get_freqs_cis, transformer.forward)
        pe = DyPELuminaRope(transformer.rope_embedder, method=method, dype=dype, base_patches=base_patches)
        pe._stock_get = transformer._dype_orig[1]
        transformer.rope_embedder._get_freqs_cis = pe
        to_noise_level = lambda ts: 1.0 - float(ts.reshape(-1)[0])  # noqa: E731
    else:  # e.g. Qwen-Image's QwenEmbedRope (centred pos/neg indices): not ported yet
        raise TypeError(f"{type(transformer).__name__}: DyPE is implemented for FLUX (FluxPosEmbed) and Lumina2 (rope_embedder) only")

    orig_forward = transformer._dype_orig[2]

    def forward(*args, timestep=None, **kwargs):
        if timestep is None and len(args) >= 2:  # Lumina's forward(hidden_states, timestep, ...) may be positional
            timestep = args[1]
        if timestep is not None:
            pe.set_timestep(to_noise_level(timestep))
        return orig_forward(*args, **({} if timestep is None or len(args) >= 2 else {"timestep": timestep}), **kwargs)

    transformer.forward = forward
    return pe


def remove_dype(transformer: nn.Module) -> None:
    if not hasattr(transformer, "_dype_orig"):
        return
    kind, stock, orig_forward = transformer._dype_orig
    if kind == "flux":
        transformer.pos_embed = stock
    else:
        transformer.rope_embedder._get_freqs_cis = stock
    transformer.forward = orig_forward
    del transformer._dype_orig
