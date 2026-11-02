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

    def __init__(self, theta: float, axes_dim: list[int], method: str = "yarn", dype: bool = True, base_patches: int = 64,
                 use_mscale: bool = True):
        super().__init__()
        if method not in ("base", "ntk", "yarn"):
            raise ValueError(f"method must be base|ntk|yarn, got {method!r}")
        self.theta = theta
        self.axes_dim = list(axes_dim)
        self.method = method
        self.dype = dype and method != "base"
        self.base_patches = base_patches
        self.use_mscale = use_mscale  # YaRN attention-temperature factor; False = diagnostic
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
                if not self.use_mscale:
                    mscale = 1.0
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

    def __init__(self, rope: nn.Module, method: str = "yarn", dype: bool = True, base_patches: int = 64, use_mscale: bool = True):
        if method not in ("base", "ntk", "yarn"):
            raise ValueError(f"method must be base|ntk|yarn, got {method!r}")
        self.use_mscale = use_mscale
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
                if not self.use_mscale:
                    mscale = 1.0
            elif self.method == "ntk" and scale > 1:
                freqs = ntk_freqs(dim, self.theta, scale, self.kappa, fdtype, device)
            else:
                index = ids[:, :, i : i + 1].repeat(1, 1, table.shape[-1]).to(torch.int64)
                out.append(torch.gather(table.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
                continue
            ang = pos.unsqueeze(-1) * freqs  # (B, S, dim/2)
            out.append(torch.polar(torch.full_like(ang, mscale), ang).to(table.dtype))
        return torch.cat(out, dim=-1)


class DyPEQwenRope(nn.Module):
    """DyPE for diffusers' ``QwenEmbedRope`` (Qwen-Image): drop-in for ``transformer.pos_embed``.

    Same ``forward(video_fhw, device, max_txt_seq_len) -> (vid_freqs, txt_freqs)`` as the stock module. The frame
    axis and the text frequencies keep the stock tables; the height and width axes get YaRN/NTK frequencies when the
    grid exceeds ``base_patches`` (per axis: scale = extent / base_patches), as complex exponentials with ``mscale``
    as the magnitude (as for Lumina). Positions follow the stock layout: centred and signed when ``scale_rope`` is
    set (``-(n - n//2) .. n//2 - 1``), else ``0 .. n-1``. Computed per call (no lru cache), so an anchor pass at a
    smaller grid gets its own, less extrapolated frequencies, and kappa(t) can change every step.

    Qwen's transformer receives ``timestep = sigma`` (1 = noise), so ``set_timestep`` takes it directly.
    """

    def __init__(self, rope: nn.Module, method: str = "yarn", dype: bool = True, base_patches: int = 83, use_mscale: bool = True):
        super().__init__()
        if method not in ("base", "ntk", "yarn"):
            raise ValueError(f"method must be base|ntk|yarn, got {method!r}")
        self.rope = rope
        self.theta = float(rope.theta)
        self.axes_dim = list(rope.axes_dim)
        self.scale_rope = bool(getattr(rope, "scale_rope", False))
        self.method = method
        self.dype = dype and method != "base"
        self.use_mscale = use_mscale
        self.base_patches = base_patches
        self.t = 1.0

    def set_timestep(self, t: float) -> None:
        self.t = float(t)

    @property
    def kappa(self) -> float:
        return self.t**2 if self.dype else 1.0

    def _axis(self, n: int, axis: int, device, cdtype) -> torch.Tensor:
        """(n, dim/2) complex frequencies for one spatial axis of extent n."""
        pos_freqs, neg_freqs = self.rope._get_device_freqs(device) if device is not None else (self.rope.pos_freqs, self.rope.neg_freqs)
        dims = [d // 2 for d in self.axes_dim]
        fp, fn = pos_freqs.split(dims, dim=1)[axis], neg_freqs.split(dims, dim=1)[axis]
        scale = n / self.base_patches
        if scale <= 1 or self.method == "base":
            return torch.cat([fn[-(n - n // 2):], fp[: n // 2]], dim=0) if self.scale_rope else fp[:n]
        fdtype = torch.float64 if device is None or device.type != "mps" else torch.float32
        dim = self.axes_dim[axis]
        if self.method == "yarn":
            freqs, mscale = yarn_freqs(dim, self.theta, scale, self.kappa, self.base_patches, fdtype, device)
            if not self.use_mscale:
                mscale = 1.0
        else:
            freqs, mscale = ntk_freqs(dim, self.theta, scale, self.kappa, fdtype, device), 1.0
        if self.scale_rope:
            pos = torch.arange(-(n - n // 2), n // 2, dtype=fdtype, device=device)
        else:
            pos = torch.arange(n, dtype=fdtype, device=device)
        ang = torch.outer(pos, freqs)
        return torch.polar(torch.full_like(ang, mscale), ang).to(cdtype)

    def _video_freqs(self, frame: int, height: int, width: int, idx: int, device) -> torch.Tensor:
        pos_freqs, _ = self.rope._get_device_freqs(device) if device is not None else (self.rope.pos_freqs, None)
        fp0 = pos_freqs.split([d // 2 for d in self.axes_dim], dim=1)[0]
        f_frame = fp0[idx: idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        f_h = self._axis(height, 1, device, pos_freqs.dtype).view(1, height, 1, -1).expand(frame, height, width, -1)
        f_w = self._axis(width, 2, device, pos_freqs.dtype).view(1, 1, width, -1).expand(frame, height, width, -1)
        return torch.cat([f_frame, f_h, f_w], dim=-1).reshape(frame * height * width, -1).contiguous()

    def forward(self, video_fhw, device=None, max_txt_seq_len=None):
        if max_txt_seq_len is None:
            raise ValueError("`max_txt_seq_len` must be provided.")
        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]
        vid, max_vid_index = [], 0
        for idx, (frame, height, width) in enumerate(video_fhw):
            vid.append(self._video_freqs(frame, height, width, idx, device))
            max_vid_index = max(height // 2, width // 2, max_vid_index) if self.scale_rope else max(height, width, max_vid_index)
        pos_freqs, _ = self.rope._get_device_freqs(device) if device is not None else (self.rope.pos_freqs, None)
        txt = pos_freqs[max_vid_index: max_vid_index + int(max_txt_seq_len), ...]
        return torch.cat(vid, dim=0), txt


def apply_dype(transformer: nn.Module, method: str = "yarn", dype: bool = True, base_patches: int | None = None, use_mscale: bool = True):
    """Enable DyPE on a FLUX (``pos_embed``), Lumina-2 (``rope_embedder``) or Qwen-Image (``QwenEmbedRope``) transformer.

    ``base_patches`` (the native extent in patches) defaults per model: 64 (1024 px / 16) for FLUX and Lumina, 83
    (1328 px / 16, Qwen-Image's training resolution) for Qwen.

    Wraps ``forward`` so the per-step timestep reaches the position embedding: FLUX passes
    ``timestep`` in (0, 1] with 1 = pure noise; Lumina passes ``1 - t/1000`` (1 = clean) and is
    inverted here. Idempotent: calling again just reconfigures. Returns the DyPE object.
    """
    pe_name = type(getattr(transformer, "pos_embed", None)).__name__
    if pe_name in ("QwenEmbedRope", "DyPEQwenRope"):  # Qwen-Image
        if not hasattr(transformer, "_dype_orig"):
            transformer._dype_orig = ("qwen", transformer.pos_embed, transformer.forward)
        stock = transformer._dype_orig[1]
        pe = DyPEQwenRope(stock, method=method, dype=dype, base_patches=base_patches or 83, use_mscale=use_mscale)
        transformer.pos_embed = pe
        to_noise_level = lambda ts: float(ts.reshape(-1)[0])  # noqa: E731  (sigma, 1 = noise)
    elif pe_name in ("FluxPosEmbed", "DyPEFluxPosEmbed"):
        if not hasattr(transformer, "_dype_orig"):
            transformer._dype_orig = ("flux", transformer.pos_embed, transformer.forward)
        stock = transformer._dype_orig[1]
        pe = DyPEFluxPosEmbed(stock.theta, stock.axes_dim, method=method, dype=dype, base_patches=base_patches or 64, use_mscale=use_mscale)
        transformer.pos_embed = pe
        to_noise_level = lambda ts: float(ts.reshape(-1)[0])  # noqa: E731
    elif hasattr(transformer, "rope_embedder"):  # Lumina-Image 2.0
        if not hasattr(transformer, "_dype_orig"):
            transformer._dype_orig = ("lumina", transformer.rope_embedder._get_freqs_cis, transformer.forward)
        pe = DyPELuminaRope(transformer.rope_embedder, method=method, dype=dype, base_patches=base_patches or 64, use_mscale=use_mscale)
        pe._stock_get = transformer._dype_orig[1]
        transformer.rope_embedder._get_freqs_cis = pe
        to_noise_level = lambda ts: 1.0 - float(ts.reshape(-1)[0])  # noqa: E731
    else:
        raise TypeError(f"{type(transformer).__name__}: DyPE is implemented for FLUX, Lumina2 and Qwen-Image transformers only")

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
    if kind in ("flux", "qwen"):
        transformer.pos_embed = stock
    else:
        transformer.rope_embedder._get_freqs_cis = stock
    transformer.forward = orig_forward
    del transformer._dype_orig
