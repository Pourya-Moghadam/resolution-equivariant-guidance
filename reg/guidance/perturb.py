"""Perturbation guidance for Lumina-Image 2.0: PAG, SEG and TPG as ``PertTerm`` subclasses.

All three run a second forward pass through a perturbed network and guide away from it:
v = v_cond + s·(v_cond − v_pert). They differ in the perturbation of the chosen transformer
blocks (``layers``, indices into ``transformer.layers``; Lumina has 32):

- PAG (Ahn et al. 2024): self-attention map replaced by the identity, so each token attends to
  itself: out = W_o · V.
- SEG (Hong 2024): attention logits blurred with a 2D Gaussian over the image token grid,
  implemented on the queries (blur(Q)·Kᵀ, Prop. 3.1 of the paper); sigma in latent patches
  (∞ = mean query). Scale 3.0 and sigma ≥ 10 in the paper.
- TPG (arXiv 2506.10036): image tokens permuted before the block and un-permuted after, a
  fresh permutation per layer and step; scale 3.0 in the paper.

Lumina's blocks see a joint sequence [text tokens, image tokens]; the perturbations act on the
image part only (SEG, TPG) or on all tokens (PAG; text tokens attending to themselves is harmless).
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb

from .bands import gaussian_blur
from .base import PertTerm


def _image_grid_view(x: torch.Tensor, text_len: int, grid: tuple[int, int]) -> torch.Tensor:
    """(B, L, ...) -> the image tokens as (B, h, w, ...)."""
    h, w = grid
    return x[:, text_len : text_len + h * w].reshape(x.shape[0], h, w, *x.shape[2:])


class PerturbedAttnProcessor:
    """Drop-in for Lumina2AttnProcessor2_0 with mode 'identity' (PAG) or 'blur' (SEG)."""

    def __init__(self, mode: str, sigma: float = 10.0, text_len: int = 0, grid: tuple[int, int] = (0, 0)):
        if mode not in ("identity", "blur"):
            raise ValueError(mode)
        self.mode, self.sigma, self.text_len, self.grid = mode, sigma, text_len, grid

    def __call__(self, attn, hidden_states, encoder_hidden_states, attention_mask=None, image_rotary_emb=None, base_sequence_length=None):
        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim
        dtype = query.dtype
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)
        n_rep = attn.heads // kv_heads

        if self.mode == "identity":
            # attention map = I: every token gets its own value (GQA heads repeated as in the original)
            out = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3) if n_rep > 1 else value
            out = out.reshape(batch_size, -1, attn.heads * head_dim).type_as(query)
            return attn.to_out[1](attn.to_out[0](out))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=False)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=False)
        query, key = query.to(dtype), key.to(dtype)
        # SEG: blur the (RoPE'd) queries of the image tokens over the 2D patch grid
        h, w = self.grid
        q_img = _image_grid_view(query, self.text_len, self.grid)  # (B, h, w, heads, d)
        q_img = q_img.permute(0, 3, 4, 1, 2).reshape(batch_size, attn.heads * head_dim, h, w)
        if math.isinf(self.sigma):
            q_img = q_img.mean(dim=(-2, -1), keepdim=True).expand_as(q_img)
        else:
            q_img = gaussian_blur(q_img.float(), self.sigma).to(dtype)
        q_img = q_img.reshape(batch_size, attn.heads, head_dim, h, w).permute(0, 3, 4, 1, 2).reshape(batch_size, h * w, attn.heads, head_dim)
        query = torch.cat([query[:, : self.text_len], q_img, query[:, self.text_len + h * w :]], dim=1)

        softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale if base_sequence_length is not None else attn.scale
        if n_rep >= 1:
            key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        if attention_mask is not None:
            attention_mask = attention_mask.bool().view(batch_size, 1, 1, -1)
        out = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=attention_mask, scale=softmax_scale)
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).type_as(query)
        return attn.to_out[1](attn.to_out[0](out))


def default_layers(transformer) -> list[int]:
    """Three middle blocks (Lumina: 15..17 of 32). Calibrated at 1K (jobs/pert_calib_1k.jsonl, 08): eight
    blocks desaturate and flatten at any scale; three blocks at scale 1.5-2 give a visible, clean effect;
    one block needs scale >= 3 to differ from plain CFG."""
    n = len(transformer.layers)
    return list(range(n // 2 - 1, n // 2 + 2))


class AttnPertTerm(PertTerm):
    """Shared machinery for PAG and SEG: swap the attention processors of the chosen blocks."""

    mode = "identity"

    def __init__(self, scale: float, layers: list[int] | None = None, sigma: float = 5.0, start: float = 0.0, end: float = 1.0, **kw):
        super().__init__(scale, start, end, **kw)
        self.layers = layers
        self.sigma = float(sigma)

    @contextlib.contextmanager
    def perturbation(self, transformer, ctx: dict):
        layers = self.layers if self.layers is not None else default_layers(transformer)
        proc = PerturbedAttnProcessor(self.mode, self.sigma, ctx["text_len"], ctx["grid"])
        saved = {k: transformer.layers[k].attn.processor for k in layers}
        try:
            for k in layers:
                transformer.layers[k].attn.set_processor(proc)
            yield
        finally:
            for k, p in saved.items():
                transformer.layers[k].attn.set_processor(p)

    def describe(self) -> dict:
        return {**super().describe(), "layers": self.layers, **({"sigma": self.sigma} if self.mode == "blur" else {})}


class PAG(AttnPertTerm):
    name, mode = "pag", "identity"


class SEG(AttnPertTerm):
    name, mode = "seg", "blur"


class TPG(PertTerm):
    name = "tpg"

    def __init__(self, scale: float, layers: list[int] | None = None, frac: float = 0.5, start: float = 0.0, end: float = 1.0, **kw):
        """``frac``: fraction of image tokens that take part in the permutation (the paper shuffles
        'a small fraction' for DiTs without giving it; 1.0 = all image tokens)."""
        super().__init__(scale, start, end, **kw)
        self.layers = layers
        self.frac = float(frac)

    @contextlib.contextmanager
    def perturbation(self, transformer, ctx: dict):
        layers = self.layers if self.layers is not None else default_layers(transformer)
        text_len, (h, w) = ctx["text_len"], ctx["grid"]
        gen: torch.Generator = ctx["generator"]
        n_img = h * w
        state = {}

        def pre(k):
            def hook(module, args, kwargs):
                hs = args[0] if args else kwargs["hidden_states"]
                n_sel = max(2, int(round(self.frac * n_img)))
                sel = torch.randperm(n_img, generator=gen, device=gen.device)[:n_sel].to(hs.device) + text_len
                perm = sel[torch.randperm(n_sel, generator=gen, device=gen.device).to(hs.device)]
                state[k] = (sel, perm)
                hs = hs.clone()
                hs[:, sel] = hs[:, perm]
                if args:
                    return (hs, *args[1:]), kwargs
                kwargs = {**kwargs, "hidden_states": hs}
                return args, kwargs

            return hook

        def post(k):
            def hook(module, args, kwargs, out):
                sel, perm = state.pop(k)
                res = out.clone()
                res[:, perm] = out[:, sel]  # un-permute: token that was moved to position sel[j] goes back to perm[j]
                return res

            return hook

        handles = []
        try:
            for k in layers:
                handles.append(transformer.layers[k].register_forward_pre_hook(pre(k), with_kwargs=True))
                handles.append(transformer.layers[k].register_forward_hook(post(k), with_kwargs=True))
            yield
        finally:
            for hd in handles:
                hd.remove()

    def describe(self) -> dict:
        return {**super().describe(), "layers": self.layers, "frac": self.frac}
