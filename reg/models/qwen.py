"""Qwen-Image (20B DiT, 2D RoPE, real CFG): the large model for the main results.

Memory on a 32 GB card: the bf16 transformer is 39 GB, so it is loaded into host RAM and
quantized to float8 weight-only layer by layer onto the GPU (~20 GB) with torchao. The
Qwen2.5-VL text encoder (16 GB) runs in a separate first phase that fills the embedding
cache and is freed before the transformer loads, as for FLUX.

Guidance is real CFG (``true_cfg_scale``, default 4.0; the pipeline has no embedded
guidance). The negative prompt the authors use is a single space.

Sampling runs in our own loop (``_sample``, a copy of the pipeline loop with the branch combination
delegated to ``reg.guidance.Guidance``: CFG, REG anchors; perturbation terms are not ported); ``use_pipeline``
runs the stock pipeline for the bit-identity check (``scripts/qwen_port_check.py``).

Schedule: dynamic shifting calibrated to 256..8192 tokens (max_shift 0.9, exponential), so
2K (16384 tokens) and 4K (65536) extrapolate mu; ``shift_mu`` overrides it.
"""

from __future__ import annotations

import contextlib
import gc
import time
from unittest import mock

import numpy as np
import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from diffusers.pipelines.qwenimage import pipeline_qwenimage

from ..guidance import Guidance, build_guidance
from ..guidance.base import Step
from .embed_cache import TextEmbedCache

MODEL_ID = "Qwen/Qwen-Image"
NAME = "qwen-image"
MAX_SEQ_LEN = 512  # the pipeline's __call__ default; encode_prompt truncates to it
NEGATIVE_DEFAULT = " "


def schedule_mu(pipe: QwenImagePipeline, height: int, width: int) -> float:
    c = pipe.scheduler.config
    return pipeline_qwenimage.calculate_shift(
        (height // 16) * (width // 16),
        c.get("base_image_seq_len", 256),
        c.get("max_image_seq_len", 8192),
        c.get("base_shift", 0.5),
        c.get("max_shift", 0.9),
    )


class Qwen:
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16, quant: str | None = "fp8"):
        if quant not in (None, "fp8"):
            raise ValueError(f"quant must be None or 'fp8', got {quant!r}")
        self.device = torch.device(device)
        self.dtype = dtype
        self.quant = quant
        self.cache = TextEmbedCache(NAME, max_sequence_length=MAX_SEQ_LEN)
        self._enc: QwenImagePipeline | None = None
        self._pipe: QwenImagePipeline | None = None

    # -- text ---------------------------------------------------------------------------
    def _encoder(self) -> QwenImagePipeline:
        if self._enc is None:
            self._enc = QwenImagePipeline.from_pretrained(MODEL_ID, transformer=None, vae=None, torch_dtype=self.dtype)
            self._enc.to(self.device)
        return self._enc

    def _encode_raw(self, text: str) -> dict[str, torch.Tensor]:
        """Unpadded (len, 3584) hidden states after the chat template is stripped (len <= 512)."""
        embeds, mask = self._encoder()._get_qwen_prompt_embeds([text], device=self.device)
        n = int(mask[0].sum().item())
        return {"embeds": embeds[0, :n][:MAX_SEQ_LEN]}

    def _batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        items = [self.cache.get(t, self._encode_raw, device=self.device)["embeds"] for t in texts]
        n = max(e.shape[0] for e in items)
        embeds = torch.stack([torch.cat([e, e.new_zeros(n - e.shape[0], e.shape[1])]) for e in items])
        mask = torch.stack([torch.cat([torch.ones(e.shape[0], dtype=torch.long, device=e.device),
                                       torch.zeros(n - e.shape[0], dtype=torch.long, device=e.device)]) for e in items])
        return embeds, mask

    def encode(self, prompts: list[str], negative_prompt: str | None = NEGATIVE_DEFAULT) -> dict[str, torch.Tensor]:
        embeds, mask = self._batch(prompts)
        out = {"prompt_embeds": embeds, "prompt_embeds_mask": mask}
        if negative_prompt is not None:
            n_embeds, n_mask = self._batch([negative_prompt or NEGATIVE_DEFAULT] * len(prompts))
            out["negative_prompt_embeds"] = n_embeds
            out["negative_prompt_embeds_mask"] = n_mask
        return out

    def free_text_encoders(self) -> None:
        if self._enc is not None:
            self._enc = None
            gc.collect()
            torch.cuda.empty_cache()

    # -- images -------------------------------------------------------------------------
    @property
    def pipe(self) -> QwenImagePipeline:
        if self._pipe is None:
            transformer = QwenImageTransformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=self.dtype)
            if self.quant == "fp8":
                from torchao.quantization import Float8WeightOnlyConfig, quantize_

                quantize_(transformer, Float8WeightOnlyConfig(), device=self.device)  # CPU bf16 -> GPU fp8, layer by layer
            transformer.to(self.device)
            gc.collect()
            self._pipe = QwenImagePipeline.from_pretrained(
                MODEL_ID, transformer=transformer, text_encoder=None, tokenizer=None, torch_dtype=self.dtype
            )
            self._pipe.to(self.device)
            self._pipe.vae.enable_tiling()
            self._pipe.set_progress_bar_config(disable=True)
            torch.cuda.empty_cache()
        return self._pipe

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        height: int = 1024,
        width: int = 1024,
        steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int = 0,
        negative_prompt: str = NEGATIVE_DEFAULT,
        shift_mu: float | None = None,
        output_type: str = "pil",
        guidance: Guidance | None = None,
        use_pipeline: bool = False,
    ):
        """Return (images, info). ``guidance_scale`` is the true CFG scale (<= 1 disables CFG); ``guidance`` (a
        ``reg.guidance.Guidance``, e.g. from ``build_guidance(row)``) replaces it. ``use_pipeline`` runs the stock
        pipeline (CFG only) for the bit-identity check."""
        if guidance is None:
            guidance = build_guidance({"guidance": guidance_scale})
        need_neg = use_pipeline and guidance_scale > 1 or any(t.kind in ("cfg", "anchor") for t in guidance.terms)
        embeds = self.encode(prompts, negative_prompt if need_neg else None)
        self.free_text_encoders()
        pipe = self.pipe
        generators = [torch.Generator(self.device).manual_seed(seed + i) for i in range(len(prompts))]
        mu_used = shift_mu if shift_mu is not None else schedule_mu(pipe, height, width)
        shift_ctx = (
            mock.patch.object(pipeline_qwenimage, "calculate_shift", lambda *a, **k: shift_mu)
            if shift_mu is not None and use_pipeline
            else contextlib.nullcontext()
        )
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        with shift_ctx:
            if use_pipeline:
                images = pipe(
                    prompt=None,
                    negative_prompt=None,
                    true_cfg_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    generator=generators,
                    max_sequence_length=MAX_SEQ_LEN,
                    output_type=output_type,
                    **embeds,
                ).images
            else:
                images = self._sample(embeds, height, width, steps, generators, guidance, seed, output_type, shift_mu)
        torch.cuda.synchronize(self.device)
        info = {
            "model": NAME,
            "quant": self.quant or "bf16",
            "true_cfg_scale": guidance_scale,
            "guidance": guidance.describe(),
            "shift_mu": mu_used,
            "seconds": time.perf_counter() - t0,
            "peak_vram_gb": torch.cuda.max_memory_allocated(self.device) / 2**30,
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
        }
        return images, info

    def _sample(self, embeds: dict, height: int, width: int, steps: int, generators: list, guidance: Guidance, seed: int,
                output_type: str, shift_mu: float | None):
        """QwenImagePipeline.__call__ steps 4–7 with ``guidance`` deciding the passes per step and how they combine.

        Differences from the Lumina loop: the transformer works on packed tokens (B, S, 64), so latents are unpacked
        to (B, 16, H/8, W/8) for the guidance code and repacked for every call; Qwen predicts the flow velocity
        v = ε − x0, so the guidance sees ``-v`` (Lumina's u = x0 − ε units, which the anchor's ``base`` mix assumes)
        and the scheduler gets ``-combine``; the timestep is σ itself (1 = noise); the CFG normalization is applied
        on the packed tokens after combining (over the 64 channels of a token, as the stock pipeline does), so the
        plain-CFG case stays bit-identical to the pipeline. An anchor pass calls the transformer on the pooled
        latent with ``img_shapes`` of that grid, which makes ``QwenEmbedRope`` (or DyPE) build its own RoPE.
        """
        pipe, sched, tr = self.pipe, self.pipe.scheduler, self.pipe.transformer
        def cond(e, mask):  # the pipeline passes no text mask for an unpadded batch; an all-ones mask selects a different
            return dict(encoder_hidden_states=e, encoder_hidden_states_mask=None if mask is None or bool(mask.all()) else mask)  # SDPA kernel (not bit-identical)

        pos = cond(embeds["prompt_embeds"], embeds["prompt_embeds_mask"])
        neg = cond(embeds.get("negative_prompt_embeds"), embeds.get("negative_prompt_embeds_mask"))
        batch = pos["encoder_hidden_states"].shape[0]
        ch = tr.config.in_channels // 4
        latents = pipe.prepare_latents(batch, ch, height, width, pos["encoder_hidden_states"].dtype, self.device, generators, None)  # (B, S, 64)
        hl, wl = 2 * (height // (pipe.vae_scale_factor * 2)), 2 * (width // (pipe.vae_scale_factor * 2))

        def unpack(z: torch.Tensor, h: int, w: int) -> torch.Tensor:  # (B, S, 4C) -> (B, C, h, w); inverse of _pack_latents
            b, _, c4 = z.shape
            return z.view(b, h // 2, w // 2, c4 // 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c4 // 4, h, w)

        def pack(x: torch.Tensor) -> torch.Tensor:
            return pipe._pack_latents(x, x.shape[0], x.shape[1], x.shape[-2], x.shape[-1])

        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = shift_mu if shift_mu is not None else pipeline_qwenimage.calculate_shift(
            latents.shape[1],
            sched.config.get("base_image_seq_len", 256),
            sched.config.get("max_image_seq_len", 4096),
            sched.config.get("base_shift", 0.5),
            sched.config.get("max_shift", 1.15),
        )
        timesteps, steps = pipeline_qwenimage.retrieve_timesteps(sched, steps, self.device, sigmas=sigmas, mu=mu)
        sched.set_begin_index(0)
        attention_kwargs = getattr(pipe, "_attention_kwargs", None) or {}
        normalization = guidance.normalization
        guidance.normalization = False  # done below on the packed tokens

        def run(cond: dict, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
            if cond["encoder_hidden_states"] is None:
                raise ValueError("this guidance needs an unconditional pass: encode with a negative prompt")
            v = tr(hidden_states=pack(x), timestep=ts, guidance=None, img_shapes=[[(1, x.shape[-2] // 2, x.shape[-1] // 2)]] * x.shape[0],
                   attention_kwargs=attention_kwargs, return_dict=False, **cond)[0]
            return -unpack(v, x.shape[-2], x.shape[-1])  # u = x0 - eps

        try:
            for i, t in enumerate(timesteps):
                step = Step(i, steps, float(t) / 1000)
                x = unpack(latents, hl, wl)
                ts = t.expand(batch).to(latents.dtype) / 1000  # cast then divide, as the pipeline does (bf16 rounding)
                passes = guidance.passes(step)
                preds = {"cond": run(pos, x, ts)}
                if "uncond" in passes:
                    preds["uncond"] = run(neg, x, ts)
                if "pert" in passes:
                    raise NotImplementedError("perturbation guidance (PAG/SEG/TPG) is not ported to Qwen-Image")
                if "anchor" in passes:  # REG: cond/uncond on the pooled latent, one pair per pyramid level
                    anchor = guidance.anchor_terms(step)[0]
                    if anchor.ratio == 1:
                        preds["anchor_cond"], preds["anchor_uncond"], preds["anchor_x"] = preds["cond"], preds["uncond"], x
                    else:
                        for r in anchor.levels:
                            x_a, sigma_a = anchor.anchor_input(x, step.sigma, r)
                            if x_a.shape[-2] % 2 or x_a.shape[-1] % 2:
                                raise ValueError(f"anchor level {r}: pooled latent {tuple(x_a.shape[-2:])} is not packable (needs even sides)")
                            preds[anchor.key(r, "x")] = x_a
                            ts_a = torch.full_like(ts, sigma_a)
                            preds[anchor.key(r, "cond")] = run(pos, x_a, ts_a)
                            preds[anchor.key(r, "uncond")] = run(neg, x_a, ts_a)
                v = pack(-guidance.combine(preds, step))
                if normalization and guidance.active_terms(step):
                    v = v * (torch.norm(pack(preds["cond"]), dim=-1, keepdim=True) / torch.norm(v, dim=-1, keepdim=True))
                latents = sched.step(v, t, latents, return_dict=False)[0]
        finally:
            guidance.normalization = normalization

        if output_type == "latent":
            return latents
        z = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor).to(pipe.vae.dtype)
        z_dim = pipe.vae.config.z_dim
        mean = torch.tensor(pipe.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(z.device, z.dtype)
        std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(z.device, z.dtype)
        z = z / std + mean
        image = pipe.vae.decode(z, return_dict=False)[0][:, :, 0]
        return pipe.image_processor.postprocess(image, output_type=output_type)
