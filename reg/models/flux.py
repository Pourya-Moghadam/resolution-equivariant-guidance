"""FLUX.1-dev (12B DiT, 2D RoPE, distilled guidance): the distilled-guidance case.

Memory on a 32 GB card: the bf16 transformer (~24 GB) and T5-XXL (~9.5 GB) do not fit
together, so loading is two-phase. A text-encoder-only pipeline fills the embedding cache
and is freed; a transformer+VAE pipeline then generates from cached embeddings.

Guidance: ``guidance_scale`` is the *embedded* (distilled) guidance FLUX was trained with.
Real CFG with a negative prompt is ``true_cfg_scale > 1`` (two transformer passes).

Schedule: FLUX uses dynamic timestep shifting (mu from token count, calibrated 256..4096
tokens). Beyond 1K the stock pipeline extrapolates linearly: mu ~ 3.2 at 2K and ~ 11.5 at
4K, which pins nearly every sigma at 1. ``shift_mu`` overrides it. DyPE's reference code
passes the current token count as the calibration maximum, i.e. clamps mu to max_shift
(1.15) at every resolution, for its baseline as well as for DyPE (see 07-dype-reference.md);
use ``shift_mu=1.15`` for any FLUX run above 1K that should be comparable.
"""

from __future__ import annotations

import contextlib
import gc
import time
from unittest import mock

import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux import pipeline_flux

from .embed_cache import TextEmbedCache

MODEL_ID = "black-forest-labs/FLUX.1-dev"
NAME = "flux1-dev"
MAX_SEQ_LEN = 512  # T5 tokens; the pipeline default


def schedule_mu(pipe: FluxPipeline, height: int, width: int) -> float:
    """The mu the pipeline would use for this resolution (patch 2 on an 8x VAE => /16)."""
    c = pipe.scheduler.config
    return pipeline_flux.calculate_shift(
        (height // 16) * (width // 16),
        c.get("base_image_seq_len", 256),
        c.get("max_image_seq_len", 4096),
        c.get("base_shift", 0.5),
        c.get("max_shift", 1.15),
    )


class Flux:
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16, quant: str | None = None):
        """``quant="fp8"`` quantizes the transformer weights to float8 (torchao, weight-only,
        ~12 GB instead of ~24 GB) after loading; needed for 4K on a 32 GB card."""
        if quant not in (None, "fp8"):
            raise ValueError(f"quant must be None or 'fp8', got {quant!r}")
        self.device = torch.device(device)
        self.dtype = dtype
        self.quant = quant
        self.cache = TextEmbedCache(NAME, max_sequence_length=MAX_SEQ_LEN)
        self._enc: FluxPipeline | None = None
        self._pipe: FluxPipeline | None = None

    # -- text ---------------------------------------------------------------------------
    def _encoder(self) -> FluxPipeline:
        if self._enc is None:
            self._enc = FluxPipeline.from_pretrained(MODEL_ID, transformer=None, vae=None, torch_dtype=self.dtype)
            self._enc.to(self.device)
        return self._enc

    def _encode_raw(self, text: str) -> dict[str, torch.Tensor]:
        """T5 (512, 4096) sequence embeddings and the CLIP (768,) pooled vector for one string."""
        embeds, pooled, _ = self._encoder().encode_prompt(
            prompt=text, prompt_2=None, device=self.device, max_sequence_length=MAX_SEQ_LEN
        )
        return {"t5": embeds[0], "pooled": pooled[0]}

    def encode(self, prompts: list[str], negative_prompt: str | None = None) -> dict[str, torch.Tensor]:
        pos = self.cache.get_batch(prompts, self._encode_raw, device=self.device)
        out = {"prompt_embeds": pos["t5"], "pooled_prompt_embeds": pos["pooled"]}
        if negative_prompt is not None:
            neg = self.cache.get_batch([negative_prompt] * len(prompts), self._encode_raw, device=self.device)
            out["negative_prompt_embeds"] = neg["t5"]
            out["negative_pooled_prompt_embeds"] = neg["pooled"]
        return out

    def free_text_encoders(self) -> None:
        if self._enc is not None:
            self._enc = None
            gc.collect()
            torch.cuda.empty_cache()

    # -- images -------------------------------------------------------------------------
    @property
    def pipe(self) -> FluxPipeline:
        if self._pipe is None:
            self._pipe = FluxPipeline.from_pretrained(
                MODEL_ID,
                text_encoder=None,
                text_encoder_2=None,
                tokenizer=None,
                tokenizer_2=None,
                torch_dtype=self.dtype,
            )
            self._pipe.to(self.device)
            if self.quant == "fp8":
                from torchao.quantization import Float8WeightOnlyConfig, quantize_

                quantize_(self._pipe.transformer, Float8WeightOnlyConfig())
                gc.collect()
                torch.cuda.empty_cache()
            self._pipe.vae.enable_tiling()
            self._pipe.set_progress_bar_config(disable=True)
        return self._pipe

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        height: int = 1024,
        width: int = 1024,
        steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int = 0,
        true_cfg_scale: float = 1.0,
        negative_prompt: str = "",
        shift_mu: float | None = None,
        output_type: str = "pil",
    ):
        """Return (images, info). Encodes (and caches) first, frees the encoders, then loads the DiT."""
        embeds = self.encode(prompts, negative_prompt if true_cfg_scale > 1 else None)
        self.free_text_encoders()
        pipe = self.pipe
        generators = [torch.Generator(self.device).manual_seed(seed + i) for i in range(len(prompts))]
        mu_used = shift_mu if shift_mu is not None else schedule_mu(pipe, height, width)
        shift_ctx = (
            mock.patch.object(pipeline_flux, "calculate_shift", lambda *a, **k: shift_mu)
            if shift_mu is not None
            else contextlib.nullcontext()
        )
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        with shift_ctx:
            images = pipe(
                prompt=None,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                true_cfg_scale=true_cfg_scale,
                generator=generators,
                max_sequence_length=MAX_SEQ_LEN,
                output_type=output_type,
                **embeds,
            ).images
        torch.cuda.synchronize(self.device)
        info = {
            "model": NAME,
            "quant": self.quant or "bf16",
            "true_cfg_scale": true_cfg_scale,
            "shift_mu": mu_used,
            "seconds": time.perf_counter() - t0,
            "peak_vram_gb": torch.cuda.max_memory_allocated(self.device) / 2**30,
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
        }
        return images, info
