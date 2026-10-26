"""Lumina-Image 2.0 (2.6B DiT, 2D RoPE, real CFG): the main development model.

Load once per process, encode prompts through the on-disk cache, then generate from
embeddings only so the Gemma-2 text encoder can be moved off the GPU for 2K/4K runs.
"""

from __future__ import annotations

import time

import torch
from diffusers import Lumina2Pipeline

from .embed_cache import TextEmbedCache

MODEL_ID = "Alpha-VLLM/Lumina-Image-2.0"
NAME = "lumina2"
MAX_SEQ_LEN = 256  # the pipeline's default; >512 is rejected by the pipeline


def load_lumina(device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> Lumina2Pipeline:
    pipe = Lumina2Pipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe.to(device)
    pipe.vae.enable_tiling()  # 4K decode would otherwise need ~tens of GB in one shot
    pipe.set_progress_bar_config(disable=True)
    return pipe


class Lumina:
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16, pipe: Lumina2Pipeline | None = None):
        self.device = torch.device(device)
        self.pipe = pipe if pipe is not None else load_lumina(device, dtype)
        self.cache = TextEmbedCache(NAME, max_sequence_length=MAX_SEQ_LEN)

    # -- text ---------------------------------------------------------------------------
    def _encode_raw(self, text: str) -> dict[str, torch.Tensor]:
        """Run Gemma on one already-formatted string; returns (256, 2304) embeds and (256,) mask."""
        te = self.pipe.text_encoder
        if te.device != self.device:  # was offloaded; bring it back for the miss
            te.to(self.device)
        embeds, mask = self.pipe._get_gemma_prompt_embeds([text], device=self.device, max_sequence_length=MAX_SEQ_LEN)
        return {"embeds": embeds[0], "mask": mask[0]}

    def encode(self, prompts: list[str], negative_prompt: str = "", system_prompt: str | None = None) -> dict[str, torch.Tensor]:
        """Cached Gemma embeddings in the exact form Lumina2Pipeline.encode_prompt produces.

        The system prefix goes on the positive prompt only; the negative prompt (normally "")
        is encoded bare, matching the pipeline.
        """
        if system_prompt is None:
            system_prompt = self.pipe.system_prompt
        pos_texts = [system_prompt + " <Prompt Start> " + p for p in prompts]
        pos = self.cache.get_batch(pos_texts, self._encode_raw, device=self.device)
        neg = self.cache.get_batch([negative_prompt] * len(prompts), self._encode_raw, device=self.device)
        return {
            "prompt_embeds": pos["embeds"],
            "prompt_attention_mask": pos["mask"],
            "negative_prompt_embeds": neg["embeds"],
            "negative_prompt_attention_mask": neg["mask"],
        }

    def offload_text_encoder(self) -> None:
        """Free ~5 GB of VRAM once all prompts of a job are cached. Misses bring it back."""
        self.pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()

    # -- images -------------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        height: int = 1024,
        width: int = 1024,
        steps: int = 30,
        guidance_scale: float = 4.0,
        seed: int = 0,
        cfg_trunc_ratio: float = 1.0,
        cfg_normalization: bool = True,
        negative_prompt: str = "",
        system_prompt: str | None = None,
        output_type: str = "pil",
        shift: float | None = None,
    ):
        """Return (images, info). One seed per prompt (seed + index) so rows are reproducible alone.

        ``shift`` overrides the scheduler's flow-matching time shift for this call. Lumina ships
        shift=6 with use_dynamic_shifting=False, so its sigma schedule is the same at every
        resolution (the pipeline computes a token-count mu but the scheduler ignores it); FLUX
        instead shifts by token count. Restored afterwards.
        """
        embeds = self.encode(prompts, negative_prompt=negative_prompt, system_prompt=system_prompt)
        generators = [torch.Generator(self.device).manual_seed(seed + i) for i in range(len(prompts))]
        sched = self.pipe.scheduler
        orig_shift = sched.config.shift
        if shift is not None:
            sched.register_to_config(shift=shift)
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        try:
            images = self.pipe(
                prompt=None,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generators,
                cfg_trunc_ratio=cfg_trunc_ratio,
                cfg_normalization=cfg_normalization,
                max_sequence_length=MAX_SEQ_LEN,
                output_type=output_type,
                **embeds,
            ).images
        finally:
            if shift is not None:
                sched.register_to_config(shift=orig_shift)
        torch.cuda.synchronize(self.device)
        info = {
            "model": NAME,
            "shift": shift if shift is not None else orig_shift,
            "seconds": time.perf_counter() - t0,
            "peak_vram_gb": torch.cuda.max_memory_allocated(self.device) / 2**30,
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
        }
        return images, info
