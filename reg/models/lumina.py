"""Lumina-Image 2.0 (2.6B DiT, 2D RoPE, real CFG): the main development model.

Load once per process, encode prompts through the on-disk cache, then generate from
embeddings only so the Gemma-2 text encoder can be moved off the GPU for 2K/4K runs.
"""

from __future__ import annotations

import contextlib
import time

import numpy as np
import torch
from diffusers import Lumina2Pipeline
from diffusers.pipelines.lumina2.pipeline_lumina2 import calculate_shift, retrieve_timesteps

from ..guidance import Guidance, Step, build_guidance
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
        guidance: Guidance | None = None,
        use_pipeline: bool = False,
    ):
        """Return (images, info). One seed per prompt (seed + index) so rows are reproducible alone.

        Sampling runs in our own loop (``_sample``), a copy of Lumina2Pipeline.__call__'s loop with
        the branch combination delegated to a ``reg.guidance.Guidance``; with the default guidance
        (CFG ``guidance_scale`` on the interval (0, ``cfg_trunc_ratio``], ``cfg_normalization``) the
        output is bit-identical to the pipeline (``tests/test_guidance_real.py``). ``use_pipeline``
        runs the stock pipeline instead (CFG only), kept for that check.

        ``shift`` overrides the scheduler's flow-matching time shift for this call. Lumina ships
        shift=6 with use_dynamic_shifting=False, so its sigma schedule is the same at every
        resolution (the pipeline computes a token-count mu but the scheduler ignores it); FLUX
        instead shifts by token count. Restored afterwards.
        """
        if guidance is None:
            guidance = build_guidance({"guidance": guidance_scale, "cfg_trunc_ratio": cfg_trunc_ratio, "cfg_normalization": cfg_normalization})
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
            if use_pipeline:
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
            else:
                images = self._sample(embeds, height, width, steps, generators, guidance, seed, output_type)
        finally:
            if shift is not None:
                sched.register_to_config(shift=orig_shift)
        torch.cuda.synchronize(self.device)
        info = {
            "model": NAME,
            "shift": shift if shift is not None else orig_shift,
            "guidance": guidance.describe(),
            "seconds": time.perf_counter() - t0,
            "peak_vram_gb": torch.cuda.max_memory_allocated(self.device) / 2**30,
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
        }
        return images, info

    def _sample(self, embeds: dict, height: int, width: int, steps: int, generators: list, guidance: Guidance, seed: int, output_type: str):
        """Lumina2Pipeline.__call__ steps 4–7 with ``guidance`` deciding the passes per step and
        how they combine. Kept line-for-line close to the pipeline so the CFG case stays identical
        (including its mu computed from latents.shape[1], which the fixed-shift scheduler ignores)."""
        pipe, sched, tr = self.pipe, self.pipe.scheduler, self.pipe.transformer
        pos = dict(encoder_hidden_states=embeds["prompt_embeds"], encoder_attention_mask=embeds["prompt_attention_mask"])
        neg = dict(encoder_hidden_states=embeds["negative_prompt_embeds"], encoder_attention_mask=embeds["negative_prompt_attention_mask"])
        batch = pos["encoder_hidden_states"].shape[0]
        latents = pipe.prepare_latents(batch, tr.config.in_channels, height, width, pos["encoder_hidden_states"].dtype, self.device, generators, None)
        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = calculate_shift(
            latents.shape[1],
            sched.config.get("base_image_seq_len", 256),
            sched.config.get("max_image_seq_len", 4096),
            sched.config.get("base_shift", 0.5),
            sched.config.get("max_shift", 1.15),
        )
        timesteps, steps = retrieve_timesteps(sched, steps, self.device, sigmas=sigmas, mu=mu)
        sched.set_begin_index(0)
        attention_kwargs = getattr(pipe, "_attention_kwargs", None)
        p = tr.config.patch_size
        text_lens = pos["encoder_attention_mask"].sum(-1).tolist()
        if len(set(text_lens)) != 1:
            raise ValueError("perturbation guidance needs equal text lengths in a batch; run one prompt at a time")
        ctx = {"text_len": int(text_lens[0]), "grid": (latents.shape[-2] // p, latents.shape[-1] // p),
               "generator": torch.Generator("cpu").manual_seed(seed)}
        n_train = sched.config.num_train_timesteps

        def run(cond: dict, x: torch.Tensor | None = None, ts: torch.Tensor | None = None) -> torch.Tensor:
            x = latents if x is None else x
            ts = current_timestep if ts is None else ts
            return tr(hidden_states=x, timestep=ts, return_dict=False, attention_kwargs=attention_kwargs, **cond)[0]

        for i, t in enumerate(timesteps):
            step = Step(i, steps, float(t) / n_train)
            current_timestep = (1 - t / n_train).expand(latents.shape[0])  # Lumina: t=0 noise, t=1 image
            passes = guidance.passes(step)
            preds = {"cond": run(pos)}
            if "uncond" in passes:
                preds["uncond"] = run(neg)
            if "pert" in passes:
                with contextlib.ExitStack() as stack:
                    for term in guidance.pert_terms(step):
                        stack.enter_context(term.perturbation(tr, ctx))
                    preds["pert"] = run(pos)
            if "anchor" in passes:  # REG (C): cond/uncond at native resolution on the downsampled latent
                anchor = guidance.anchor_terms(step)[0]
                if anchor.ratio == 1:
                    preds["anchor_cond"], preds["anchor_uncond"] = preds["cond"], preds["uncond"]
                else:
                    x_a, sigma_a = anchor.anchor_input(latents, step.sigma)
                    ts_a = torch.full_like(current_timestep, 1.0 - sigma_a)
                    preds["anchor_cond"] = run(pos, x_a, ts_a)
                    preds["anchor_uncond"] = run(neg, x_a, ts_a)
            noise_pred = -guidance.combine(preds, step)
            latents = sched.step(noise_pred, t, latents, return_dict=False)[0]

        if output_type == "latent":
            return latents
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type=output_type)
