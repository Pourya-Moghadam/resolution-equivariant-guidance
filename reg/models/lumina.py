"""Lumina-Image 2.0 (2.6B DiT, 2D RoPE, real CFG): the main development model.

Load once per process, encode prompts through the on-disk cache, then generate from
embeddings only so the Gemma-2 text encoder can be moved off the GPU for 2K/4K runs.
"""

from __future__ import annotations

import contextlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import Lumina2Pipeline
from diffusers.pipelines.lumina2.pipeline_lumina2 import calculate_shift, retrieve_timesteps

from ..guidance import Guidance, Step, build_guidance
from .embed_cache import TextEmbedCache

MODEL_ID = "Alpha-VLLM/Lumina-Image-2.0"
NAME = "lumina2"
MAX_SEQ_LEN = 256  # the pipeline's default; >512 is rejected by the pipeline


def _butterworth(h: int, w: int, ratio: float, device, n: int = 4) -> torch.Tensor:
    """HiFlow's low-pass mask for an fft-shifted 2-D spectrum: 1 / (1 + (d / ratio)^(2n)), d the radial frequency with 1 at
    the per-axis Nyquist (butterworth_low_pass_filter_2d in the official utils.py, vectorised)."""
    yy = 2 * torch.arange(h, device=device, dtype=torch.float32) / h - 1
    xx = 2 * torch.arange(w, device=device, dtype=torch.float32) / w - 1
    d2 = yy[:, None] ** 2 + xx[None, :] ** 2
    return 1 / (1 + (d2 / ratio ** 2) ** n) if ratio > 0 else torch.zeros_like(d2)


def _unsharp(image: torch.Tensor, sigma: float, amount: float) -> torch.Tensor:
    """(amount + 1) * image - amount * GaussianBlur_3x3(image), reflect padding (HiFlow's gaussian_blur_image_sharpening)."""
    k = torch.exp(-torch.tensor([-1.0, 0.0, 1.0], device=image.device) ** 2 / (2 * sigma ** 2)); k = (k / k.sum()).to(image.dtype)
    c = image.shape[1]
    blur = F.conv2d(F.pad(image, (1, 1, 0, 0), mode="reflect"), k.view(1, 1, 1, 3).expand(c, 1, 1, 3), groups=c)
    blur = F.conv2d(F.pad(blur, (0, 0, 1, 1), mode="reflect"), k.view(1, 1, 3, 1).expand(c, 1, 3, 1), groups=c)
    return (amount + 1) * image - amount * blur


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
        hires: dict | None = None,
        hiflow: dict | None = None,
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

        ``hires`` = {"strength": 0.5, "base_size": 1024, "base_guidance": 4.0} runs the two-stage "hires fix"
        baseline instead of sampling from noise: stage 1 samples at ``base_size`` with plain CFG at
        ``base_guidance``; the decoded image is Lanczos-upscaled to (height, width), VAE-encoded, noised to the
        sigma of step ``steps - round(steps * strength)`` of the schedule, and stage 2 runs the remaining steps
        at the target size with ``guidance`` (the row's guidance). Time and peak memory cover both stages.

        ``hiflow`` runs the HiFlow cascade (``_hiflow``) instead; the row's ``guidance`` is not used (weights are per stage).
        """
        if guidance is None:
            guidance = build_guidance({"guidance": guidance_scale, "cfg_trunc_ratio": cfg_trunc_ratio, "cfg_normalization": cfg_normalization})
        embeds = self.encode(prompts, negative_prompt=negative_prompt, system_prompt=system_prompt)
        generators = [torch.Generator(self.device).manual_seed(seed + i) for i in range(len(prompts))]
        sched = self.pipe.scheduler
        orig_shift = sched.config.shift
        if shift is not None:
            sched.register_to_config(shift=shift)
        cuda = self.device.type == "cuda"
        if cuda:
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
            elif hiflow:
                images = self._hiflow(embeds, height, width, steps, generators, seed, output_type, hiflow, cfg_normalization)
            elif hires:
                bs = int(hires.get("base_size", 1024))
                g1 = build_guidance({"guidance": float(hires.get("base_guidance", 4.0)), "cfg_normalization": cfg_normalization})
                base = self._sample(embeds, bs, bs, steps, generators, g1, seed, "pil")
                init = self._encode_images([im.resize((width, height), Image.LANCZOS) for im in base])
                del base
                images = self._sample(embeds, height, width, steps, generators, guidance, seed, output_type,
                                      init_latents=init, strength=float(hires.get("strength", 0.5)))
            else:
                images = self._sample(embeds, height, width, steps, generators, guidance, seed, output_type)
        finally:
            if shift is not None:
                sched.register_to_config(shift=orig_shift)
        if cuda:
            torch.cuda.synchronize(self.device)
        info = {
            "model": NAME,
            "shift": shift if shift is not None else orig_shift,
            "guidance": guidance.describe(),
            "seconds": time.perf_counter() - t0,
            "peak_vram_gb": torch.cuda.max_memory_allocated(self.device) / 2**30 if cuda else 0.0,
            **({"hires": dict(hires)} if hires else {}),
            **({"hiflow": dict(hiflow)} if hiflow else {}),
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
        }
        return images, info

    def _encode_images(self, images: list) -> torch.Tensor:
        """PIL images -> scaled VAE latents (the inverse of the decode at the end of ``_sample``; tiled)."""
        vae = self.pipe.vae
        x = self.pipe.image_processor.preprocess(images).to(self.device, vae.dtype)
        lat = vae.encode(x).latent_dist.mode()
        return (lat - vae.config.shift_factor) * vae.config.scaling_factor

    def _hiflow(self, embeds: dict, height: int, width: int, steps: int, generators: list, seed: int, output_type: str, cfg: dict,
                cfg_normalization: bool = True):
        """HiFlow (Bu et al., 2025, arXiv:2504.06232) ported from the official FLUX pipeline (github.com/Bujiazi/HiFlow,
        flux_pipeline_hiflow.py, Apache-2.0) to this sampler. A cascade base -> sizes[0] -> ... -> (height, width) on one
        shared sigma schedule (Lumina's fixed shift makes the schedule resolution-independent, as the official code requires):

          stage 0   plain CFG at ``base_size`` and ``base_guidance``; the denoised prediction x0 = x - sigma v of every step is kept.
          stage k   initialisation alignment: decode the previous stage, bicubic-upsample in pixel space, unsharp mask
                    (3x3 Gaussian, amount 1), VAE-encode, noise to the sigma of step ``steps - n_k`` and run the last n_k steps.
                    direction alignment, every step: x0 <- x0 + a_i (LP(x0_ref) - LP(x0)), x0_ref the previous stage's raw x0 of
                    the same step, bicubic-upsampled in latent space; LP a 4th-order Butterworth low-pass on the 2-D FFT with
                    cutoff ``filter_ratio`` (1 = Nyquist of the stage's latent grid); then v = (x - x0) / sigma.
                    acceleration alignment, from the second step: v <- v + b_i (v_prev + v_ref - v_ref_prev - v), with
                    v_ref = (x - x0_ref) / sigma. a_i = alphas[k] (n_k - i) / n_k, b_i likewise. The stage's raw x0 replaces the
                    stored reference for the next stage.

        ``cfg`` keys and the official script's values: sizes [2048, 4096], steps [16, 10] (of 30), alphas [1.0, 0.25],
        betas [0.5, 0.5], filter_ratio [0.2, 0.2], guidance per stage (official: FLUX guidance embedding 3.5 at the base and
        4.5, 6 above; here real CFG weights), base_size 1024, base_guidance, sharpen_sigma (the official code draws the blur
        sigma uniformly from (0.1, 2.0) per call through torchvision; default here is the mean, 1.05, for reproducibility).
        Not ported, because they are FLUX stabilisers and not part of the three alignments: the shifted 2x2 patch grid on
        alternate steps, the fixed NTK factor 10 and proportional attention; the position-encoding extrapolation is whatever
        the caller installed (the grid's dynamic NTK), as for every other method.
        """
        sizes = [int(x) for x in cfg.get("sizes", [2048, 4096])]
        if sizes[-1] != height or height != width:
            raise ValueError("hiflow: the last stage size must equal the (square) output size")
        k = len(sizes)
        lst = lambda key, default: [float(x) for x in (cfg.get(key, default) if isinstance(cfg.get(key, default), (list, tuple)) else [cfg.get(key, default)] * k)]
        n_high, alphas, betas, ratios = [int(x) for x in lst("steps", [16, 10])], lst("alphas", [1.0, 0.25]), lst("betas", [0.5, 0.5]), lst("filter_ratio", 0.2)
        weights = lst("guidance", float(cfg.get("base_guidance", 4.0)))
        if not all(len(v) == k for v in (n_high, alphas, betas, ratios, weights)):
            raise ValueError("hiflow: steps, alphas, betas, filter_ratio and guidance need one value per stage")
        bs = int(cfg.get("base_size", 1024)); vae = self.pipe.vae
        ref: dict[int, torch.Tensor] = {}

        def keep(i, sigma, x, v):
            ref[i] = (x.float() - sigma * v.float())
            return v

        g0 = build_guidance({"guidance": float(cfg.get("base_guidance", 4.0)), "cfg_normalization": cfg_normalization})
        latents = self._sample(embeds, bs, bs, steps, generators, g0, seed, "latent", flow_hook=keep)
        for size, n, a, b, ratio, w in zip(sizes, n_high, alphas, betas, ratios, weights):
            image = vae.decode((latents / vae.config.scaling_factor + vae.config.shift_factor).to(vae.dtype), return_dict=False)[0].float()
            image = F.interpolate(image, (size, size), mode="bicubic", align_corners=False)
            image = _unsharp(image, float(cfg.get("sharpen_sigma", 1.05)), 1.0)
            init = (vae.encode(image.to(vae.dtype)).latent_dist.mode() - vae.config.shift_factor) * vae.config.scaling_factor
            del image
            lp = _butterworth(init.shape[-2], init.shape[-1], ratio, init.device)
            low = lambda x: torch.fft.ifft2(torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fft2(x)) * lp)).real
            first, prev, state = steps - n, dict(ref), {}

            def align(i, sigma, x, v, first=first, n=n, a=a, b=b, prev=prev, state=state, low=low):
                x32, x0 = x.float(), x.float() - sigma * v.float()
                ref[i] = x0                                   # raw prediction: the next stage's reference
                x0_ref = F.interpolate(prev[i], x0.shape[-2:], mode="bicubic", align_corners=False)
                decay = (n - (i - first)) / n
                x0 = x0 + a * decay * (low(x0_ref) - low(x0))
                v_new, v_ref = (x32 - x0) / (sigma + 1e-6), (x32 - x0_ref) / (sigma + 1e-6)
                if "v" in state:
                    v_new = v_new + b * decay * (state["v"] + v_ref - state["v_ref"] - v_new)
                state["v"], state["v_ref"] = v_new, v_ref
                return v_new.to(v.dtype)

            g = build_guidance({"guidance": w, "cfg_normalization": cfg_normalization})
            last = size == sizes[-1]
            latents = self._sample(embeds, size, size, steps, generators, g, seed, output_type if last else "latent",
                                   init_latents=init, strength=n / steps, flow_hook=align)
        return latents

    def _sample(self, embeds: dict, height: int, width: int, steps: int, generators: list, guidance: Guidance, seed: int, output_type: str,
                init_latents: torch.Tensor | None = None, strength: float = 1.0, flow_hook=None):
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
        first = 0
        if init_latents is not None:  # hires fix: start from the noised upscaled image at step ``first`` of the same schedule
            first = max(0, min(steps - 1, steps - round(steps * strength)))
            s0 = float(sched.sigmas[first])
            latents = (1.0 - s0) * init_latents.to(latents.dtype) + s0 * latents
            timesteps = timesteps[first:]
        sched.set_begin_index(first)
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

        for i, t in enumerate(timesteps, start=first):
            step = Step(i, steps, float(t) / n_train)
            current_timestep = (1 - t / n_train).expand(latents.shape[0])  # Lumina: t=0 noise, t=1 image
            passes = guidance.passes(step)
            preds = {"cond": run(pos), "x": latents}  # "x": the noisy latent, for terms that work on the denoised estimate (APG)
            if "uncond" in passes:
                preds["uncond"] = run(neg)
            if "pert" in passes:
                with contextlib.ExitStack() as stack:
                    for term in guidance.pert_terms(step):
                        stack.enter_context(term.perturbation(tr, ctx))
                    preds["pert"] = run(pos)
            if "anchor" in passes:  # REG (C): cond/uncond on the downsampled latent, one pair per pyramid level
                anchor = guidance.anchor_terms(step)[0]
                if anchor.ratio == 1:
                    preds["anchor_cond"], preds["anchor_uncond"], preds["anchor_x"] = preds["cond"], preds["uncond"], latents
                else:
                    for r in anchor.levels:
                        x_a, sigma_a = anchor.anchor_input(latents, step.sigma, r)
                        preds[anchor.key(r, "x")] = x_a
                        ts_a = torch.full_like(current_timestep, 1.0 - sigma_a)
                        preds[anchor.key(r, "cond")] = run(pos, x_a, ts_a)
                        preds[anchor.key(r, "uncond")] = run(neg, x_a, ts_a)
            noise_pred = -guidance.combine(preds, step)
            if flow_hook is not None:  # HiFlow: (step index, sigma, noisy latent, velocity) -> velocity; x0 = x - sigma * velocity
                noise_pred = flow_hook(i, float(sched.sigmas[i]), latents, noise_pred)
            latents = sched.step(noise_pred, t, latents, return_dict=False)[0]

        if output_type == "latent":
            return latents
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type=output_type)
