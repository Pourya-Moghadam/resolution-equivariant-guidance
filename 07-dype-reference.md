# 07 — DyPE reference implementation (for the FLUX reproduction and the Lumina port)

Facts read from the authors' code on 2026-10-26, so the port does not have to re-derive them.

## Source
- Code: `https://github.com/guyyariv/DyPE` (MIT license; the "patent pending" line in the README is a notice, research use is covered by the MIT grant). Cloned read-only to `/data1/moghadamp/reg_data/third_party/DyPE-guyyariv` on reg1 (commit `436fe61`, 2026-05-18). `noamissachar/DyPE` is only the project page.
- Paper: arXiv 2510.20766, **accepted at ICML 2026**. Models in the code: FLUX (`flux/`) and Qwen-Image (`qwen/`); no Lumina.
- Layout: vendored copies of diffusers' `pipeline_flux.py` and `transformer_flux.py` with the changes below; `run_dype.py` is a 90-line driver.

## Driver defaults (`run_dype.py`)
- Model **FLUX.1-Krea-dev** (not FLUX.1-dev), `guidance_scale=4.5`, 28 steps, 4096×4096, `--method yarn` (choices `yarn | ntk | base`), DyPE on unless `--no_dype`, `enable_model_cpu_offload()`.
- Our reproduction (Phase 0 step 4) uses the FLUX.1-dev weights already in `$HF_HOME`; same architecture, so their transformer class loads them. State this wherever the numbers are reported.

## What the transformer change does (`flux/transformer_flux.py`)
- `FluxPosEmbed(theta, axes_dim, method, dype)` replaces diffusers' class. `base_resolution=1024`, `patch_size=16` (8× VAE × patch 2) → `base_patches=64`. Scaling applies only to the two spatial axes (`i > 0`), only when `current_patches > base_patches`.
- The timestep reaches it from `FluxTransformer2DModel.forward`: `self.pos_embed.set_timestep(timestep.item())`, where FLUX's `timestep` is already `t/1000 ∈ (0,1]`, 1 = pure noise. RoPE is therefore recomputed **every step** (and every pass, so twice per step with true CFG).
- **YaRN branch** (default): `scale = current_patches / base_patches`; three frequency bands blended with linear ramps in frequency-index space:
  - `freqs_linear = freqs_base / scale`, `freqs_ntk` from `new_base = theta · scale^(dim/(dim−2))`;
  - first blend between linear and NTK with correction range from `β0=1.25, β1=0.75` rotations;
  - second blend towards the unscaled `freqs_base` with range from `γ0=16, γ1=2` rotations (high-frequency dims keep their original frequencies);
  - attention-temperature `mscale = 0.1·ln(scale) + 1` multiplied into cos and sin.
- **NTK branch**: `ntk_factor = (scale^(dim/(dim−2)))^(2·κ)` with κ as below, floor 1.0.
- **DyPE dynamics**: `κ(t) = t^2` (λ_t = 2). All four correction thresholds β0, β1, γ0, γ1 are multiplied by κ. At t→1 (noise) the full YaRN rescaling is active; as t→0 the thresholds shrink toward 0, i.e. ever more frequency dims fall into the "keep original frequency" band and the encoding relaxes back to the native one. `method=base` disables everything (and forces `dype=False`).

## What the pipeline change does (`flux/pipeline_flux.py`)
- The only functional change: in `calculate_shift(...)` the calibration maximum `max_image_seq_len` (4096) is replaced by the **current** `image_seq_len`, so `mu = max_shift = 1.15` at every resolution instead of extrapolating (stock diffusers gives mu ≈ 3.2 at 2K and ≈ 11.5 at 4K). This applies to their `base` baseline too, so DyPE-vs-baseline comparisons hold the schedule fixed. **Our FLUX baselines must do the same** (`Flux.generate(shift_mu=1.15)`), or 4K "vanilla" failure is partly a degenerate schedule. Lumina is unaffected (fixed `shift=6`, see 06).
- Everything else is import-path and typing cosmetics against an older diffusers.

## Our implementation and the Phase 0 step 4 reproduction
- `reg/extrap/dype.py`: `DyPEFluxPosEmbed` (drop-in for diffusers' `FluxPosEmbed`) + `apply_dype(transformer, method, dype)` / `remove_dype`, which also wraps `forward` to pass the per-step timestep. `tests/test_dype.py` checks bit-identity with stock at native resolution, the YaRN band structure, relaxation to native as t→0, and the hook. The authors' code was read but not executed (environment policy); the tests stand in for a side-by-side run.
- `scripts/repro_dype_flux.py`: same prompt/seed per method through `reg.models.flux.Flux` with `shift_mu=1.15` for all methods.
- **2026-10-26, FLUX.1-dev, 2048², 28 steps, guidance 3.5, seed 0** (`$REG_DATA/outputs/dype_repro/`): `base` 100 s, `dy_yarn` 102 s (+2%), both 24.7 GB. Both coherent — with the clamped schedule FLUX handles 2K on its own; DyPE adds fine high-frequency content (falling snow, grass stalks) and a different composition. 2K is therefore not a discriminating test for FLUX; 4K is (run next).

## Lumina port — first results (2026-10-26, same fox prompt/seed as the vanilla images in 06)
- `DyPELuminaRope` in `reg/extrap/dype.py`, enabled via `apply_dype(pipe.transformer)` / `sample.py --extrap yarn`. Zero overhead: 2K 93 s, 4K 681 s, same VRAM as vanilla.
- **2K (`dy_yarn`): clean.** The vanilla duplication (second tail, body at the frame edge) is gone; sharp subject, coherent background. Validation target met.
- **4K (`dy_yarn`): a real scene instead of the vanilla flat colour** (pixel std 44 vs 18), fox and snowfield legible, but residual artefacts: the fox has extra legs (subject-level repetition) and dense grass texture dominates the frame. DyPE's constants (base 64 patches, β/γ, λ_t = 2) were tuned on FLUX 12B; a 2.6B model trained at 1K is a harder case. `jobs/extrap_4k.jsonl` compares yarn/ntk × dynamic/static before any constant tuning.
- Working interpretation for the project: at 4K with DyPE, Lumina is in the "coherent but fragile" regime where guidance settings plausibly matter most — the right place for the kill-or-go grid.
- **4K variants (`jobs/extrap_4k.jsonl`, one image per cell, ~690–720 s each across the cluster), partial (6/8) as of 17:20:** the method matters at 4K. Fox: **dynamic NTK clean** (correct anatomy, clean snow field); static NTK clean with a slightly doubled muzzle; dynamic YaRN extra legs + texture field. Lighthouse: dynamic YaRN keeps a (small, doubled) lighthouse; static YaRN a tall coherent one; **static NTK lost the lighthouse entirely** (sea only — content dropout, not repetition). Pending: fox static YaRN, lighthouse dynamic NTK. One image per cell cannot decide this; `jobs/extrap_4k_b.jsonl` (6 prompts × 2 seeds × {dy_yarn, dy_ntk}) is the follow-up before fixing the method for the kill-or-go grid.
- **24-row comparison (`jobs/extrap_4k_b.jsonl`: 6 prompts × 2 seeds × {dy_yarn, dy_ntk}, 694 s/image mean, reg1/3/4/5, 2026-10-26 evening). Verdict: dynamic NTK is the default extrapolation for Lumina in the kill-or-go grid.** NTK keeps one subject where YaRN repeats it: bicycle ×3 (YaRN, both seeds) vs 1 (NTK); pears duplicated across the table vs a single group; portrait skin texture duplicated vs intact face; Tokyo signs tiled vs a readable street. YaRN renders more fine detail (cathedral glass) but that is exactly the repetition-prone regime. Full `extrap_4k` 2×4 sheet completed too (lighthouse dy_ntk redone on reg3).
- **Open issue — white speckle texture at 4K in nearly all 24 images, both methods, bf16.** The same dotted texture appears in the FLUX 4K DyPE images (FP8), so it is not an FP8 artefact; guidance 3.5 vs 4.5 did not change it on FLUX. It is something in DyPE-as-we-run-it at 4K (or in these models at 4× extrapolation). `jobs/speckle_4k.jsonl` localises it: 50 steps, YaRN without `mscale`, 3072² (does it scale with the factor), CFG 2. Not visible at 2K.
- **Dynamic ≈ static within a method at both 2K and 4K.** Verified it is not a plumbing fault: `tests/test_dype_hooks_real.py` runs tiny real `Lumina2Transformer2DModel` / `FluxTransformer2DModel` instances at a >64-patch extent and checks the embedding receives the per-step noise level (Lumina's `1 − t/1000` inverted) and that dynamic ≠ static ≠ plain outputs. The small visual effect is a property of the schedules: with Lumina's fixed `shift=6` (and FLUX's `mu=1.15`) the noise level σ stays ≥0.75 for two thirds of the steps, so κ=σ² only shrinks the YaRN thresholds in the last few steps, after the layout is fixed. Hypothesis to test later: a smaller λ_t (κ=t^λ with λ<2, or κ tied to the sigma schedule) would let the dynamics act earlier on these models.
- **2K variants (`jobs/extrap_2k.jsonl`, reg1, 94 s each):** yarn, ntk, each dynamic and static, on the fox and lighthouse prompts — all eight clean, no duplication. Dynamic ≈ static within a method (near-identical images); YaRN vs NTK differ in composition only. At 2K the variant choice is immaterial for Lumina; the dynamics can only matter at 4K (`jobs/extrap_4k.jsonl`, workers).

## FLUX 4K (reg3, FP8 weight-only, 28 steps, guidance 3.5, mu 1.15, seed 0; bf16 OOMs on every shared host)
- `base`: **flat grey** (total collapse, like vanilla Lumina at 4K) — even with the clamped schedule. 839 s, 17.8 GB.
- `dy_yarn`: coherent fox, correct scene; artefacts: a dense dotted/halftone texture over sky and fur, one extra leg. 845 s (+0.7%).
- So DyPE's central claim reproduces with our implementation on FLUX.1-dev: vanilla 4K is unusable, DyPE 4K is a real image. The dotted texture is not in the paper's figures; candidates: FP8 weight-only, FLUX.1-dev vs Krea-dev, guidance 3.5 vs their 4.5, 28 steps. **Guidance ruled out** (reg4, 4.5: same texture, same composition, 841 s). FP8 vs bf16 isolated via the 2K FP8 DyPE run on reg5 against the clean 2K bf16 DyPE image (pending). Krea-dev would need a separate download; the authors' figures are Krea, so a residual difference may simply be the model.

## Port to Lumina (Phase 1 step 1) — where it hooks
- diffusers `Lumina2RotaryPosEmbed` precomputes complex `freqs_cis` tables per axis (`axes_lens=(300,512,512)`, `axes_dim=(32,32,32)`, θ=10000, float64) and gathers rows by integer position; image rows/cols are axes 1 and 2, axis 0 carries caption positions (image tokens all get `cap_seq_len` there). Native 1024² = 64×64 patches, matching DyPE's `base_patches=64`.
- Minimal port: replace the gathered spatial-axis frequencies with the YaRN/NTK-scaled ones computed from the same `get_1d_rotary_pos_embed` math (complex form instead of cos/sin), recomputed per step from the pipeline's `current_timestep` (Lumina passes `1 − t/1000`, so **1 = clean image** there — invert before computing κ). Leave axis 0 untouched. Apply `mscale` the same way.
- Validation target (PLAN Phase 1): Lumina + DyPE at 2K/4K coherent vs. the vanilla collapse documented in 06.
