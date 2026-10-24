# 05 — Method and Contributions

**Working title:** *Guidance Does Not Extrapolate: Resolution-Equivariant Guidance for Diffusion Transformers*

**Method name:** REG (Resolution-Equivariant Guidance)

## Key idea
When resolution grows by a factor r (tokens per side), two things happen that current guidance ignores:

1. **Frequencies shift relative to the token grid.** A given level of detail in the image (cycles per image) sits at an r× lower frequency on the token grid. Any guidance with grid-based parameters is tuned to the wrong level of detail: FDG/FreCaS cutoffs, SEG blur width, perturbation neighbourhoods.
2. **Noise affects frequency bands differently.** Low frequencies average over r² more tokens, so at the same timestep their effective SNR rises by about r². Per-token high-frequency SNR stays the same. This is why SD3/FLUX shift the timestep with resolution, but that shift is applied uniformly to everything. Guidance has never been shifted, and the shift should differ per band.

**Principle:** the right guidance at resolution r is the native-resolution guidance moved into image-relative frequency and an equivalent SNR level, separately for each band. **No new hyperparameters:** everything is inherited from the native-resolution setting.

## Method: four training-free plug-in components
Let Δ_t be the guidance signal: v_cond − v_uncond for CFG, or v_clean − v_perturbed for PAG/SEG/TPG.

### (A) Image-relative band split
Decompose Δ_t with the cutoff κ fixed in **cycles per image** (grid cutoff = κ / H_latent), so "low" always means the same semantic scale (layout, objects). Use the native band weights (w_low, w_high).

### (B) Band-wise SNR-matched schedule
Define each band's guidance schedule and interval on the native time axis. At resolution r, map the current t to the equivalent native time **per band**:
- **Low band:** apply the resolution shift (SD3 form with α = r: t_shift = α t / (1 + (α − 1) t), inverted as needed)
- **High band:** no shift

Prediction: low-frequency guidance switches off earlier at high resolution, which should reduce duplicated subjects and over-saturation. This explains the URAE/Switti observation (high resolution wants less CFG) from a rule rather than by observation.

### (C) Native-anchored low-band guidance (optional, strongest variant)
- Compute the low-band guidance from one extra forward pass on x_t downsampled to native resolution, where the model is in-distribution.
- Downsampling by r averages r² noise samples, so the noise level is exactly known and lower. Query the model at the matched time (same shift relation as (B), so the theory stays unified).
- High-band guidance and the base prediction stay at full resolution.
- **Cost:** about 1/r² extra tokens per step, negligible.
- **How it differs:** FreCaS splits the high-resolution score; ScaleCrafter swaps convolution kernels; TokenScale, HiFlow and ResMaster anchor the latent. We anchor only the guidance signal, with a derived noise level.

### (D) Perturbation-guidance calibration
Set the perturbation so the branch is degraded by the same amount as at native resolution:
- **SEG:** blur width in image-relative units, σ_tok = r · σ₀
- **TPG/PAG:** perturb within windows covering the native receptive field, or set the strength so the clean-versus-perturbed **attention-entropy gap** matches the native one (our perturbation-side counterpart to the log-n attention temperature scaling used on the model side)

## Contributions (paper framing)
1. **Finding:** the first systematic study showing that guidance tuned at native resolution is mistuned under resolution extrapolation. It covers CFG-type and perturbation-type guidance across DyPE, SEGA and UltraImage. We measure guidance spectrum, guidance size relative to the base prediction, and attention entropy versus resolution, and link the mismatch to repetition, over-saturation and loss of detail.
2. **Theory:** a resolution-equivariance principle based on band-wise SNR shifts. It explains the hand-tuned results in FreCaS, RectifiedHR, URAE and Switti as special cases. Stretch goal: closed-form w(r) under a Gaussian / linear-diffusion model.
3. **Method:** REG, a training-free wrapper with no new hyperparameters (A + B, optional C, D for perturbation guidance). Works with any guidance method and any extrapolation method.
4. **Evaluation:** Lumina-Image 2.0 and Qwen-Image (real CFG), FLUX.1-dev as the distilled-guidance case. 2K/4K quantitative, 8K qualitative. Metrics: patch FID/KID, CLIP, GenEval, HPS/ImageReward, a repetition/duplicate metric, overhead. Optional video extension (Wan + RIFLEx).

> Note: SD3.5 was originally proposed but **does not use RoPE**, so DyPE/SEGA/UltraImage don't apply to it. Replaced by Lumina-Image 2.0 (see 06).

## Experiment order
1. **Kill-or-go (first experiment):** Lumina-Image 2.0 + DyPE. Grid-search guidance scale × interval × band weights at 1K/2K/4K. **Go** if the best settings move with resolution in the direction (B) predicts, especially low-band guidance switching off earlier. **Stop** if they don't move.
2. **Diagnostics + (A), (B), (D):** check that they reach the grid-search optimum with no tuning. This is the paper's headline.
3. **(C), full matrix, ablations.**
4. **Target:** ICML 2027 (late January 2027); fallback NeurIPS 2027.

## Risks
- (B) may only match the tuned optimum roughly. Still publishable as "explains and approximates"; (C) is the fallback that makes the gains large.
- Scooping ([name removed]'s group is best placed). Re-check arXiv every 2 weeks; consider contacting the [name removed] lab to collaborate.
- Read the FreCaS SD3 setting and the RectifiedHR appendix again in full before writing related work.
