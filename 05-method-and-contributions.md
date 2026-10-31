# 05 — Method and Contributions

**Working title:** *Guidance Does Not Extrapolate: Resolution-Equivariant Guidance for Diffusion Transformers*

**Method name:** REG (Resolution-Equivariant Guidance)

## Key idea
When resolution grows by a factor r (tokens per side), two things happen that current guidance ignores:

1. **Frequencies shift relative to the token grid.** A given level of detail in the image (cycles per image) sits at an r× lower frequency on the token grid. Any guidance with grid-based parameters is tuned to the wrong level of detail: FDG/FreCaS cutoffs, SEG blur width, perturbation neighbourhoods.
2. **Noise affects frequency bands differently.** Low frequencies average over r² more tokens, so at the same timestep their effective SNR rises by about r². Per-token high-frequency SNR stays the same. This is why SD3/FLUX shift the timestep with resolution, but that shift is applied uniformly to everything. Guidance has never been shifted, and the shift should differ per band.

**Principle:** the right guidance at resolution r is the native-resolution guidance moved into image-relative frequency and an equivalent SNR level, separately for each band. **No new hyperparameters:** everything is inherited from the native-resolution setting.

## Revision 2026-10-28: what the pilots changed
The Phase 1 pilots (08) settled two things before the grid finished:

- **The coarse band needs *more* guidance at high resolution, not less.** With FDG's cutoff placed in cycles per image (A), only the band spanning the whole native image responds to resolution: its optimal weight goes from ≈ 4 at 1K to ≈ 8 at 4K (window 6–10; 12 repeats subjects), while the finer bands are indifferent between 2 and 8. Plain CFG's optimum rises the same way because CFG raises all bands together. This is the opposite sign to the prediction under (B) as written above ("switches off earlier"), so (B) in that form is dropped as a prediction; the SNR shift still motivates the per-band treatment.
- **(B′) Band-wise magnitude matching (replaces B).** Hypothesis: under extrapolation the model's coarse-band guidance signal Δ_low shrinks relative to its base prediction, so the native weight under-guides the layout. REG sets w_low(r) = w_low(native) · ρ_native / ρ_r with ρ = ‖Δ_low‖ / ‖v_cond,low‖ measured at matched noise level, i.e. it restores the native *effective* guidance per band. No new hyperparameters: ρ is measured at run time (or once per model and r). **Test:** `jobs/reg_diag.jsonl` records ρ per step at 1K–4K with CFG 4; the prediction w_low(4K) must land in the measured 6–10 window and w_low(3K) in 6–8, with 2K ≈ flat. If ρ does not move, (C) (native anchor) becomes the lead component and the paper's method claim is scoped to "band-wise, with the coarse weight set by a one-time calibration".
- **What REG does not fix:** the 4K speckle is present at every guidance setting; it belongs to the extrapolation method (07).
- **2026-10-28, after the anchor pilot and diagnostics (08):** (B′) is refuted (Δ_low keeps its relative size). Anchoring only Δ's sub-native band (C as written) fails at 4K. Anchoring the *base prediction's* sub-native band at the derived noise level, with the native weight, gives a coherent 4K image where CFG 4 collapses (preliminary, one prompt; full pilot running). That variant is Upsample Guidance (2404.01709) in flow-matching form (04). **Revised contribution:** the finding and its diagnosis (the shift of the guidance optimum with resolution is a symptom of a shared coarse-band error of the base prediction, located by the band dissection), and the prescription that guidance weights stay native once the base's sub-native band is native — tested by whether the CFG optimum returns to ≈ 4 at 3K/4K with the anchor. (C) is kept as that prescription, credited to UG; (A) and (D) unchanged; the √r weight law is the no-anchor baseline.
- **2026-10-28, stage C (08):** the anchored weight curve at 3K coincides with the 2K plain curve at w = 2/4/8 on HPSv2 (within 0.3), and 4K anchored sits 1.5–2 below it, against 5–18 below for plain CFG at matched weights; the w4→w8 sensitivity returns to the native 1 point (plain 3K/4K: 3.5/8.8). HPSv2 has no interior optimum at any resolution, so the contribution is stated as *matched-weight equivalence*: with the base anchor each CFG weight behaves as at native (w2 under-guided but whole, w4 native quality, w8 a little better), and no resolution-dependent retuning is needed. Anchoring Δ alone is below plain CFG at every weight at 4K.
- **2026-10-29, method direction (after the paired grid result, 08):** the base anchor generalises in two ways that UG does not cover, both derived from the band dissection and both hyperparameter-free: (i) a **pyramid anchor**, each band below the full-resolution Nyquist taken from the lowest-resolution pass that represents it, at that pass's derived noise level (UG = the two-level case); (ii) a **super-native rule**, no guidance above the native Nyquist, where the model has never seen any frequency. (iii) a derived anchor interval (when the anchor is needed) and (iv) anchored perturbation guidance follow. If (i) or (ii) improves on the single-level anchor at 4K (pilot `jobs/reg_pyramid_pilot.jsonl`), the paper is a method paper: finding, diagnosis, and a general method of which UG is the first instance.
- **2026-10-29, pyramid pilot (08):** (i) confirmed on the six pilot prompts at 4K: the pyramid anchor beats the single-level anchor on HPS (+0.9, 5/6 paired), IR, CLIP, patch-CLIP and patch-KID (0.031 vs 0.051; equal to the single-level anchor at w8), with less speckle by eye and a 13 % time cost. (ii) refuted with the opposite sign: removing guidance above the native Nyquist adds speckle on 6/6 prompts (−1.7 HPS, KID 0.103); guidance there suppresses noise, so the rule becomes *more* weight above the native Nyquist (stage 2: super_scale 8/12 on the pyramid, pilot `jobs/reg_pyramid_pilot2.jsonl`). **The paper is a method paper:** finding, diagnosis, the pyramid anchor (UG = two-level case) and, if stage 2 confirms it, a super-native weight; the components remain hyperparameter-free except the super-native weight, which would need its own calibration argument.
- **2026-10-29, stage 2 (08):** the corrected super-native rule holds: pyramid + super_scale 8/12 at 4K, w4, beats the single-level anchor on 6/6 prompts (+1.4/+1.5 HPS), patch-KID 0.021 (below the single-level anchor at w8 and equal to 3K), speckle gone by eye, and HPS 31.1 = plain CFG w4 at 2K. The pyramid alone is neutral at 3K. **Method as of now (REG):** base anchor (UG) + pyramid levels + super-native weight, 13 % time over plain CFG at 4K. Open: a derivation for the super-native weight (SNR matching in the band, measurable with the recorder) so the method stays hyperparameter-free; its sweep (8/12/16/24) is stage 3.
- **2026-10-30, stage 3 (08):** ablation at 4K: super-native weight alone +0.5 HPS, pyramid alone +0.9, both +1.5 (additive, both needed). Super-native plateau 12–16 (KID 0.018 at 16), over-sharpening at 24. At 3K both components are a wash: the method at 3K is the single-level anchor. So the two new components are r-dependent in strength, and the derivation to write is the rule for that dependence (candidate: the super-native weight equals the plain-CFG optimum at that r, confined to the band above the native Nyquist; 4 → 12–16 from 1K to 4K in both the grid and the sweep). Grid-scale paired run (pyramid + 12, 50 4K prompts) running on reg4.
- **2026-10-30, grid scale (08):** on the 50 4K kill-or-go prompts REG (w4) beats the single-level anchor by +1.1 HPS (paired CI [0.7, 1.6], 74 % of prompts) and halves patch-KID (0.044 → 0.023, below plain CFG at any weight, 0.037 at w16); CLIP level; ImageReward −0.06 (small, significant). Over plain CFG's best weight: +3.2 HPS, +0.72 ImageReward, +2.1 CLIP. The headline claim is therefore three-tiered: anchoring the base restores the native weight (+15 HPS over plain at the same weight), the pyramid + super-native weight remove the residual super-native speckle (+1.1 HPS, KID halved), and the whole thing runs at w = 4 with no per-resolution tuning. Component ablation at this scale and the SEG/TPG/YaRN generality pilot are running.
- **2026-10-30 23:30, PAG / 5K pilot (08):** PAG at its 1K setting does not rescue plain CFG at 4K (HPS 15) and over-perturbs the REG base (−3.9 HPS on 6/6). So "guidance stays native" is true of the CFG weight, not of a perturbation term computed over r² more tokens: component (D), token-count calibration of PAG/SEG/TPG, is back in the method (scale sweep running; anchored Δ_pert as the fallback). REG at 5K works unchanged at w = 4 (HPS/IR ≥ the 4K values), with residual speckle growing with r at fixed super-native weight — the r-dependence of that weight is still the open derivation; 5K sweep running.

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
