# 04 — Novelty Check

Done 2026-10-24. Question: has anyone already studied or adapted **guidance (CFG-type or perturbation-type) as a function of resolution, or at extrapolated resolution in DiTs**?

## Verdict
**The core idea has not been done.** We found no paper that:
1. Systematically measures how the best guidance setting changes with resolution in RoPE-extrapolated DiTs;
2. Studies perturbation guidance (PAG/SEG/TPG/S²) at extrapolated resolution (**zero hits** from every source);
3. Derives a rule that sets guidance scale, interval or frequency cutoffs from resolution;
4. Combines guidance fixes with DyPE/SEGA/UltraImage.

SEGA explicitly uses the default guidance scales at 4K, which is a clean gap to point to.

Prior work only shows the **symptom**: hand-tuned CFG changes at high resolution (RectifiedHR, FreCaS), or CFG switched off (URAE, Switti). That's our motivation.

## Search log
1. **arXiv API**, 9 abstract queries, sorted by date (~400 results screened):
   - guidance + high-resolution + diffusion + training-free
   - "classifier-free guidance" + resolution (51 results)
   - guidance + resolution extrapolation / higher resolution / ultra-high
   - PAG / SEG / attention guidance / self-guidance + resolution
   - guidance + RoPE / positional encoding + extrapolation
   - guidance + resolution-dependent / resolution-aware / scale-aware / token count / timestep shift
   - guidance scale/strength/weight + 4K/2K/high-resolution
   - guidance + video + length extrapolation / long video
2. **Semantic Scholar citation tracing:** all papers citing DyPE (16), FDG (14), TPG (18), UltraImage (13), SEGA (3)
3. **OpenReview:** ICLR 2027 submissions for "classifier-free guidance high-resolution" (33) and "guidance resolution" (31), plus a general OpenReview search
4. **Full-text reading:** FreCaS, both RectifiedHR papers, ScaleCrafter (below)

Limitations: abstracts were screened, not every full text. Papers from the last few days, or not on arXiv, may be missing. Some OpenReview pages were truncated. **Re-check every ~2 weeks.**

## Prior art to cite and distinguish from

| Paper | What it does with guidance | How we differ |
|---|---|---|
| **ScaleCrafter** (2310.07702) | Noise-damped CFG for extrapolation | UNet with dilated convolutions; one fixed trick; default CFG scale at all resolutions |
| **FreCaS** (2410.18410) | Frequency-aware CFG inside a cascade; cutoff = previous stage's Nyquist | **Closest conceptually.** Hand-tuned, cascade only, no analysis, no RoPE |
| **RectifiedHR** (2503.02537) | CFG raised across stages by a power-law schedule to offset energy decay | Hyperparameter search; fixes its own pipeline's side effect; no RoPE, no frequency or perturbation guidance |
| **RectifiedHR** (2507.09441) | Energy-based CFG schedules | SD 1.5 at 512² only; no resolution dependence; marked "do not cite" |
| **URAE** (2503.16322) | CFG off (scale 1) for FLUX during 2K/4K adaptation | Empirical observation; **supports our hypothesis** |
| **Switti** (2412.01819) | CFG disabled at high-resolution scales (scale-wise autoregressive model) | Different architecture; empirical; **supports our hypothesis** |
| **EditCrafter** (2604.10268) | NDCFG++ for high-resolution editing | Editing task |
| **ZeResFDG** (2510.12954) | Frequency-split guidance helping high-resolution texture | No resolution dependence |
| **SAFE-DiT+SW** (2606.29360) | Sensitivity-weighted CFG variant in a high-resolution speed-up paper | Speed-up paper; CFG change not tied to resolution |
| Upsample Guidance, HiFlow, ResMaster, TokenScale, WAVEALIGN, LookFlow | Guidance from a low-resolution reference | A different kind of "guidance" (structure anchoring), not CFG or perturbation guidance |
| Diff-VF (2608.05976) | Timestep-dependent guidance for long video | Video length; not resolution-derived |

## Detailed reading of the four closest papers

### RectifiedHR: Energy Rectification (2503.02537v4)
- **Method:** noise refresh (predicted x0 → decode → bilinear resize in RGB → re-encode → re-noise) at stage timesteps T_i = ⌊(T_max − T_min)((i−1)/N)^{M_T} + T_min⌋.
- **Energy rectification:** CFG raised across stages, ω_i = (ω_max − ω_min)(i/(N−1))^{M_ω} + ω_min. Energy is E[x_t²].
- **Values (SDXL):**

  | Target | Stages N | ω_min | ω_max | M_ω |
  |---|---|---|---|---|
  | 2048² | 2 | 5 | 30 | 1 |
  | 4096² | 3 | 5 | 50 | 0.5 |

  Per-task tables also exist (WAN 960×1664: ω 5→10; 3072²: 5→40).
- **Hyperparameters:** "obtained through a hyperparameter search". No rule derived from resolution.
- **Models:** SDXL (main), SD1.4 (DreamBooth), SD3-medium (one table, 2048², no guidance details), WAN 1.3B.
- **Not covered:** RoPE, frequency-split guidance, perturbation guidance, statistics of the guidance term (ε_c − ε_∅) versus resolution.
- **Theory:** appendix links energy to resolution via γ = (H′/H · W′/W)² and an SNR correction. That's a hint in our direction that it never develops.

### RectifiedHR: Adaptive Guidance Scheduling (2507.09441v2)
- Linear, cosine, step, exponential and sigmoid CFG schedules over t/T, plus energy clipping.
- **SD 1.5 at 512×512 only.** "Our study is limited to SD 1.5 and 512×512 resolution."
- No resolution dependence, RoPE, frequency or perturbation guidance.
- The header says "Please do not cite"; the bibliography has errors. Cite at most in passing.

### FreCaS (2410.18410v2)
- **Method:** cascaded stages of increasing resolution. FA-CFG splits uncond and cond scores into a low part (downsample → resize back) and a high part (the residual), with cutoff = previous stage's Nyquist frequency. Separate weights w_l and w_h; w_l = w_h recovers CFG.
- **Weights, hand-tuned:**
  - w_l = 7.5 (SD2.1, SDXL), 7.0 (SD3)
  - w_h = 45 (SD2.1 ×4), 35 (SD2.1 ×16, SDXL ×4/×16, SD3 ×4)
  - Swept only on SDXL ×4: best range 25–35, artifacts at 45
- **Scope:** FA-CFG is used only in the cascade stages after the first, never in direct high-resolution generation.
- **SD3:** a single ×4 entry; positional embeddings not discussed.
- **Not covered:** why high resolution needs different guidance, any guidance interval, any link between guidance and its own timestep shift, perturbation guidance.

### ScaleCrafter (2310.07702)
- **Noise-damped CFG:** ε_θ(x_t) + w·(ε̃_θ(x_t, y) − ε̃_θ(x_t)). The base prediction comes from the original UNet; the guidance difference comes from the re-dilated or dispersed-convolution UNet, where the shared errors cancel.
- Uses the default CFG scale at all resolutions (7.5 for SD1.5/2.1, 5.0 for SDXL). Only the choice of modified blocks varies with resolution.
- UNet-convolution specific.

## Monitoring list (scoop risk)
- the TPG/SEGA authors (our idea combines both)
- Sadat et al. (APG, FDG)
- DyPE, UltraImage, TIDE authors
- ICLR 2027 public submissions; new arXiv papers citing DyPE, SEGA, FDG or TPG

## 2026-10-28: prior art for the native-resolution anchor (after the anchor pilot, 08)
- **Upsample Guidance (2404.01709, Hwang et al. 2024)** adds one term to the noise prediction: w_t·U[ε(D[x_t]/√P, τ)/m − D[ε(x_t)]], with D block averaging by m, P = α_t + (1−α_t)/m² the power-matching factor and τ the timestep at which the low-resolution SNR matches; w_t = θ·H(t − (1−η)T) with (θ, η) ≈ (1, 0.6) for latent models (switched off late to avoid VAE artifacts). **This is our base-anchored variant** (`reg/guidance/anchor.py`, `base=True`): our c and σ′ are the flow-matching form of their √P and τ, and w = 1 replaces the sub-native band by the native prediction, as we do. Differences: pixel/latent UNets (SD1.5, m = 2, ≤ 1280²), nearest-neighbour upsampling, no DiT/RoPE extrapolation, **no statement about CFG** (scale or interaction), no analysis of why high resolution needs it.
- **HiFlow (2504.06232)**: a full low-resolution trajectory as reference; replaces the low band (Butterworth, D = 0.4) of the *denoised estimate* with the reference's and aligns acceleration, weights t/τ; FLUX 2K/4K cascade. CFG scale not discussed.
- **ScaleCrafter's noise-damped CFG** already uses the structure we found: base from one network, Δ from another, "shared errors cancel".
- **Consequence for the paper:** anchoring is not ours. What is ours: (i) the measured finding that the CFG optimum shifts with resolution and the band dissection locating it in the native-image band; (ii) the diagnosis that it is a shared base error, not a guidance-signal error (Δ-only anchoring fails, Δ keeps its relative size, spectrum drift); (iii) the prescription that follows: anchor the base's sub-native band at the derived noise level and keep *native* guidance weights per band — guidance becomes resolution-equivariant — with the explicit test that the CFG optimum returns to the native value at 3K/4K; (iv) the DiT + RoPE-extrapolation setting (DyPE/NTK, 3–4K) where none of the above was tested. Cite UG as the ancestor of the anchor and position REG as "why and what to anchor, and what it does to guidance".
