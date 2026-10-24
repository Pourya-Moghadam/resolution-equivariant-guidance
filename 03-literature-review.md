# 03 — Literature Review: Guidance × Resolution Extrapolation

Snapshot as of 2026-10-24.

## A. Guidance (improving sample quality)

**CFG fixes and analysis**
- APG: Eliminating Oversaturation and Artifacts of High Guidance Scales (Sadat et al., ICLR 2025): [2410.02416](https://arxiv.org/abs/2410.02416)
- CFG-Zero* (2503.18886), Rectified-CFG++ (2510.07631): CFG for flow-matching models
- Stage-wise Dynamics of CFG (ICLR 2026): [2509.22007](https://arxiv.org/abs/2509.22007)
- Analytic Distribution of CFG for Schedule Design: [2607.19725](https://arxiv.org/abs/2607.19725)
- Limited-interval guidance (Kynkäänniemi et al., NeurIPS 2024)
- Rethinking Oversaturation in CFG via Low Frequency: [2506.21452](https://arxiv.org/abs/2506.21452)
- PathGuide (2608.29107), Information-Bottleneck CFG (2610.13369): dynamic CFG scales and activation
- Many ICLR 2027 submissions on CFG schedules (DG-CFG, D²-CFG, TrajCFG, PMC-CFG, ...), **none tied to resolution**

**Frequency-split guidance**
- FDG: Guidance in the Frequency Domain (Sadat et al., 2025): [2506.19713](https://arxiv.org/abs/2506.19713). Separate low and high frequency guidance scales.
- FreSca: [2504.02154](https://arxiv.org/abs/2504.02154)
- CADE 2.5 / ZeResFDG: [2510.12954](https://arxiv.org/abs/2510.12954)
- Selective Low-Frequency Recovery (SRG), ICLR 2027 submission

**Perturbation guidance**
- PAG (identity attention map), SEG: [2408.00760](https://arxiv.org/abs/2408.00760) (Gaussian-blurred attention)
- Token Perturbation Guidance ([name removed] et al., NeurIPS 2025): [2506.10036](https://arxiv.org/abs/2506.10036)
- S²-Guidance (ICLR 2026): [2508.12880](https://arxiv.org/abs/2508.12880)
- Where and How to Perturb: [2506.10978](https://arxiv.org/abs/2506.10978)
- MUG (Merging and Unmerging Guidance), ICLR 2027 submission

**Manifold view**
- Not All Prediction Targets Keep Training-Free Guidance on the Manifold (Jul 2026): [2607.00647](https://arxiv.org/abs/2607.00647). Finds x-prediction keeps samples most realistic.

## B. Resolution extrapolation (generating above training size)

**RoPE / attention fixes (DiTs)**
- DyPE (ICML 2026): [2510.20766](https://arxiv.org/abs/2510.20766). Rescales RoPE over the denoising steps; FLUX up to 16 MP.
- UltraImage: [2512.04504](https://arxiv.org/abs/2512.04504). Blames repetition on a dominant RoPE frequency; also adjusts attention concentration.
- SEGA ([name removed] et al., NeurIPS 2026): [2605.22668](https://arxiv.org/abs/2605.22668). Reweights RoPE bands per step based on spectral energy. Evaluated on FLUX at 4096² and Qwen-Image. **Uses the default guidance scales.**
- TIDE: [2603.08928](https://arxiv.org/abs/2603.08928). Attention temperature control.
- AReA ([name removed], 2610.17227): adaptive receptive-field attention.
- Cardinality-Calibrated Attention (ICLR 2027 submission): attention logit rescaling by token count.
- SigMa (ICML 2026): scale-adaptive schedule.

**Pipeline and latent methods**
- ScaleCrafter (2310.07702), FouriScale (2403.12963), DemoFusion, HiDiffusion, FreeScale (2412.09626), ScaleDiff (2510.25818), DiffuseHigh, HRDiT (2608.07003), I-Max (2410.07536)
- Guidance from a low-resolution reference: Upsample Guidance (2404.01709), HiFlow (2504.06232), ResMaster (2406.16476), TokenScale and WAVEALIGN (ICLR 2027 submissions), LookFlow (AAAI 2026)

**Video length extrapolation**
- RIFLEx (ICML 2025): lowers the intrinsic RoPE frequency
- UltraViCo (ICLR 2026): decays out-of-window attention
- Diff-VF (2608.05976): timestep-dependent guidance for long video
- No single training-free method found that handles spatial and temporal extrapolation together

## C. The gap
The two literatures haven't met:
- Extrapolation papers fix the model (RoPE, attention, latent) and **keep guidance as tuned at the training resolution**.
- Guidance papers **evaluate only at native resolution**.

Reasons to expect guidance to break at extrapolated resolution:
1. **Frequency cutoffs are tied to the grid.** FDG and FreSca split frequencies in latent-grid units, so at r× resolution the same cutoff covers a different level of image detail.
2. **Perturbation strength is tied to token count.** SEG's blur kernel is sized in tokens. PAG and TPG perturb attention or tokens whose statistics change when attention spreads over r² more tokens (the same spreading SEGA and UltraViCo fix in the model).
3. **The timing is off.** At higher resolution a given noise level destroys less structure; SD3 and FLUX already shift the timestep with resolution. Guidance intervals and schedules have never been shifted.
4. **Hypothesis:** strong low-frequency guidance at high resolution contributes to duplicated subjects and repetition.

Backup ideas if the main idea fails:
- Joint space-and-time training-free extrapolation for video DiTs (higher compute)
- Prediction target × resolution (follow-up to 2607.00647)
