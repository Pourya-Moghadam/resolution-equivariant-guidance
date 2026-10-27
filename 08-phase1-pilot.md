# 08 — Phase 1 pilot: guidance × resolution on Lumina + DyPE (2026-10-26)

First direct measurement of the kill-or-go question before the grid proper. Lumina-Image 2.0, dynamic NTK (DyPE) above 1K, 30 steps, seed 0, two prompts (fox, bicycle), CFG ∈ {2, 3, 4, 5, 6, 8} × {1024², 2048², 3072², 4096²}: 48 images, `jobs/pilot_cfg_sweep.jsonl`, run across reg1/3/4/5 in ~1 h. Per-image scores in `results/pilot_cfg_sweep.csv`; images in `$REG_DATA/outputs/pilot_cfg_sweep/`.

## Scores (pixel std = contrast/content; centre-crop Laplacian = fine-scale energy, incl. speckle)

| fox | CFG 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|
| 1K | 57 / 4.8 | 59 / 4.6 | 60 / 4.6 | 59 / 4.8 | 60 / 5.0 | 61 / 5.3 |
| 2K | 47 / 13.5 | 51 / 13.9 | 52 / 13.7 | 53 / 14.0 | 54 / 14.2 | 54 / 14.4 |
| 3K | **22 / 3.6** | 34 / 8.1 | 40 / 12.6 | 44 / 16.5 | 47 / 19.9 | 49 / 23.4 |
| 4K | **15 / 3.3** | **19 / 2.9** | 26 / 8.4 | 33 / 15.5 | 36 / 20.3 | 40 / 23.4 |

| bicycle | CFG 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|
| 1K | 53 / 44 | 60 / 41 | 65 / 38 | 67 / 36 | 70 / 34 | 70 / 31 |
| 2K | 48 / 51 | 55 / 43 | 58 / 38 | 60 / 35 | 62 / 32 | 64 / 33 |
| 3K | **29 / 33** | 38 / 39 | 44 / 44 | 48 / 46 | 50 / 47 | 53 / 48 |
| 4K | **19 / 51** | 31 / 55 | 46 / 55 | 49 / 58 | 54 / 57 | 49 / 58 |

Bold = visually collapsed (flat colour or ghost). Vanilla 4K reference: std 18 / lap 2.7.

## What the images show
- **1K (native):** every CFG 2–8 gives a good image; at CFG 8 the bicycle is over-guided (saturated, a second bike appears). Native optimum ≈ 3–4, as expected.
- **2K:** all cells fine; CFG 2 slightly washed out.
- **3K:** CFG 2 is a ghost of the subject, CFG 3 washed out, CFG 4 acceptable, 5–8 progressively better — no over-guidance visible even at 8.
- **4K:** CFG 2 and 3 flat, CFG 4 a small speckled fox / a usable bicycle, 5 → 8 progressively better; CFG 8 is the best 4K cell for both prompts, with speckle remaining in the sky/wall.
- Contrast (pixel std) is flat in CFG at 1K and climbs steeply with CFG at 3K–4K: the model needs much more guidance to produce the same amount of image at 4× the tokens.

## Interpretation
1. **The usable guidance range shifts up with resolution**: lower edge ≈ 2 at 1K, ≈ 2–3 at 2K, ≈ 4 at 3K, ≈ 5 at 4K. **The optimum shifts up too**: ~3–4 at native vs ≥ 8 at 4K (still rising at the top of the sweep). The native and 4K optima do not overlap. This is the direction the plan's kill-or-go gate asks for (PLAN.md Phase 1: "best settings move with resolution").
2. Mechanistically consistent with 07's probes: at 4× the conditional signal per token is diluted; both the YaRN attention temperature and the guidance magnitude have to be raised for the signal to beat the noise over 65K tokens. REG's components (B: band-wise schedule) and (D: token-count calibration) target exactly this.
3. The 4K speckle is a separate phenomenon (present at every CFG ≥ 4, absent at 3K), to be handled in the extrapolation port, not by guidance.

## Consequences for the kill-or-go grid (PLAN Phase 1, step 3)
- Sweep CFG to at least 10–12 at 3K/4K; the optimum is above 8.
- 3K is a good "high" resolution for Lumina (clean, 4.9 min/image, 2.4× cheaper than 4K) — use 1K/2K/3K for the quantitative grid and 4K for a reduced set, until the speckle is understood.
- Two prompts are enough to see the effect, not to measure it: `jobs/pilot_cfg_sweep_b.jsonl` (6 varied prompts × 2 seeds × same sweep, 288 rows, ~22 GPU-h) runs next; the quantitative verdict needs the step-7 metrics (patch-FID/KID, CLIP, repetition count) on ≥100 prompts.
- Band-wise behaviour (component B's specific prediction, low-band guidance switching off earlier) is not visible from scalar CFG; it needs the guidance-spectrum diagnostic (`reg/diag/`), Phase 2.
