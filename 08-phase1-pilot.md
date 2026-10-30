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
- Two prompts are enough to see the effect, not to measure it: `jobs/pilot_cfg_sweep_b.jsonl` (6 varied prompts × 2 seeds × same sweep, 288 rows, ~22 GPU-h) measured it — see "Sweep B" below; the gate itself needs ≥100 prompts and the other guidance/extrapolation methods.
- Band-wise behaviour (component B's specific prediction, low-band guidance switching off earlier) is not visible from scalar CFG; it needs the guidance-spectrum diagnostic (`reg/diag/`), Phase 2.

## Metrics on the pilot (first run of `scripts/run_eval.py`, 2026-10-26 22:20)
CLIP (OpenCLIP ViT-L/14) and patch-KID against 200 Aesthetic-4K 2K photos (3,200 reference crops); `results/pilot_cfg_sweep_eval.csv`. Two images per cell, so only the large effects count:

| | CLIP global, CFG 2 → 8 | patch-KID, CFG 2 → 8 |
|---|---|---|
| 1K | 35.7 → 36.4 (flat, peak 37.5 at 6) | 0.029 → 0.018 |
| 2K | 37.0 → 36.1 (flat) | 0.108 → 0.117 (flat) |
| 3K | **28.4** → 38.3 (monotone up) | **0.170** → 0.078 (monotone down) |
| 4K | **8.8** → 35.2 (collapse → image at CFG ≥ 4) | **0.398** → 0.146 (best 0.120 at CFG 6) |

- The metrics reproduce the visual verdict: at 1K/2K nothing depends on CFG; at 3K/4K CLIP rises and KID falls with CFG, with the collapses at CFG 2–3 standing out by an order of magnitude. The 4K optimum by KID is CFG 6, by CLIP CFG 8; the 3K optimum is at the top of the sweep for both. Supports extending the grid to CFG 10–12 (PLAN step 3).
- **Methodology notes for the grid analysis.** (i) Patch-FID is meaningless at this sample size (~330–470 everywhere); KID (unbiased) is the set metric until there are hundreds of images per cell. (ii) Compare settings *within* a resolution: 1K images enter as whole frames (one crop), higher resolutions as 16 crops of 1024², so absolute values across resolutions are not comparable. (iii) `clip_patch` (mean over crops) drops from ~33 at 2K to ~15 at 4K even for good images — a 1024² crop of a 4K image shows a fraction of the scene, so it measures local agreement, not failure; use it only within resolution too.

## Sweep B: 6 prompts × 2 seeds (2026-10-27, `jobs/pilot_cfg_sweep_b.jsonl`)
Same sweep (CFG {2,3,4,5,6,8} × {1K, 2K, 3K, 4K}, Lumina + dy_ntk above 1K, 30 steps) on six varied prompts (portrait, street, cathedral, still life, mountain, lighthouse) × seeds {0, 1}: 288 rows, 21.6 GPU-h, ran overnight on reg1/3/4/5 (5.7 h wall; reg2 held by another user's kernel). Measured per image: 16 s / 94 s / 282 s / 692 s at 1K/2K/3K/4K. Metrics: `results/pilot_cfg_sweep_b_eval.csv` (12 images per cell; CLIP ViT-L/14 global; patch-KID vs 200 Aesthetic-4K 2K photos). Contact sheets: `$REG_DATA/outputs/pilot_cfg_sweep_b/sheets/<prompt>_seed{0,1}.jpg` (`scripts/contact_sheet.py`).

| CLIP global (± s.e.) | CFG 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|
| 1K | 35.5 ±0.5 | 36.1 | 36.0 | 36.1 | 35.9 | 36.0 ±0.5 |
| 2K | 35.6 ±0.5 | 35.6 | 36.1 | 36.1 | 36.1 | 36.3 ±0.5 |
| 3K | **20.9 ±3.1** | 34.6 ±0.8 | 35.4 | 35.6 | 35.7 | 35.7 ±0.5 |
| 4K | **5.9 ±0.7** | **12.0 ±2.0** | 25.8 ±3.0 | 30.9 ±1.7 | 33.4 ±1.5 | 34.3 ±0.6 |

| patch-KID (± std over subsets; lower is better) | CFG 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|
| 1K (whole frames) | 0.040 | 0.036 | 0.034 | 0.034 | 0.033 | 0.036 |
| 2K | **0.031** | 0.037 | 0.040 | 0.040 | 0.038 | 0.035 |
| 3K | **0.125** | 0.038 | 0.030 | 0.025 | 0.023 | **0.022** |
| 4K | **0.282** | **0.243** | 0.151 | 0.096 | 0.060 | **0.039** |

(KID std ≈ 0.002–0.006 at ≥2K, ≈ 0.012 at 1K where each image is a single crop.)

**Result.** With 12 images per cell the pilot's picture is now statistically clean:
1. **1K and 2K: no CFG dependence.** CLIP is flat within one s.e. from 2 to 8; KID is flat at 1K and *lowest at CFG 2* at 2K (0.031 vs 0.035–0.040). The native model is indifferent to guidance over this range, if anything preferring the low end.
2. **3K: the lower edge moves to ≈3 and the optimum to ≥6.** CFG 2 collapses on all six prompts (CLIP 21, KID 0.125); CFG 3 is already acceptable by CLIP but KID keeps falling monotonically to CFG 8 (0.038 → 0.022). Per prompt, the CLIP optimum is at 6 or 8 for all six.
3. **4K: the lower edge moves to ≈4–5 and the optimum is at the top of the sweep.** CFG 2 and 3 collapse everywhere (CLIP 6 / 12, KID 0.28 / 0.24); CFG 4 is a mixture (good on portrait/stilllife/street, collapsed on cathedral/lighthouse); both metrics improve monotonically to 8 (KID 0.039 at 8, still falling). Per prompt, CLIP's optimum is 8 on four prompts and 6 on two.
4. So the guidance optimum goes from "anything in 2–8, slightly favouring 2" at native to "≥ 8" at 4× the side length — the native and extrapolated optima do not overlap, and the shift is monotone in resolution (2K ≈ native, 3K intermediate, 4K extreme). **This is the kill-or-go direction, measured.** What the gate still needs is the proper grid (≥100 prompts, CFG up to 12, the other guidance methods and extrapolation methods), since a shift that one global scalar can absorb is not yet a result — the question is whether a resolution-dependent *schedule* (REG) beats the best resolution-dependent scalar.

**Qualitative notes from the sheets.**
- The 4K speckle is present at every CFG ≥ 4 on every prompt; on the cathedral (dark, high-frequency interior) it already appears at 3K from CFG 4 — "3K clean" in 07 was prompt-dependent. It is unrelated to the collapse (which happens at low CFG) and still has to be fixed in the extrapolation port.
- At 4K the **composition drifts**: the portrait becomes a tight face crop, the lighthouse scene is mostly sea until CFG 6–8, the street loses its figure. The dynamic-NTK port preserves local texture but not the global layout the 1K/2K model produces. Relevant to component (C) (native anchor) and to the choice of repetition/layout metrics.
- Over-guidance is visible at 1K/CFG 8 (saturated still life, street) but the metrics do not penalise it; CLIP rewards saturation. KID at 1K is a single-crop measurement and too noisy (std 0.012) to see it. Human/aesthetic preference (ImageReward/HPSv2, parked) would be needed to catch it; for the grid, treat CLIP + KID as a floor detector, not an optimum detector, at native resolution.

**Methodology confirmed.** KID with 12 images × 16 crops has std ≈ 0.003, enough to separate the 3K cells (0.038 → 0.022) — so the grid's 100 prompts/cell is ample for KID. Patch-FID (205–394) is still dominated by sample-size bias at n = 192 patches; report it only in Phase 3 with ≥1,000 images per cell. `clip_patch` tracks `clip` within each resolution and adds nothing — drop it from grid tables.

## Sweep C: CFG 10 / 12 / 16 at 3K and 4K (2026-10-27, `jobs/pilot_cfg_sweep_c.jsonl`)
Same six prompts × 2 seeds, 72 rows, 9.7 GPU-h. `results/pilot_cfg_sweep_c_eval.csv`; sheets in `$REG_DATA/outputs/pilot_cfg_sweep_c/sheets/`.

| | CFG 8 (sweep B) | 10 | 12 | 16 |
|---|---|---|---|---|
| 3K CLIP | **35.7** | 35.4 | 35.2 | 34.6 |
| 3K KID | 0.022 | 0.023 | 0.022 | **0.020** |
| 4K CLIP | 34.3 | **34.6** | 33.9 | 33.5 |
| 4K KID | 0.039 | 0.030 | 0.031 | **0.026** |

- **The CLIP optimum is at CFG 6–8 (3K) and 8–10 (4K) and falls beyond it**; by eye CFG 12–16 is over-guided at both resolutions (saturated colours, a second ghost lighthouse at 4K, black blotches on the 4K street, grain on flat 3K backgrounds). Together with sweep B: native optimum anywhere in 2–8 (flat), 3K ≈ 6–8, 4K ≈ 8–10, i.e. **the optimum moves up by roughly the resolution ratio and then turns over**. The kill-or-go grid's CFG axis {2 … 12, 16} at 3K/4K brackets it.
- **KID keeps falling to CFG 16 at both resolutions while the images get visibly worse.** Inception features on 1024² crops reward the extra local contrast that over-guidance adds. For the grid: KID detects collapse and under-guidance well, but its optimum is biased upward; take CLIP (and the repetition count) as the optimum detectors and KID as the floor detector, as noted for 1K above. A human-preference metric (HPSv2/ImageReward in a separate venv) would resolve this and should be added before Phase 3.
- The 4K speckle changes character with CFG (bright grain at 8–10, dark blotches at 12–16) but is never absent; it stays an extrapolation-port issue (07).

## Guidance zoo at 1K (2026-10-27, `jobs/guidance_zoo_1k.jsonl`, 84 rows)
Phase 1 step 2 check: six prompts, native 1K, 14 settings (`$REG_DATA/outputs/guidance_zoo_1k/sheets/zoo_<prompt>.jpg`). The sampler is bit-identical to the diffusers pipeline for CFG (`tests/test_guidance_real.py`), so differences are the methods', not the loop's.

- **CFG family behaves as published.** Truncation at 0.7 and the late-start interval (CFG 6 on 0.15–0.8) are indistinguishable from CFG 4 at a glance and 15–20 % cheaper (13 s vs 15 s). **FDG** reproduces the paper's claim: w_low = 1 / w_high = 4 gives a softer, less saturated image with the fine detail of CFG 4 (low-band guidance is what drives saturation); 2/6 and 4/8 move toward plain CFG. Cost identical to CFG.
- **PAG, SEG and TPG at my first defaults are over-perturbed.** Default = perturb the middle quarter of the blocks (12–19 of 32) at scale 3: PAG alone collapses to a grey stipple (the street becomes a tunnel of texture), and PAG/SEG/TPG + CFG 4 all converge on the same hard-edged, oversaturated "vector-illustration" look. The three methods agreeing points at the shared cause — too many perturbed blocks make Δ_pert enormous — rather than at any one implementation. TPG with only 10 % of the tokens shuffled is the mildest, consistent with that. The diffusers DiT conventions (PixArt `blocks.14`, SD3 `blocks.13`) perturb **one** middle block. Perturbation methods cost 3 passes: 23 s vs 15 s per 1K image.
- **Calibration running** (`jobs/pert_calib_1k.jsonl`, 26 settings × 6 prompts, ~1 GPU-h on reg3 next to the grid): layers {16} / {15,16,17} / {12…19} × scale {1, 2, 3} for PAG; layers × σ {2, 5, 10} (+ scale 1.5) for SEG; layers × frac {0.1, 0.5, 1} (+ scale 1.5) for TPG. Target: the papers' qualitative behaviour — cleaner structure than CFG alone without the illustration look. The defaults in `reg/guidance/perturb.py` will be set from it; the PAG/SEG/TPG grid (PLAN step 3, second job) waits for that.

### Perturbation calibration result (2026-10-27, 156 rows, all with CFG 4 unless "alone")
Sheets: `$REG_DATA/outputs/pert_calib_1k/sheets/calib_{pag,seg,tpg}.jpg` (6 prompts × settings).

- **Eight blocks is the problem, for all three methods.** PAG on blocks 12–19 desaturates and flattens at every scale, even scale 1 (grey water, smooth rock, washed-out street). On three or one block, no setting breaks the image.
- **One middle block is close to plain CFG** at scale ≤ 3. It is safe but barely visible, so it would make a weak baseline.
- **Three middle blocks (15–17) give the effect the papers describe.** PAG at scale 2–3 adds local contrast and texture. SEG and TPG at scale 3 overshoot: speckled pears, striped cathedral floor, oversharpened portrait. At scale 1.5 they are clean and still visibly different from CFG.
- **New defaults** in `reg/guidance/perturb.py`: layers 15–17, SEG σ = 5, TPG frac = 0.5. The usable scale is about 1.5–3 at these layers. The second grid sweeps scale {1, 1.5, 2, 3} at 1K and up, so no single scale default is baked in.

## REG pilot: FDG band cutoff at 4K (2026-10-27, reg2, `jobs/reg_pilot_fdg.jsonl`, early look at 3 prompts)
Question: does scaling FDG's band cutoff by the resolution ratio (down 2 at 1K → 8 at 4K, REG-A) beat the fixed cutoff? Baselines: sweep B CFG 4/8 at 4K, same prompts and seed. Sheets: `$REG_DATA/outputs/reg_pilot_fdg/sheets/early_{full,crop}.jpg`.

- **The weight of the low band decides 4K, not the cutoff.** FDG w_low/w_high = 2/6 and 4/8 collapse into speckle at 4K at both cutoffs, like CFG 4. Only 8/4 is coherent, at quality comparable to CFG 8. Raising the high band to 8 does not save the image, and dropping it to 4 does not hurt.
- **So the upward shift of the CFG optimum with resolution is a low-frequency effect.** At 4K the high band is satisfied with native-level guidance. This is direct support for band-wise guidance (05), and an early hint that "CFG 8 at 4K" over-guides the high band.
- **Scaling the cutoff did not help yet.** At 8/4, down 8 is close to down 2, with slightly less saturated windows. Two prompts only.
- Follow-up queued on reg2 after this job (`jobs/reg_pilot_fdg_b.jsonl`, 36 rows): w_low {6, 8, 12} × w_high {2, 4, 6} around 8/4, and cutoff {2, 4, 8}.

### REG pilot stage A, all 6 prompts, scored (2026-10-27 evening)
CLIP and patch-KID, 6 prompts at seed 0, references are 400 Aesthetic-4K 2K images. The CFG baselines are re-scored on the same references (`results/reg_pilot_cfg4k_baseline_eval.csv`). FDG: `results/reg_pilot_fdg_eval.csv`. The 4K "low band" is below 1/d of the latent grid. At d = 8 that is the same image-relative band as d = 2 at 1K.

| 4K setting | CLIP | ± SE | patch-KID |
|---|---|---|---|
| CFG 4 | 25.0 | 4.2 | 0.158 |
| CFG 6 | 34.6 | 0.8 | 0.053 |
| CFG 8 | 34.2 | 0.5 | 0.040 |
| CFG 10 | 34.5 | 0.3 | 0.028 |
| FDG 2/6, d2 / d8 | 7.1 / 9.6 | | 0.277 / 0.247 |
| FDG 4/8, d2 / d8 | 25.4 / 27.8 | | 0.142 / 0.094 |
| FDG 8/4, d2 | 33.9 | 0.6 | 0.063 |
| FDG 8/4, d8 | 33.6 | 1.1 | 0.079 |

- **The coarsest image-relative band is the one that needs more guidance at 4K.** FDG 8/4 with d8 guides only the band below 1/8 of the grid at 8 and everything else at 4, and it is coherent. FDG 4/8 with d8 is the mirror image, with the coarse band at 4 and the rest at 8, and it collapses. So the extra guidance 4K needs sits in the band that corresponds to the whole native-resolution image. This is the REG-A/anchor picture in 05, measured.
- **FDG 8/4 is level with CFG 6 to 8 on CLIP, not better.** KID favours CFG 8 to 10, but KID is biased toward over-guidance (sweep C). With 6 images the difference is inside the noise. REG has to beat retuned CFG on a human-preference metric or on a larger set before it is a result.
- Stage B (36 rows, running overnight on reg2) tests whether the high band can drop to 2 when the coarse band is at 8, and maps w_low {6, 8, 12}.

### REG pilot stage B, first 17 rows by eye (2026-10-27 21:00; sheets `stageB_{full,crop}.jpg`)
- **The fine bands do not matter at 4K.** FDG 8/2 is almost indistinguishable from 8/4 and from CFG 8 on all six prompts. Guidance on the fine bands can range from 2 to 8 with no visible change. The coarse band alone decides the outcome.
- **The coarse band has a window around 6 to 10.** At 6/4 the images are coherent but weaker: the cathedral windows lose colour and the lighthouse shrinks. At 12/4 the composition changes and starts repeating: a second lighthouse appears, the still life is rearranged, and the street turns into heavy speckle. This is the same turnover as CFG 12 to 16 in sweep C, now located in the coarse band.
- **No guidance setting removes the 4K speckle**, with CFG 8 included (see the portrait crops). The speckle is an extrapolation artifact (07) and not REG's problem. The paper should not claim to fix it.
- **Consequence for REG.** Lowering fine-band guidance buys no visible quality over CFG 8, so "same coherence with less over-sharpening" is not supported at 4K. What remains is the mechanism: the resolution-dependent part of guidance is the coarse image-relative band. A REG rule that sets that band from the resolution ratio, without a sweep, is the claim to test next.

### REG pilot stage B scored (2026-10-28 00:40; `results/reg_pilot_fdg_eval.csv`, 6 prompts, 4K)
| FDG w_low/w_high, cutoff | CLIP | ± SE | patch-KID |
|---|---|---|---|
| 6/4, d2 | 34.1 | 0.8 | 0.071 |
| 8/4, d2 / d4 / d8 | 33.9 / 33.6 / 33.6 | | 0.063 / 0.072 / 0.079 |
| 8/6, d2 | 34.0 | 0.6 | **0.047** |
| 8/2, d2 | 33.6 | 0.7 | 0.082 |
| 8/2, d8 | 31.0 | 1.4 | 0.167 |
| 12/4, d2 | 33.7 | 0.2 | 0.048 |
| CFG 6 / 8 / 10 (ref.) | 34.6 / 34.2 / 34.5 | | 0.053 / 0.040 / 0.028 |

- **Three regimes in cycles per image at 4K** (latent side 512; cutoff d ↔ ≈ 256/d cycles/image): the band below ≈ 32 c/i (the whole native image; d8) needs ≈ 8 and collapses at 4; the band 32–128 c/i (native fine detail; between d8 and d2) needs ≥ 4 (8/2 at d8 fails, 8/4 at d8 works); the band above 128 c/i (detail that does not exist at native) is indifferent between 2 and 8 (8/2 at d2 is level with 8/4). At 1K the band below 32 c/i was happy at 2 (FDG 2/6 in the zoo). **So the band that changes its requirement with resolution is exactly the native-image band, by a factor ≈ 2–4 between 1K and 4K.** The diagnostics job (`jobs/reg_diag.jsonl`, running) measures whether the guidance signal in that band shrinks by the same factor.
- No FDG setting beats CFG 8–10 on these metrics; 8/6 ties CFG 8 on KID with slightly lower CLIP. The quality claim for REG must come from a parameter-free rule, not from a better 4K setting.

## Guidance diagnostics (2026-10-28 04:00, `jobs/reg_diag.jsonl`, 36 rows, `results/reg_diag.{csv,png}`)
Per-step ‖Δ_low‖/‖v_cond,low‖ at six cutoffs, 6 prompts, CFG 4 at 1K–4K and CFG 8 at 3K/4K. Compared step by step (same sigma schedule at every resolution), native cutoff d2 at 1K against the same cutoff in cycles per image at ratio r (d = 2r).

| | median ratio, native-image band | (B′) predicted w_low | measured window |
|---|---|---|---|
| 1K | 0.040 | 4 (def.) | 2–8 flat |
| 2K | 0.035 | 5.3 | flat |
| 3K | 0.034 | **3.8** | 6–8 |
| 4K | 0.045 | **4.8** | 6–10 |

- **(B′) magnitude matching is refuted.** The relative size of the coarse-band guidance signal is preserved under extrapolation (it even grows slightly at 4K), so "the native weight under-guides because Δ_low shrank" is wrong. The step-matched prediction lands at 4–5 where 8 is needed. What degrades at 4K is the *direction* or informativeness of Δ_low, not its size: the model's coarse-band conditional/unconditional difference is less useful per unit norm, and a larger weight partly compensates.
- **Spectrum drift (paper Figure 2 material):** fraction of Δ power above 64 cycles/image (beyond the native Nyquist) is < 1 % at 1K and 17 % at 4K; the share in the coarsest octaves (0–4 c/i) rises from 0.37 to 0.48. The guidance signal at 4K spends a sixth of its energy on frequencies the model never guided during training — the speckle band.
- **Decision (standing instruction: decide and proceed).** Per the 05 revision, component **(C), the native-resolution anchor**, becomes the lead method: compute the coarse-band guidance from the model queried at native resolution on the area-downsampled latent at its derived noise level, where Δ is in-distribution, and keep the native weight. If native-weight anchored guidance gives coherent 4K images where CFG 4 collapses, the paper's method claim is "native guidance extrapolates once it is evaluated where the model is native", with no retuning. The empirical rule w_low ≈ w_native·√r (2K 5.7, 3K 6.9, 4K 8.0, all inside the measured windows) is kept as the one-line baseline REG-lite, to be tested against the grid's optimum curve. Implementation and a 2K/3K/4K pilot on reg2 follow now.

## REG (C) anchor pilot (2026-10-28 08:15, `jobs/reg_anchor_pilot.jsonl`, 30 rows, `results/reg_anchor_pilot_eval.csv`)
Anchored CFG: the sub-native band of Δ from cond/uncond passes at native resolution on the area-downsampled latent at its derived noise level (`reg/guidance/anchor.py`); native weight. Sheets `$REG_DATA/outputs/reg_anchor_pilot/sheets/anchor_{full,crop}.jpg`. Overhead: 108 / 294 / 693 s per image at 2K / 3K / 4K, i.e. none measurable (1/r² tokens per anchor pass).

| | CLIP | patch-KID | by eye |
|---|---|---|---|
| anchor w4, 2K | 36.5 | 0.039 | = CFG 4 |
| anchor w4, 3K | 34.2 | 0.040 | coherent, dim and low-contrast; CFG 4 at 3K is brighter |
| anchor w2 / w4 / w6, 4K | 6.8 / 23.9 / 26.7 | 0.299 / 0.219 / 0.214 | all collapse; worse than CFG 4 (25.0 / 0.158) |

- **Anchoring the guidance *difference* does not rescue 4K.** The native weight still collapses, and weight 6 too. So the coarse band of Δ is not what goes wrong, consistent with the diagnostics (Δ_low keeps its relative size).
- **Reading:** guidance amplifies the difference between branches; it cannot remove an error the branches share. If the *base* prediction's coarse band is off-manifold at 4K (a shared component), weight 8 works only because the clean difference then dominates it. That predicts: anchoring the base's coarse band as well (`anchor: {"base": true}`, derivation in `reg/guidance/anchor.py`) should make the native weight work. Pilot queued on reg2 (4K w2/w4, 3K w4).
- If that also fails, the resolution dependence is not a coarse-band phenomenon the native model can supply, and the paper's method becomes the measured weight law (coarse band ∝ √r) with the diagnostics as the explanation.

### Base-anchored variant, first image (2026-10-28 09:00, `anchor: {"base": true}`, cathedral, 4K, w = 4)
`sheets/first_base_cathedral.jpg`: CFG 4 (speckle collapse) | CFG 8 | anchor-Δ w4 (mush) | **anchor-base w4: a coherent nave, stained glass, pews, at the native weight**. The 4K speckle (> native Nyquist) is still there, as expected — the anchor supplies nothing above 64 c/i. One prompt; the other 17 rows (4K w4/w2, 3K w4) land ~11:30. If they agree: **the resolution dependence of the guidance optimum is a symptom of a shared coarse-band error in the base prediction, not of guidance itself** — and it disappears when the sub-native band comes from where the model is native. Framing consequence to settle before writing: anchoring the base's sub-native band is a prediction-space anchor, closer to FreCaS/HiFlow/ResMaster than the Δ-only anchor 05 proposed; the novelty then rests on the finding (why guidance optima shift: the band dissection, the diagnostics, the Δ-only null result) and on deriving the anchor's noise level and weight, not on anchoring per se. Re-read 03/04 on those methods before Gate 1.

### Base-anchored CFG at 4K, all six prompts (2026-10-28 10:20; `sheets/base4k_{full,crop}.jpg`, `results/reg_anchor_pilot_eval.csv`)
| 4K, 6 prompts | CLIP | ± SE | patch-KID |
|---|---|---|---|
| CFG 4 | 25.0 | 4.2 | 0.158 |
| CFG 8 | 34.2 | 0.5 | 0.040 |
| anchor-Δ w4 | 23.9 | 4.3 | 0.219 |
| **anchor-base w4** | **33.4** | 0.5 | **0.051** |
| anchor-base w2 (3 prompts so far) | 30.9 | 0.9 | 0.105 |

- **Confirmed on 6/6 prompts: with the base's sub-native band anchored, the native weight 4 gives coherent 4K images.** Metrics are level with CFG 8 (CLIP −0.8, KID +0.01; n = 6, inside noise) and far from CFG 4. By eye the anchored images keep the native composition (the lighthouse on its rock, the full portrait framing, the figure in the street) where CFG 8 drifts; the 4K speckle is unchanged.
- **Weight 2 is degraded but not collapsed** at 4K with the anchor, as it is at 1K — the anchored band behaves natively. Whether the anchored optimum sits at ≈ 4 (native) rather than 8 is what stage C (w6/w8 at 4K, w2/w8 at 3K) answers this afternoon; CLIP/KID will not settle a 4-vs-6 difference at n = 6, so a human-preference metric (HPSv2 / ImageReward, separate venv) is being set up now.

### Human-preference scores (2026-10-28 11:00; HPSv2.1 ×100 / ImageReward, `scripts/run_pref.py`, 6 prompts, seed 0, 4K)
| 4K | HPS | ± SE | IR | ± SE |
|---|---|---|---|---|
| CFG 4 / 6 / 8 / 10 | 16.6 / 22.7 / 25.4 / 27.0 | 2.4 / 1.7 / 1.3 / 0.7 | −0.79 / 0.79 / 1.07 / 1.30 | |
| FDG best (12/4, d2) | 26.5 | 0.6 | 1.28 | 0.26 |
| anchor-Δ w4 | 12.4 | 1.6 | −1.71 | |
| **anchor-base w4** | **29.6** | 0.9 | **1.43** | 0.24 |
| anchor-base w2 (n = 3) | 26.1 | 0.4 | 1.02 | 0.40 |
| 1K reference (FDG/CFG 4) | 32.4–32.9 | 0.4 | 1.52–1.59 | |

- **First quality win:** base-anchored CFG at the *native* weight beats every plain-CFG setting at 4K on both preference metrics, including the CLIP/KID-preferred CFG 10 (HPS +2.6, about 3 SE; IR +0.13). CLIP/KID had called it level with CFG 8 — the preference metrics see the composition and over-guidance that CLIP/KID miss (sweep C's KID bias, confirmed).
- The gap to the 1K reference (HPS 32.5 vs 29.6) is the speckle and whatever else 4K extrapolation costs; the anchor closes most of the guidance part of the gap (CFG 4 → anchored 4: 16.6 → 29.6).
- Stage C (anchored w6/w8 at 4K, w2/w8 at 3K) will say whether 4 is the anchored optimum; HPS/IR are now the metrics for that call.

Plain-CFG preference baselines (`results/pilot_cfg_baseline_pref.csv`, seed 0, 6 prompts): HPS rises with CFG at every resolution — 1K 31.7/32.5/33.1 (CFG 2/4/8), 2K 29.0/31.2/31.6/32.1 (2/4/6/8), 3K 11.8/26.3/28.3/29.8/30.6 (2/4/6/8/10), 4K 16.6/22.7/25.4/27.0 (4/6/8/10). IR is flat from CFG 4 up at 1K–3K (1.5) and still rising at 4K. **So HPS, like KID, favours stronger guidance and will not locate an optimum on its own**; the clean statements are at matched weight (anchored w4 at 4K: HPS 29.6 vs CFG 4's 16.6, and above CFG 10's 27.0) and against the best weight. "The optimum returns to native with the anchor" has to be argued from the collapse boundary (w2 survives with the anchor) and from matched-weight comparisons, not from a metric maximum.

### Stage B complete: base-anchored CFG at 3K and 4K, all six prompts (2026-10-28 11:15; `sheets/base3k_{full,crop}.jpg`, `sheets/base4k_w2_{full,crop}.jpg`; `results/reg_anchor_pilot_{eval,pref}.csv`)
| seed 0, 6 prompts | CLIP | patch-KID | HPS | ± SE | IR |
|---|---|---|---|---|---|
| 3K CFG 2 / 4 / 6 / 8 / 10 | 20.9 / 35.4 / 35.7 / 35.7 / 35.4 | 0.125 / 0.030 / 0.023 / 0.022 / 0.023 | 11.8 / 26.3 / 28.3 / 29.8 / 30.6 | 2.5 / 1.8 / 1.4 / 1.3 / 1.3 | −1.20 / 1.20 / 1.47 / 1.50 / 1.45 |
| 3K anchor-Δ w4 | 34.2 | 0.040 | 22.6 | 1.7 | 0.87 |
| **3K anchor-base w4** | 35.2 | **0.024** | **31.0** | 0.7 | **1.51** |
| 4K CFG 4 / 8 / 10 | 25.0 / 34.2 / 34.6 | 0.158 / 0.040 / 0.030 | 16.6 / 25.4 / 27.0 | 2.4 / 1.3 / 0.7 | −0.79 / 1.07 / 1.30 |
| 4K anchor-base w2 | 32.3 | 0.099 | 27.2 | 1.1 | 1.33 |
| **4K anchor-base w4** | 33.4 | 0.051 | **29.6** | 0.9 | **1.43** |

- **3K agrees with 4K.** At the native weight the base-anchored CFG matches the best plain-CFG setting at 3K on every metric (HPS 31.0 vs CFG 10's 30.6, KID 0.024 vs 0.022, IR 1.51 vs 1.50) and beats CFG 4 by 4.7 HPS (about 2.5 SE). The Δ-only anchor at 3K is below CFG 4 on HPS (22.6), so the 3K result also separates the two variants: it is the base's coarse band that matters, not Δ's.
- **The collapse boundary moves back to native.** With the anchor, w2 at 4K gives whole images (HPS 27.2, level with plain CFG 10) where plain CFG 2 at 3K/4K is noise and CFG 4 at 4K is speckle. At 1K, CFG 2 is the mildly under-guided end of the usable range (HPS 31.7 vs 32.5); anchored 4K w2 sits the same way relative to anchored w4 (27.2 vs 29.6). That is the "guidance stays native" prediction at the low end; stage C (w6/w8) tests the high end this afternoon.
- **By eye (3K sheet):** anchored w4 keeps the composition and the contrast; versus CFG 4 at 3K it is brighter and the subjects are complete (the mountain gets its peak and sky, the street its signage and figure). Versus CFG 8/10 the anchored images are similar in contrast and slightly different in layout, with no over-saturation. The 4K sheet: anchored w2 and w4 keep the full framing (whole portrait with window, lighthouse on its headland, cathedral nave with pews) where CFG 8 crops tight or drifts; the above-native speckle is identical across columns, as it must be.
- **Cost:** none measurable (anchor passes are 1/r² of the tokens): 294 s at 3K and 695 s at 4K per image, the same as plain CFG.
- Standing interpretation unchanged: resolution dependence of the guidance optimum is a symptom of the base prediction's sub-native band being off-manifold under extrapolation; supplying that band from the native model restores the native guidance weight at 3K and 4K. Stage C and the 150-row `jobs/reg_anchor_grid.jsonl` on the kill-or-go prompts (3K/4K, anchored w4, same prompts and seeds as the grid, so the comparison with the grid's CFG optimum is paired) follow on reg2.

### Stage C: the anchored weight curve, 3K and 4K (2026-10-28 14:45; `sheets/stageC_{3k,4k}_{full,crop}.jpg`; `results/reg_anchor_pilot_{eval,pref}.csv`; figure `results/anchor_weight_curve.png`, `scripts/anchor_weight_curve.py`)
HPSv2.1 (×100) by CFG weight, 6 prompts, seed 0, ± SE ≈ 0.3–1.3 at 1K/2K/anchored and 1.3–2.5 for plain 3K/4K:

| weight w | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| 1K plain CFG | 31.7 | 32.5 | – | 33.1 | – |
| 2K plain CFG | 29.0 | 31.2 | 31.6 | 32.1 | – |
| 3K plain CFG | 11.8 | 26.3 | 28.3 | 29.8 | 30.6 |
| **3K anchor-base** | **29.3** | **31.0** | – | **32.1** | – |
| 4K plain CFG | (noise) | 16.6 | 22.7 | 25.4 | 27.0 |
| **4K anchor-base** | **27.2** | **29.6** | **30.5** | **31.2** | – |

ImageReward: 3K anchored 1.44 / 1.51 / 1.56 at w2/4/8 (plain 3K −1.20 / 1.20 / 1.50); 4K anchored 1.33 / 1.43 / 1.51 / 1.54 at w2/4/6/8 (plain 4K −0.79 / 0.79 / 1.07 / 1.30 at w4/6/8/10). Patch-KID: 3K anchored 0.025 / 0.024 / 0.024 (plain 0.125 / 0.030 / 0.022 at w2/4/8); 4K anchored 0.099 / 0.051 / 0.038 / 0.030 (plain 0.158 / 0.053 / 0.040 / 0.028 at w4/6/8/10). CLIP is flat above w4 for both. The Δ-only anchor at 4K is below plain CFG at every weight (HPS 6.5 / 12.4 / 15.1 at w2/4/6, IR < −1.2): anchoring only the guidance difference does harm, so the base's coarse band is the whole effect.

- **The anchored weight curve is the native one.** The 3K anchored curve coincides with the 2K plain curve at every weight (29.3 / 31.0 / 32.1 vs 29.0 / 31.2 / 32.1), and the 4K anchored curve sits 1.5–2 HPS below it. Plain 3K/4K CFG is 5–18 HPS below at matched weights. The weight sensitivity from w4 to w8 is 0.6 (1K), 0.8 (2K), 1.1 (3K anchored), 1.7 (4K anchored) against 3.5 (3K plain) and 8.8 (4K plain): the steep weight dependence that moves the optimum with resolution is gone once the base's sub-native band is native. Read as a shift in weight, anchored w2 ≈ plain w8 at 3K, anchored w4 ≈ plain w10 at 3K, and at 4K anchored w2 ≈ plain w10 and anchored w4 is above every plain weight tried.
- **What "optimum returns to native" means here.** HPSv2 has no interior maximum at any resolution: it rises monotonically through w8 even at 1K (31.7 → 33.1), as the kill-or-go preview also shows (peak at 5–6 then a slow fall only at 1K). So the native optimum is not a number the metric defines, and the paper's statement is the matched-weight one above: with the anchor, each weight produces what it produces at native (collapse boundary at w2, native quality at w4, small gains to w8), and the resolution-dependent retuning disappears. The √r law (w = 5.7 / 6.9 / 8.0) stays as the no-anchor baseline.
- **By eye, 4K sheet:** plain CFG 4 is noise on 6/6, plain 6–10 are coherent but carry dense white speckle on every surface (skin, pears, water, see the crops); anchored w2 is coherent on 6/6 but grainy (the "mildly under-guided" look of 1K CFG 2); anchored w4 has the correct framing on 6/6 and visibly less speckle than any plain column; anchored w6/w8 are the cleanest images in the set, with the speckle reduced further and no over-saturation (plain CFG 10 pushes the skin magenta). So the residual 4K speckle is also reduced by the anchor, not just the layout. **3K sheet:** anchored w2 is already as clean as plain CFG 8; anchored w4/w8 match plain 8/10 in contrast with slightly different, consistently complete layouts.
- **Decision (standing instruction 2026-10-27):** the paper's method claim is the prescription "keep guidance native once the base's sub-native band is native" with the matched-weight curve as its evidence; the headline figure is `results/anchor_weight_curve.png` redrawn from the paired 150-row anchored run on the kill-or-go prompts (`reg_anchor_grid`, launched on reg2 14:31, ETA Thursday evening), with bootstrap CIs. Anchored w2 and w8 on the grid prompts would complete that curve at n = 50/25 and are the next job for reg2 after it (about 18 GPU-h more, 2026-10-30).

### Paired anchored run on the kill-or-go prompts (2026-10-29 10:30; `jobs/reg_anchor_grid.jsonl`, 150 rows on reg2, all ok; `results/reg_anchor_grid_{eval,pref}.csv`, `results/anchor_paired{,_ir}.{csv,png}`, `scripts/anchor_paired.py`)
Base-anchored CFG at the native weight 4 on the grid's own prompts and seeds (3K: 100 prompts; 4K: 50), against every plain-CFG cell of the grid scored so far (3K: 58–65 per cell, both truncations; 4K: 33–40). Paired differences are anchored − plain over the prompts present in both, bootstrap 95 % CI over prompts.

| HPSv2.1 | anchored w4 | plain w4 | plain w8 | plain w10 | plain w12 | plain w16 |
|---|---|---|---|---|---|---|
| 3K (end 1.0) | **30.9** [30.3, 31.5] | 23.4 | 27.9 | 28.9 | 29.5 | 30.0 |
| 3K paired diff, win | – | +7.4 [6.3, 8.6], 100 % | +2.9 [2.2, 3.5], 95 % | +2.0 [1.5, 2.5], 85 % | +1.3 [0.8, 1.7], 74 % | +0.9 [0.3, 1.4], 66 % |
| 4K | **29.2** [28.4, 30.0] | 14.9 | 24.7 | 25.7 | 26.5 | 27.1 |
| 4K paired diff, win | – | +13.7 [11.6, 15.7], 100 % | +4.4 [3.4, 5.5], 100 % | +2.8 [2.0, 3.8], 91 % | +2.5 [1.7, 3.4], 82 % | +1.5 [0.6, 2.5], 73 % |

ImageReward: 3K anchored 1.42 [1.31, 1.52] vs plain w4 0.55 / w8 1.08 / w16 1.30 (paired +0.87 / +0.33 / +0.12, all CIs above 0); 4K anchored 1.37 [1.23, 1.50] vs plain w4 −1.08 / w8 0.47 / w16 0.80 (paired +2.46 / +0.89 / +0.57, win 94–97 %). CLIP 37.6 (3K) / 37.1 (4K); patch-KID 0.017 (3K) / 0.044 (4K); repetition: 0 duplicated subjects at 3K, 2/50 at 4K (grid baselines for these land Friday).

- **The anchored native weight beats every plain-CFG weight the grid tried, up to 16, at both sizes, and every paired CI excludes zero.** The plain curve is still rising at w16 (3K 30.0, 4K 27.1) and has not reached the anchored w4 value; by the 6-prompt stage C the anchored curve itself still gains ≈ 1 HPS from w4 to w8, so the gap to plain CFG at its best is larger than the table's last column. Truncating CFG at 70 % costs 0.5–1.3 HPS at 3K at every weight (as at 1K/2K), so the interval is not the lever.
- This is the paper's headline figure (`results/anchor_paired.png`): with the base's sub-native band native, the native weight is the right weight, and no plain weight recovers the gap. The 300-row w2/w8 run (`jobs/reg_anchor_grid_w28.jsonl`, half on reg2 from 08:37, ETA Sat; other half to the first free grid machine Friday) adds the anchored curve's ends at this sample size.
- The 6-prompt pilot numbers (31.0 / 29.6) were within 0.4 of the n = 100/50 means; the pilots were representative.

### Pyramid anchor and super-native weight: pilot launched (2026-10-29 12:30; `jobs/reg_pyramid_pilot.jsonl`, 18 rows, 4K, reg2, ETA 18:30)
Two generalisations of the base anchor, from the band picture (05, 2026-10-29): **pyramid** (`anchor.levels = [2]` at 4K: the band between the 1K and 2K Nyquists comes from a 2K pass at its own derived noise level, so only the band above the 2K Nyquist is left to the 4K pass; Upsample Guidance is the single-level case) and **super-native weight** (`super_scale = 1`: no guidance above the native Nyquist, where the model has seen no frequency at all; tests whether the residual 4K speckle is guidance-amplified noise). Variants at w4, base-anchored, six prompts: `pyr`, `sup`, `pyrs` (both). Unit tests: constant signals pass through the pyramid unchanged, the single-level formula is bit-identical to stage B/C, `super_scale` acts on Δ − low(Δ) only.

- **Speckle metric caveat.** `scripts/speckle.py` (full-resolution centre-crop Laplacian std, 3×3-median outlier fraction, spectral power above the native Nyquist) does not separate speckle from genuine texture: plain CFG 4K w10 has a "speck" fraction of 1.6 %, anchored w4 2.6 %, and the 2K reference 3.0 % (stained glass, pear skins). The preference scorers downscale to 1K and cannot see the speckle either. The pilot is therefore judged by the crops by eye, with the spectral fraction as a secondary number (anchored base 0.031, plain w4 0.019, w10 0.028; the anchored images carry more real fine detail). A reference-based speckle measure (energy above the native Nyquist that is spatially white) is a to-do.
- The w2/w8 run on reg2 is paused for the pilot (25 rows done, resume-safe) and restarts automatically after it.

### Pyramid pilot, stage 1 result (2026-10-29 16:30; 4K, w4, 6 prompts, seed 0; `results/reg_pyramid_pilot_{pref,eval,speckle}.csv`, sheets `reg_pyramid_pilot/sheets/pyr_4k_{full,crop4,crop8}.jpg`)
| 4K, w4 | HPSv2 | IR | CLIP | patch-CLIP | patch-KID | speck % | p>native | s/image |
|---|---|---|---|---|---|---|---|---|
| single-level anchor (stage C) | 29.6 | 1.43 | 33.4 | 20.0 | 0.051 | 2.6 | 0.031 | 698 |
| **pyramid** (levels [2]) | **30.5** | **1.51** | **33.7** | **21.5** | **0.031** | 1.9 | 0.044 | 788 |
| super-native rule (super_scale 1) | 27.9 | 1.35 | 32.8 | 15.6 | 0.103 | 2.9 | 0.032 | 697 |
| pyramid + super-native | 30.1 | 1.49 | 34.0 | 20.6 | 0.047 | 2.1 | 0.045 | 788 |

Paired against the single-level anchor on the same prompt and seed: pyramid HPS +0.9 (5/6 prompts up; per prompt +1.9, −1.2, +0.1, +1.1, +1.5, +2.1), IR +0.08; super-native −1.7 (0/6), IR −0.07; both −0.5/+0.07 (4/6).

- **(i) The pyramid improves on the single-level anchor.** Every metric moves the right way, KID at w4 (0.031) matches the single-level anchor at w8 (0.030), and the crops show it: less speckle on the cathedral, portrait skin and pear skins (the 3×3-median speck fraction drops from 2.6 % to 1.9 %), same layout and content. The one loss (lighthouse, −1.2 HPS) is white specks in the sky where the single-level image is clean; the stormy-sky prompt may read them as spray. Cost: one extra 2K pass pair per step, +13 % at 4K (788 vs 698 s; plain CFG 4K is ~690 s).
- **(ii) The super-native rule has the wrong sign.** Removing guidance above the native Nyquist *adds* speckle on all six prompts (crop8 sheet: sky, skin, pears all worse; KID 0.103 ≈ the single-level anchor at w2) and costs 1.7 HPS. So guidance above the native Nyquist is not amplifying noise there; it is suppressing it: in that band the unconditional prediction is noisier than the conditional one, and w·Δ pulls the output toward the cleaner conditional estimate. The band dissection's "the model has seen no frequency there, so do not guide" was the wrong inference; the right one is that the band is where guidance acts as a denoiser. This predicts that *more* weight above the native Nyquist reduces the residual speckle.
- **Stage 2 (launched 16:40 on reg2, `jobs/reg_pyramid_pilot2.jsonl`, 18 rows, ~3.2 GPU-h, ETA ~19:40; w2/w8 half paused at 25 rows again, resumes after):** pyramid + `super_scale` 8 and 12 at 4K (w4 below the native Nyquist, 8/12 above), and the pyramid at 3K (ratio 3, level 2 = a 1536-px pass) against the single-level anchor at 3K.
- **Decision (05):** with (i) confirmed on the pilot, the paper is written as a method paper: finding, diagnosis, and the pyramid anchor as the general method, of which Upsample Guidance is the two-level case. The super-native weight becomes a third component if stage 2 confirms its sign.

### Pyramid pilot, stage 2 result (2026-10-29 23:10; same six prompts; sheets `pyr2_{4k,3k}_{full,crop4,crop8}.jpg`)
| w4, base-anchored | HPSv2 | paired Δ vs single level (wins) | IR | CLIP | patch-CLIP | patch-KID | speck % | s/image |
|---|---|---|---|---|---|---|---|---|
| 4K single level | 29.6 | – | 1.43 | 33.4 | 20.0 | 0.051 | 2.6 | 698 |
| 4K pyramid | 30.5 | +0.9 (5/6) | 1.51 | 33.7 | 21.5 | 0.031 | 1.9 | 788 |
| 4K pyramid + super_scale 8 | 31.0 | +1.4 (6/6) | 1.52 | 33.9 | 22.6 | 0.023 | 1.7 | 789 |
| **4K pyramid + super_scale 12** | **31.1** | **+1.5 (6/6)** | 1.43 | 33.8 | **22.6** | **0.021** | **1.6** | 789 |
| 3K single level | 31.0 | – | 1.51 | 35.2 | 28.3 | 0.024 | 2.4 | 295 |
| 3K pyramid | 30.8 | −0.2 (2/6) | 1.45 | 34.5 | 27.8 | 0.024 | 1.9 | 341 |

- **The corrected super-native rule holds.** More weight above the native Nyquist (8 or 12 on top of the pyramid, w4 below) raises HPS on all six prompts, takes patch-KID at 4K/w4 to 0.021–0.023, *below* the single-level anchor at w8 (0.030) and level with 3K (0.024), and the crop8 sheet shows the 4K speckle gone: clean sky on the lighthouse, no black or white specks on the portrait skin, none on the street. Weight 8 and 12 look alike; 12 has slightly crisper stained glass. Per prompt the lighthouse, the only loss of the plain pyramid, is now +0.2/+0.4. **4K at w4 with pyramid + super-native weight scores 31.1 HPS, within 0.1 of plain CFG w4 at 2K (31.2, stage C): native quality at 4K with the native weight.**
- **At 3K the pyramid is neutral** (HPS −0.2, KID equal, crops identical): the single-level anchor already leaves no visible speckle there, so the extra level has nothing to fix. The pyramid matters where the gap between the anchor and the full Nyquist is wide (4× at 4K, 3× at 3K is on the edge). Stage 3 tests the super-native weight at 3K.
- **Why the super-native band wants more guidance, not none (working explanation for the paper).** Above the native Nyquist the model has never seen signal, so its conditional and unconditional predictions there are both mostly noise, but the conditional one is the less noisy of the two (text conditioning sharpens the prediction at every scale it can), and w·Δ in that band pulls toward it: guidance there is a denoiser. The speckle is the base prediction's noise in that band, and the cure is to lean harder on the conditional estimate. A derivation that fixes the weight (rather than sweeping 8/12/16/24) is the open item: the natural candidate is a weight that matches the band's SNR to the native band's, which the diagnostics recorder can measure.
- **Stage 3 (launched 23:15 on reg2, `jobs/reg_pyramid_pilot3.jsonl`, 30 rows, ~5 GPU-h, ETA ~04:30; w2/w8 paused at 25 rows, resumes after):** super_scale 16 and 24 on the pyramid at 4K (where it stops helping), single level + super_scale 12 at 4K (ablation: the two components separately), and single level / pyramid + super_scale 12 at 3K.
- **Grid-scale paired run queued (`jobs/reg_pyramid_grid.jsonl`, 50 rows, ~11 GPU-h):** pyramid + super_scale 12 at w4 on the 50 4K grid prompts, same seeds as `reg_anchor_grid` and `kill_or_go`, launches on the first grid machine that frees up (Friday afternoon), ahead of the second w2/w8 half. Result: the paper's headline table gets a third column, plain CFG (every weight) / single-level anchor (UG) / REG, all paired.

### Kill-or-go grid, early preference preview (2026-10-28 12:00; first 19 prompts per cell at 1K/2K/3K, 12–13 at 4K; HPSv2 / ImageReward, pre-scored on reg1 while the grid runs)
| HPS by CFG | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| 1K (trunc 0.9) | 30.7 | 31.3 | 31.8 | 32.0 | 32.1 | 31.9 | 31.7 | 31.5 | – |
| 2K (trunc 0.9) | 27.2 | 29.2 | 30.0 | 30.5 | 30.9 | 31.4 | 31.7 | 31.8 | – |
| 3K (trunc 0.7) | 12.9 | 19.1 | 22.3 | 24.3 | 25.7 | 27.2 | 27.9 | 29.2 | 29.0 |
| 4K (trunc 1.0) | – | 10.6 | – | 18.8 | – | 24.4 | – | 25.4 | – |

- Same picture as the 6-prompt pilots, now at n ≈ 19: at 1K HPS peaks at CFG 5–6 and falls above 8; at 2K it is still rising at 12; at 3K and 4K it climbs steeply through the whole range (CFG 4 at 3K is 10 HPS below the top). ImageReward is flat from CFG 4 at 1K/2K and rises with CFG at 3K/4K. So the grid will show the optimum moving up with resolution (Gate 1, in the sign measured by the pilots), and the preference metrics will not put a ceiling on it at 3K/4K; the base-anchored comparison on the same prompts (`reg_anchor_grid`, paired seeds) is what turns that into the paper's statement. Truncation: at 1K/2K, trunc 0.9 beats 0.5 by about 1 HPS at every CFG (ending CFG at 50 % costs more than it saves at native too).
- Final numbers with all 100/50/50/25 prompts per cell, bootstrap CIs and the repetition counts land with the grid (Friday).
