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
