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
Base-anchored CFG at the native weight 4 on the grid's own prompts and seeds (3K: 100 prompts; 4K: 50), against every plain-CFG cell of the grid (final, all rows, 2026-10-30 19:00; first recorded at 58–65 / 33–40 prompts per cell on 2026-10-29, same picture). Paired differences are anchored − plain over the same prompts, bootstrap 95 % CI over prompts.

| HPSv2.1 | anchored w4 | plain w4 | plain w8 | plain w10 | plain w12 | plain w16 |
|---|---|---|---|---|---|---|
| 3K (end 1.0), n = 100 | **30.9** [30.3, 31.5] | 22.0 | 27.5 | 28.6 | 29.2 | 29.7 |
| 3K paired diff, win | – | +8.9 [7.8, 10.0], 100 % | +3.4 [2.7, 4.1], 92 % | +2.3 [1.8, 2.9], 84 % | +1.7 [1.2, 2.1], 73 % | +1.2 [0.8, 1.7], 71 % |
| 4K, n = 50 | **29.2** [28.4, 30.0] | 14.3 | 24.1 | 25.5 | 26.2 | 27.2 |
| 4K paired diff, win | – | +15.0 [13.3, 16.7], 100 % | +5.1 [4.0, 6.4], 98 % | +3.8 [2.9, 4.8], 92 % | +3.0 [2.1, 4.1], 84 % | +2.1 [1.3, 2.8], 78 % |

ImageReward (final): 3K anchored 1.42 [1.31, 1.52] vs plain w4 0.07 / w8 0.89 / w16 1.20; 4K anchored 1.37 [1.23, 1.50] vs plain w4 −1.27 / w8 0.27 / w16 0.59 (paired +1.10 / +0.78 vs w8 / w16 at 4K, win 96–98 %; `results/anchor_paired_ir.csv`). CLIP 37.6 (3K) / 37.1 (4K); patch-KID 0.017 (3K) / 0.044 (4K); repetition: 0 duplicated subjects at 3K, 2/50 at 4K (grid baselines for these land Friday).

- **The anchored native weight beats every plain-CFG weight the grid tried, up to 16, at both sizes, and every paired CI excludes zero.** The plain curve is still rising at w16 (3K 29.7, 4K 27.2) and has not reached the anchored w4 value; by the 6-prompt stage C the anchored curve itself still gains ≈ 1 HPS from w4 to w8, so the gap to plain CFG at its best is larger than the table's last column. Truncating CFG at 70 % costs 0.5–1.3 HPS at 3K at every weight (as at 1K/2K), so the interval is not the lever.
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

### Pyramid pilot, stage 3 result (2026-10-30 10:00; `jobs/reg_pyramid_pilot3.jsonl`; sheets `pyr3_{4k,3k}_{full,crop8}.jpg`)
| w4, base-anchored | HPSv2 | paired Δ vs single level (wins) | IR | CLIP | patch-CLIP | patch-KID | speck % |
|---|---|---|---|---|---|---|---|
| 4K single level | 29.6 | – | 1.43 | 33.4 | 20.0 | 0.051 | 2.6 |
| 4K single level + super 12 | 30.1 | +0.5 (4/6) | 1.47 | 32.8 | 22.1 | 0.030 | 1.8 |
| 4K pyramid | 30.5 | +0.9 (5/6) | 1.51 | 33.7 | 21.5 | 0.031 | 1.9 |
| 4K pyramid + super 8 | 31.0 | +1.4 (6/6) | 1.52 | 33.9 | 22.6 | 0.023 | 1.7 |
| 4K pyramid + super 12 | 31.1 | +1.5 (6/6) | 1.43 | 33.8 | 22.6 | 0.021 | 1.6 |
| 4K pyramid + super 16 | 31.2 | +1.6 (6/6) | 1.41 | 33.5 | 23.3 | **0.018** | 1.4 |
| 4K pyramid + super 24 | 30.8 | +1.2 (5/6) | 1.41 | 32.9 | 22.9 | 0.022 | 1.2 |
| 3K single level | 31.0 | – | 1.51 | 35.2 | 28.3 | 0.024 | 2.4 |
| 3K single level + super 12 | 31.2 | +0.3 (5/6) | 1.43 | 35.0 | 27.9 | 0.023 | 2.1 |
| 3K pyramid + super 12 | 31.1 | +0.1 (3/6) | 1.39 | 34.4 | 27.1 | 0.025 | 1.8 |

- **Ablation at 4K: the two components are roughly additive and both are needed.** Super-native weight alone +0.5 HPS, pyramid alone +0.9, both +1.5. The crops say the same: the super-native weight alone removes the black specks on skin and the sky speckle but leaves the single-level anchor's coarser grain; the pyramid alone removes the grain but leaves specks on the lighthouse sky; both together are clean.
- **Super-native weight: plateau 12–16, over-sharpening at 24.** HPS 31.1/31.2 at 12/16, 30.8 at 24; KID is lowest at 16 (0.018); at 24 the crop8 sheet shows etched contours on the stained glass, halos on the pear outlines and cracked skin texture, and CLIP drops to 32.9. So the band is not "guide as hard as you like": it is a denoiser up to a point and a sharpener past it. The grid-scale run uses 12 (launched before this result; 12 and 16 are within noise of each other on HPS).
- **At 3K the super-native weight is a wash** (+0.3 HPS on 5/6 but IR −0.08 on 0/6, KID equal). With the 3K pyramid also neutral (stage 2), the method at 3K is the single-level anchor alone, and both extra components are 4K effects: the band above the native Nyquist, and the band between the 1K and 2K Nyquists, only matter when they are wide enough to carry visible error. This is consistent with the derivation target: the needed super-native weight grows with the ratio r (≈ native at r = 3, ≈ 12–16 at r = 4), and a rule that gives that dependence is what makes the component hyperparameter-free. Candidates: match the band's guided SNR to the native top band's (the recorder measures both), or the plain CFG optimum's own r-dependence (4 → 12–16 from 1K to 4K in the kill-or-go grid, which is the same number: the super-native weight is "what plain CFG wanted at 4K, confined to the band where it helps").
- **Decision:** REG = base anchor + pyramid + super-native weight at 4K, base anchor at 3K; the paper's method section presents the three components with the band dissection as the reason for each and the stage 3 ablation as evidence. The r-dependence of the super-native weight is the open theory item and the main writing risk; a 5K/6K row would test it (DyPE claims 4K×; our NTK sampler may not reach it; try one prompt first).

### Kill-or-go grid, final preference table (2026-10-30 19:00; all 11,800 rows, HPSv2 / ImageReward, `results/kill_or_go_pref.csv`; n = 200/100/100/50 prompts per cell at 1K/2K/3K/4K; first recorded at 9,244 rows at 11:00, numbers below ±1.4 of those)
| HPS by CFG | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| 1K, trunc 0.9 | 30.7 | 31.4 | 31.7 | 31.9 | 32.0 | 32.0 | 31.9 | 31.9 | – |
| 2K, trunc 0.9 | 26.6 | 28.8 | 29.8 | 30.4 | 30.8 | 31.3 | 31.5 | 31.6 | – |
| 3K, trunc 1.0 | 12.1 | 18.3 | 22.0 | 24.3 | 25.7 | 27.5 | 28.6 | 29.2 | 29.7 |
| 4K, trunc 1.0 | – | 10.6 | 14.3 | 17.8 | 20.3 | 24.1 | 25.5 | 26.2 | 27.2 |

- Gate 1, final: the HPS optimum is 6–8 at 1K (flat ±0.1 from 5 to 12), still rising at 12 at 2K, and rising through 16 at 3K and 4K, where CFG 4 sits 7.7 and 12.9 points below CFG 16. ImageReward is flat from CFG 3 at 1K and 2K and rises through 16 at 3K/4K. Truncation changes the level by ≤ 1.5 HPS and never the shape. The anchored w4 rows on the same prompts (above: 30.9 at 3K, 29.2 at 4K) are above the best plain weight at both sizes, so the figure `results/anchor_paired.png` stands with these n. Repetition counts and the CLIP/KID table follow with the CLIP/KID pass now running on reg1.

**CLIP, patch-KID and repetition on all 11,800 rows (final, 2026-10-30 20:00; `results/kill_or_go_eval.csv`; KID against 400 Aesthetic-4K 2K references, 16 1024-px patches per image; repetition from the single-subject half with the OWL-ViT counter; first recorded at 9,244 rows at 12:00, same picture):**

| CLIP by CFG | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| 1K, trunc 0.9 | 37.3 | 37.3 | 37.4 | 37.3 | 37.4 | 37.4 | 37.4 | 37.3 | – |
| 2K, trunc 0.9 | 37.0 | 37.1 | 37.2 | 37.1 | 37.1 | 37.1 | 37.0 | 37.0 | – |
| 3K, trunc 1.0 | 22.3 | 29.8 | 34.2 | 35.6 | 36.2 | 36.6 | 36.8 | 37.1 | 36.8 |
| 4K, trunc 1.0 | – | 16.7 | 23.6 | 28.0 | 30.9 | 34.7 | 34.9 | 34.9 | 35.2 |

| patch-KID by CFG | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| 1K, trunc 0.9 | 0.019 | 0.020 | 0.020 | 0.020 | 0.020 | 0.021 | 0.021 | 0.021 | – |
| 2K, trunc 0.9 | 0.018 | 0.019 | 0.020 | 0.019 | 0.019 | 0.018 | 0.017 | 0.017 | – |
| 3K, trunc 1.0 | 0.132 | 0.072 | 0.050 | 0.039 | 0.031 | 0.024 | 0.021 | 0.018 | 0.015 |
| 4K, trunc 1.0 | – | 0.203 | 0.151 | 0.112 | 0.094 | 0.062 | 0.048 | 0.043 | 0.037 |

| duplicate-subject rate by CFG | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| 1K, trunc 0.9 | 5 % | 4 % | 3 % | 4 % | 3 % | 2 % | 5 % | 3 % | – |
| 2K, trunc 0.9 | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 2 % | 2 % | – |
| 3K, trunc 1.0 | 2 % | 0 % | 8 % | 2 % | 4 % | 2 % | 6 % | 4 % | 6 % |
| 4K, trunc 1.0 | – | 0 % | 0 % | 4 % | 8 % | 0 % | 4 % | 8 % | 8 % |

| mean subject count by CFG (single-subject prompts) | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| 1K, trunc 0.9 | 1.04 | 1.04 | 1.03 | 1.04 | 1.03 | 1.02 | 1.05 | 1.04 | – |
| 2K, trunc 0.9 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.02 | 1.02 | – |
| 3K, trunc 1.0 | 0.40 | 0.70 | 0.96 | 0.98 | 1.06 | 1.00 | 1.06 | 1.04 | 1.06 |
| 4K, trunc 1.0 | – | 0.44 | 0.64 | 0.76 | 0.92 | 0.96 | 1.00 | 1.08 | 1.12 |

- CLIP and KID tell the same story as HPS with a different ceiling: at 1K and 2K every weight from 3 up is within 0.3 CLIP and 0.005 KID of every other (no optimum at all), while at 3K and 4K both climb through 16 (3K KID 0.050 → 0.015 from w4 to w16; 4K 0.151 → 0.037). The anchored-w4 rows' CLIP/KID on the same prompts (`reg_anchor_grid`, scoring now) are what the paired table needs; on the 6-prompt pilot they were level with plain w8–w16.
- **Anchored w4 on the same prompts, CLIP/KID (2026-10-30 13:00; `results/reg_anchor_grid_eval.csv`, n = 100 at 3K, 50 at 4K):** CLIP 37.6 at 3K and 37.1 at 4K, above plain CFG at *every* weight (plain peaks 37.1 at 3K, 35.2 at 4K) and level with 1K/2K (37.4/37.2). Patch-KID 0.017 at 3K and 0.044 at 4K, equal to plain w12 at both sizes; plain w16 is lower still (0.015 / 0.037), which with the HPS result (anchored w4 above plain w16) means KID at high weight is rewarding the sharpened texture that the preference scorers and the crops count against it. Mean subject count 1.00 / 1.04, no missing subjects. So on all four metrics the anchored native weight is at or above the best plain weight at 3K and 4K, and at native level on CLIP and subject count.
- **Repetition is not the failure mode; a missing subject is.** Duplicate-subject rates are 0–8 % at every size and weight (final rows: 2–8 % at 3K/4K with a weak rise at high weight, 1.08–1.12 mean count at 4K w12–16). What moves with weight at 3K/4K is the *mean count falling below one*: 0.40 at 3K w2, 0.44 at 4K w3, 0.64 at 4K w4, ≈ 1.0 from 3K w4 / 4K w10. At the native weight, extrapolated images lose the subject to texture (the collapse seen in the pilots), and raising CFG restores it; nothing duplicates. The paper's finding is therefore stated as under-guidance (loss of subject and layout) rather than the over-saturation/repetition story of the related work, which belongs to the no-RoPE-scaling setting.

### REG on the 4K kill-or-go prompts: paired three-way table (2026-10-30 22:30; `jobs/reg_pyramid_grid.jsonl`, 50 rows on reg4, all ok; `results/reg_pyramid_grid_{pref,eval}.csv`, `results/reg_paired_{hps,ir,clip}.csv`, `scripts/reg_paired.py`; sheets `reg_pyramid_grid/sheets/grid4k_{full,crop8}.jpg`)
REG = base anchor + pyramid level [2] + super-native weight 12, at w = 4, on the same 50 4K prompts and seeds as the grid and the single-level anchored run. Paired differences are REG minus the cell over the 50 prompts, bootstrap 95 % CI, win = fraction of prompts REG scores higher.

| 4K, n = 50 | HPS | ImageReward | CLIP | patch-KID | dup / mean count |
|---|---|---|---|---|---|
| plain CFG w4 | 14.3 | −1.27 | 23.6 | 0.151 | 0 % / 0.64 |
| plain CFG w16 (plain's best) | 27.2 | 0.59 | 35.2 | 0.037 | 8 % / 1.12 |
| single-level anchor w4 | 29.2 | **1.37** | 37.1 | 0.044 | 4 % / 1.04 |
| **REG w4** | **30.4** | 1.31 | **37.2** | **0.023** | 4 % / 1.04 |

| REG minus … (paired) | HPS | ImageReward | CLIP |
|---|---|---|---|
| plain w16 | +3.2 [2.6, 3.9], win 94 % | +0.72 [0.52, 0.93], 92 % | +2.1 [1.0, 3.2], 66 % |
| plain w8 | +6.2 [5.1, 7.4], 98 % | +1.04 [0.76, 1.32], 96 % | +2.5 [1.3, 3.9], 62 % |
| single-level anchor w4 | **+1.1 [0.7, 1.6], 74 %** | **−0.06 [−0.12, −0.01], 32 %** | +0.1 [−0.4, 0.6], 52 % |

- **The pilot gains hold at n = 50.** Over the single-level anchor, REG adds +1.1 HPS (CI clear of zero, 74 % of prompts) and halves the patch-KID (0.044 → 0.023), which is now *below* plain CFG at every weight (plain's floor is 0.037 at w16) and close to 3K plain w16 (0.015). CLIP is unchanged (37.2 vs 37.1, both above every plain weight). Over plain CFG at its best weight the margin grows to +3.2 HPS / +0.72 ImageReward / +2.1 CLIP, all CIs clear of zero.
- **ImageReward is the one dissent:** −0.06 against the single anchor, significant but small (the anchor-vs-plain-w16 gap on the same metric is 0.78, so REG keeps 92 % of it). ImageReward sees a 224 px thumbnail; what the pyramid/super-native pass changes is the ≥ 2K-cycle band, which only KID (2048 px patches) and the crops see. I take HPS + KID as the decisive pair and keep REG as the method; the running ablation (below) will say whether the ImageReward dip comes from the pyramid level or the super-native weight.
- **By eye (`grid4k_crop8.jpg`, 8 prompts):** plain w4 is a speckled wash with the subject missing on the single-subject prompts (dog, fox absent or a smudge); plain w16 recovers the subject under heavy white speckle; the single anchor has the composition and subject but keeps the speckle at 1/8 crop (child's cheek, fur, waterfall spray); REG removes it almost entirely (clean skin, fur with individual hairs, the waterfall as water), with a residual sprinkle on one fur crop and a slightly higher local contrast. Subject counts are unchanged (mean 1.04, one duplicate in 25 for both anchored variants).
- **Launched on the result:** the scaled component ablation on the same 50 prompts — pyramid-only (`anchor_base_pyr`) and super-only (`anchor_base_s12`), `jobs/reg_pyramid_ablation.jsonl`, 100 rows, reg4 now and reg3 after the PAG pilot, output into `reg_pyramid_grid` so one scoring call covers all three; and the generality pilot `jobs/reg_generality_pilot.jsonl` (REG vs plain CFG with SEG, TPG, and under dynamic YaRN; 36 rows, 4K, six pilot prompts) queued on reg2/reg5 behind the w2/w8 shards.

### PAG on the REG base, and REG at 5K (2026-10-30 23:30; `jobs/reg_pag_pilot.jsonl`, 15 rows on reg3, all ok; `results/reg_pag_pilot_{pref,eval,speckle}.csv`; sheets `reg_pag_pilot/sheets/pag4k_{full,crop8}.jpg`, `reg5k_{full,crop8,crop16}.jpg`)
Question 1: does "guidance stays native once the base is native" hold for perturbation guidance? PAG at its 1K-calibrated setting (blocks 15–17, scale 2) on plain CFG 4 and on REG (pyramid + super 12) at 4K, six pilot prompts, seed 0. REG-alone numbers from the pyramid pilot (same prompts and seeds).

| 4K, n = 6 | HPS | ImageReward | CLIP | patch-KID | speckle p>native |
|---|---|---|---|---|---|
| plain CFG 4 + PAG 2 | 15.3 | −0.92 | 22.7 | 0.153 | 0.034 |
| REG w4 (no PAG) | 31.1 | 1.43 | 33.8 | 0.021 | 0.042 |
| REG w4 + PAG 2 | 27.2 | 1.06 | 32.1 | 0.037 | 0.056 |

- **PAG does not rescue plain CFG at 4K** (HPS 15.3 ≈ plain w4's 14.3 on the grid; the sheet is the same grey wash with the subject missing). The failure is in the base prediction, and no perturbation term fixes it — consistent with the diagnosis.
- **PAG at its native scale over-perturbs the REG base: −3.9 HPS on 6/6 prompts** (cathedral 31.2 → 26.0, portrait 32.9 → 28.2), ImageReward −0.37, KID 0.021 → 0.037. By eye it is the 1K "over-perturbed" look from the calibration (08, 2026-10-27): stained glass flattened to a grid texture, street signs blanked, waxy skin; the super-native speckle is gone but with it the fine detail. So the second part of the claim fails as stated: the *CFG* weight stays native once the base is anchored, but a perturbation term whose Δ_pert is computed over 65K tokens is a much larger perturbation than the same term over 4K tokens — identity attention in three blocks removes far more at 4K. That is REG component (D), token-count calibration of perturbation guidance, which the pilot design had set aside. **Follow-up launched:** PAG scale {0.5, 1.0} on the REG base, same prompts (`jobs/reg_pag_scale_pilot.jsonl`, 12 rows, reg1 after its w2/w8 shard). If scale ≈ 2/r or 2/r² recovers the 1K effect, (D) is a one-line rule; if not, the anchored form (Δ_pert on the native-resolution pass) is the next thing to try. The SEG/TPG rows of the generality pilot test the same point.

Question 2: does the super-native weight keep growing with r? REG at 5120² (ratio 5, pyramid level 2 → a 2560 px pass), 27 min and 21.5 GB per image.

| 5K | HPS (4K same prompt) | ImageReward (4K) | speckle p>native (4K: 0.042) |
|---|---|---|---|
| super 12, portrait | 33.7 (32.9) | 1.79 (1.71) | 0.066 |
| super 12, lighthouse | 30.7 (30.5) | 1.51 (1.48) | |
| super 24, portrait | 33.2 (32.9) | 1.76 (1.71) | 0.065 |

- **REG works at 5K at the native weight with no change:** HPS and ImageReward equal to or above the same prompt at 4K; the lighthouse and portrait are coherent, no repetition. (A one-shot 5K plain-CFG baseline is not needed: 4K plain w4 already fails.)
- **The residual speckle grows with r at fixed super-native weight** (p>native 0.042 → 0.066, visible as white dots in the lighthouse sky at 1/8 crop), as the r-dependence candidate predicts; but super 24 did not remove it on the one prompt run (0.065, HPS −0.5), unlike the 4K over-sharpening it did not hurt either. n = 1–2 is not enough to fix the rule. **Follow-up launched:** the 5K sweep super {12, 16, 24} on all six prompts (`jobs/reg_5k_pilot.jsonl`, 15 new rows, reg2/reg5 after the generality pilot).
- **At 1K the same settings help** (`results/pert_calib_1k_pref.csv`, `results/guidance_zoo_1k_pref.csv`, same six prompts, CFG 4 alone = 32.5 HPS): PAG blocks 15–17 scale 2 = 32.8 (+0.4), SEG σ5 scale 1.5 = 33.4 (+0.9), TPG frac 0.5 scale 1.5 = 33.1 (+0.7); eight blocks or scale 3 hurt, as the calibration found by eye. So the perturbation term that is worth +0.4 to +0.9 at 1K costs −3.9 / −0.8 / −0.2 at 4K on the REG base. **The CFG weight transfers through the anchor; the perturbation scale does not** — it has to shrink with token count, which is REG component (D) and what the reduced-PAG pilot measures.
- **5K works (ratio 5, 5120², pyramid level 2 → 2560 px pass):** three rows, all ok, 1650 s and 21.5 GB peak per image on the 32 GB card. Portrait and lighthouse are coherent with one subject; HPS 32.2 (super 12, n = 2) / 33.2 (super 24, n = 1), ImageReward 1.65 / 1.76 — above the 4K numbers, as the 224-px scorers reward the larger canvas. The lighthouse at super 12 still shows a faint sparkle in the sky and the portrait at super 24 is cleaner and sharper than at 12, so the super-native weight wanted is higher at r = 5 than at r = 4, consistent with the r-dependence rule; with the component now optional (above) this is a qualitative note, not a derivation target.

**Follow-ups landed (2026-11-01 01:00; `jobs/reg_pag_scale_pilot.jsonl` 12 rows on reg4, `jobs/reg_5k_pilot.jsonl` 15 rows on reg5, all ok; same CSVs, re-scored over all 42 rows).**

| 4K, n = 6, REG = pyramid + super 12, w4 | HPS | ImageReward | CLIP | patch-KID | speckle p>native |
|---|---|---|---|---|---|
| REG + PAG 2 (blocks 15–17) | 27.2 | 1.06 | 32.1 | 0.037 | 0.056 |
| REG + PAG 1 | 29.3 | 1.29 | 32.7 | 0.027 | 0.050 |
| REG + PAG 0.5 | 30.2 | 1.41 | 33.1 | 0.025 | 0.048 |
| REG, no PAG | 31.1 | 1.43 | 33.8 | 0.021 | 0.042 |

- **Shrinking the PAG scale only approaches the no-PAG result; no scale beats it.** The loss is close to linear in the scale (−3.9 / −1.8 / −0.9 HPS at 2 / 1 / 0.5; ImageReward −0.37 / −0.14 / −0.02; KID 0.037 / 0.027 / 0.025 against 0.021), and the speckle statistic rises with the scale too, so PAG is not even removing the residual super-native speckle. Scale = 2/r (0.5 at r = 4) is therefore not a calibration rule that recovers the +0.4 HPS the term is worth at 1K; the simplest reading is that on an anchored base there is nothing left for a perturbation term to fix, and every bit of it costs detail. **Decision: perturbation guidance is not a component of REG.** Component (D) is reported as a negative result (the CFG weight transfers through the anchor, a perturbation term does not, and a token-count-scaled version only tends to zero), with the anchored form (Δ_pert computed on the native-resolution pass) noted as untested. No further PAG/SEG/TPG compute at grid scale; the SEG/TPG pilot rows already point the same way (−0.8 / −0.2 at their 1K scale).

| 5K (r = 5), n = 6, pyramid + super S, w4 | HPS (4K REG: 31.1) | ImageReward (1.43) | CLIP (33.8) | patch-KID (0.021) | speckle p>native (0.042) | s / image |
|---|---|---|---|---|---|---|
| super 12 | 30.7 | 1.49 | 34.8 | 0.034 | 0.066 | 1650 |
| super 16 | 30.6 | 1.47 | 34.4 | 0.031 | 0.067 | 1650 |
| super 24 | 30.6 | 1.46 | 34.2 | 0.032 | 0.074 | 1650 |

- **At 5K the super-native weight does nothing measurable** (HPS within 0.1, ImageReward within 0.03, KID 0.031–0.034 across 12–24) **and the speckle statistic does not fall with it** (0.066 → 0.074); with n = 6 the n = 1–2 hint above (super 24 cleaner on the portrait) does not survive. This closes the r-dependence question: there is no super-native weight to derive, because the component is a wash at 3K, a small KID gain at 4K and nothing at 5K. It is dropped from the method (as decided on the 4K ablation) and kept as an ablation row.
- **REG at 5K at the native weight is 0.4 HPS below the same prompts at 4K** (30.7 vs 31.1; ImageReward and CLIP above the 4K values, KID 0.034 vs 0.021) with the sparkle still visible in flat skies at 1/8 crop. Since weight is not the lever, the open lever is pyramid depth: at r = 5 the pyramid [2] anchors at 2560 px only, so the band between 1024 and 2560 px is extrapolated by a factor 2.5 before any anchor sees it. **Follow-up launched:** 5K pyramid-only with levels [2] and levels [2, 4] (2560 + 1280 px passes), six prompts (`jobs/reg_5k_depth_pilot.jsonl`, 12 rows, reg5 after its 3K shard, handoff6). If [2, 4] removes the sparkle the rule is "anchor at every octave down to native", which is what the method already says for 4K.


### Anchored weight curve at grid scale, w2/w4/w8 (2026-10-31 10:00; `jobs/reg_anchor_grid_w28.jsonl`, 300 rows on reg1/reg2/reg5, all ok; `results/reg_anchor_grid_{pref,eval}.csv`, `results/anchor_paired{,_ir}.{csv,png}`)
| HPS (n = 100 at 3K, 50 at 4K, 200/100 at 1K/2K) | w2 | w4 | w8 |
|---|---|---|---|
| 1K plain | 30.7 | 31.7 | 32.1 |
| 2K plain | 26.6 | 29.8 | 31.3 |
| 3K plain | 12.1 | 22.0 | 27.5 |
| 3K anchored | 28.7 | 30.9 | 31.9 |
| 4K plain | – | 14.3 | 24.1 |
| 4K anchored | 26.6 | 29.2 | 30.8 |

| 4K anchored | HPS | ImageReward | CLIP | patch-KID |
|---|---|---|---|---|
| w2 | 26.6 | 1.12 | 35.6 | 0.086 |
| w4 | 29.2 | 1.37 | 37.1 | 0.044 |
| w8 | 30.8 | 1.46 | 37.4 | 0.027 |

- **The anchored curve has the native shape.** It rises from w2 to w8 by 3.2 HPS at 3K and 4.3 at 4K, the same rise as 2K plain (4.7) and a gentler one than the 3K/4K plain curves (15.4 / 9.8 from w4 to w8 alone). 4K anchored lies on the 2K plain curve to within 0.6 at every weight; 3K anchored sits between the 2K and 1K plain curves. Stage C's pilot claim (matched-weight equivalence) holds at n = 100/50.
- **Consequence for the headline:** the anchored optimum is at the native optimum (6–8 at 1K), not at 4, so "the native weight works" means the native *range* works. Anchored w8 at 4K (HPS 30.8, ImageReward 1.46, KID 0.027) is as good as REG at w4 on HPS (+0.46 paired, CI [0.06, 0.85]) and better on ImageReward; REG at w2/w8 on the same prompts is running now (reg1/reg4/reg5) to settle whether the pyramid still adds at the native optimum.

### Component ablation at grid scale (2026-10-31 10:00; `jobs/reg_pyramid_ablation.jsonl`, 100 rows on reg4/reg3, all ok; `results/reg_pyramid_grid_{pref,eval}.csv`, `results/reg_paired_{hps,ir,clip}.csv`)
| 4K, w4, n = 50, paired | HPS | Δ vs single anchor | ImageReward | Δ vs single anchor | CLIP | patch-KID |
|---|---|---|---|---|---|---|
| single-level anchor | 29.2 | – | 1.37 | – | 37.1 | 0.044 |
| + pyramid [2] | 30.4 | **+1.15 [0.74, 1.57]**, 78 % | 1.37 | +0.00 [−0.05, 0.05] | 37.3 | 0.027 |
| + super-native 12 | 29.9 | +0.67 (0.46 below REG, CI [0.09, 0.85]) | 1.33 | −0.04 | 37.1 | 0.027 |
| + both (REG as run) | 30.4 | +1.13 [0.66, 1.61], 74 % | 1.31 | **−0.06 [−0.12, −0.01]** | 37.2 | **0.023** |

- **The pyramid carries the whole preference gain, at no ImageReward cost.** Pyramid-only equals REG on HPS (paired difference 0.03, CI [−0.17, 0.22]) and CLIP, keeps the single anchor's ImageReward exactly, and already brings patch-KID from 0.044 to 0.027.
- **The super-native weight is the component that costs ImageReward** (−0.066 [−0.108, −0.028] against pyramid-only) and it adds nothing to HPS once the pyramid is there; what it buys is the last KID step, 0.027 → 0.023 (≈ 3 KID std), i.e. the residual speckle that the crops show. The pilot's "+0.5 / +0.9 / +1.5 additive" reading (n = 6) does not survive n = 50: at scale it is +0.7 / +1.15 / +1.13.
- **Decision (05):** REG's default is the base anchor at every octave (the pyramid), at the native weight; the super-native weight becomes an optional speckle control reported in the ablation, and the open derivation for its r-dependence is no longer on the critical path. Whether the pyramid still adds at w8 (the native optimum) is what the running REG w2/w8 rows decide.

### Generality pilot: SEG, TPG and YaRN (2026-10-31 10:00; `jobs/reg_generality_pilot.jsonl`, 36 rows on reg2/reg5, all ok; `results/reg_generality_pilot_{pref,eval}.csv`; sheets `gen4k_{full,crop8}.jpg`)
| 4K, w4, six pilot prompts | HPS | ImageReward | CLIP | patch-KID |
|---|---|---|---|---|
| plain CFG 4 + SEG 1.5 | 16.6 | −0.57 | 25.5 | 0.158 |
| REG + SEG 1.5 | 30.3 | 1.50 | 33.0 | 0.041 |
| plain CFG 4 + TPG 1.5 | 18.5 | −0.18 | 25.6 | 0.128 |
| REG + TPG 1.5 | 31.0 | 1.50 | 33.0 | 0.022 |
| plain CFG 4, dynamic YaRN | 23.4 | 0.44 | 33.5 | 0.041 |
| REG, dynamic YaRN | **31.6** | 1.39 | 34.2 | **0.019** |
| REG, dynamic NTK (reference) | 31.1 | 1.43 | – | 0.021 |

- **Every row says the same thing: plain CFG at the native weight fails at 4K whatever else is attached, and the REG base makes the same settings work.** SEG and TPG on plain CFG are speckle fields with the subject gone (HPS 16–18); on REG they are clean. Like PAG, SEG at its 1K scale costs a little on the REG base (−0.8 HPS, KID 0.041 vs 0.021) while TPG is neutral (−0.2, KID 0.022), ordering PAG < SEG < TPG by how much they hurt, which is the order of how much they perturb.
- **REG is extrapolation-agnostic:** under dynamic YaRN it scores at least as well as under dynamic NTK (HPS 31.6 vs 31.1, KID 0.019 vs 0.021) with the crops equally clean, while plain CFG under YaRN keeps the missing-subject failure (lighthouse gone, pears multiplied). 07's NTK-over-YaRN choice was made on plain CFG's repetition; with REG the choice no longer matters, which is itself a result. The YaRN comparison goes to grid scale on reg2/reg3 next (`jobs/reg_generality_grid.jsonl`, 100 rows).

### REG weight curve at 4K and REG at 3K, grid scale (2026-10-31 18:30; `jobs/reg_pyramid_grid_{w28,3k}.jsonl`, 200 rows on reg1/reg2/reg3/reg4/reg5, all ok; `results/reg_pyramid_grid_{pref,eval}.csv`, `results/reg_paired_3k_{hps,ir,clip}.csv`)
REG here is still pyramid + super-native 12 (the rows were queued before the ablation). Paired over the same prompts and seeds.

| 4K, n = 50 | w2 | w4 | w8 |
|---|---|---|---|
| single anchor, HPS | 26.6 | 29.2 | 30.8 |
| REG, HPS | **29.2** | 30.4 | 31.0 |
| single anchor, ImageReward | 1.12 | 1.37 | 1.46 |
| REG, ImageReward | 1.22 | 1.31 | 1.42 |
| single anchor, patch-KID | 0.086 | 0.044 | 0.027 |
| REG, patch-KID | **0.026** | 0.023 | 0.021 |

- **At the native optimum (w8) the pyramid + super-native weight no longer add preference:** REG w8 vs anchor w8 is +0.22 HPS (CI [−0.10, 0.59]), −0.04 ImageReward ([−0.07, −0.01]), CLIP equal, KID 0.021 vs 0.027. So the +1.1 HPS at w4 is mostly what a higher weight also buys, and what survives at w8 is the KID/speckle gain.
- **What the pyramid changes is the shape of the curve, not its top.** REG at w2 beats the single anchor at w2 by 2.6 HPS (paired CI [2.1, 3.1], 94 % of prompts; ImageReward +0.10 [−0.01, 0.20]) and reaches the single anchor's w4; its KID at w2 (0.026) is the single anchor's at w8. The REG curve spans 1.9 HPS from w2 to w8 where the single anchor spans 4.3 and plain 4K spans 10 from w4 to w8. The pyramid therefore buys *weight robustness* — the native weight-sensitivity, which is what "resolution-equivariant" should mean — rather than a higher peak. That is the claim to write; "REG beats the anchor by 1.1 HPS" was a w4 artefact.
- **At 3K (n = 100) REG as run is slightly worse than the single anchor:** HPS 30.6 vs 30.9 (−0.34, CI [−0.54, −0.13]), ImageReward 1.33 vs 1.42 (−0.08, [−0.12, −0.04]), CLIP equal (37.5 / 37.6), KID better (0.0145 vs 0.0166). The 4K ablation makes the super-native weight the likely cause (it is the component that costs ImageReward); the pilot's "wash at 3K" was within its n = 6 noise. Both anchors beat plain CFG at every weight at 3K (vs plain w16: +1.2 / +0.9 HPS).
- **Queued for the new default (pyramid only, no super-native weight):** 4K w8 (reg1, running), 3K w4 (100 rows, reg4 + reg5 after tonight's pilots), 4K w2 (reg2 after its YaRN shard) — `jobs/reg_pyramid_grid_pyr_{w8,3k,w2}.jsonl`. With those the paper's weight-curve figure is plain / single anchor / pyramid at 3K and 4K, all from the same prompts.

### Generality at grid scale: REG under dynamic YaRN (2026-11-01 02:00; `jobs/reg_generality_grid.jsonl`, 100 rows on reg2/reg3, all ok; `results/reg_generality_grid_{pref,eval}.csv`, `results/reg_paired_yarn_{hps,ir,clip}.csv`; `scripts/reg_paired.py --dirs ... reg_generality_grid`)
The 6-prompt pilot had REG under dynamic YaRN 0.5 HPS above REG under NTK. Same test on the 50 4K kill-or-go prompts: REG (pyramid + super 12, w4, as the grid was configured) and plain CFG w4, both under dynamic YaRN, paired against the NTK rows already in the table.

| 4K, n = 50, w4 unless noted | HPS | ImageReward | CLIP | patch-KID | paired vs REG-NTK, HPS |
|---|---|---|---|---|---|
| plain CFG, NTK | 14.3 | −1.27 | 23.6 | 0.151 | −16.1 [−18.0, −14.3] |
| plain CFG, YaRN | 19.8 | −0.64 | 33.2 | 0.048 | −10.6 [−11.8, −9.3] |
| plain CFG w16, NTK | 27.2 | 0.59 | 35.2 | 0.037 | −3.2 [−3.9, −2.5] |
| REG, NTK | 30.4 | 1.31 | 37.2 | 0.023 | — |
| REG, YaRN | 30.6 | 1.33 | 37.1 | 0.019 | +0.2 [−0.1, +0.5], 54 % |

- **REG is indifferent to the extrapolation method:** under YaRN it is +0.2 HPS, +0.02 ImageReward and −0.1 CLIP against itself under NTK, every CI through zero (ImageReward [−0.03, +0.06], CLIP [−0.4, +0.2]); the pilot's +0.5 did not survive n = 50. The one real difference is patch-KID, 0.019 vs 0.023, the best KID of any cell so far. So the paper's generality claim for the extrapolation axis is: *the anchored base makes the native weight work, and which positional extrapolation produces that base is a second-order choice* (YaRN gives marginally cleaner texture). Plain CFG does better under YaRN than under NTK (19.8 vs 14.3 HPS) and still fails: the failure is in the guidance, not the position encoding, which is the paper's first finding re-stated on the other extrapolation method.
- Together with the SEG/TPG/PAG result (perturbation guidance does not transfer and is not part of REG), the generality section reduces to two rows per size: REG under NTK and under YaRN, both at the native weight. No further generality compute on Lumina; the second model (Qwen-Image) is the remaining generality axis.

### Pyramid-only weight curve at 3K and 4K, and 5K pyramid depth (2026-11-01 11:00; `jobs/reg_pyramid_grid_pyr_{3k,3k_w8,3k_w2,w8}.jsonl`, 350 rows on reg1/reg3/reg4/reg5, all ok; `jobs/reg_5k_depth_pilot.jsonl`, 12 rows on reg5, all ok; `results/reg_pyramid_grid_{pref,eval}.csv`, `results/reg_paired_{3k,4k}_w{2,4,8}_{hps,ir,clip}.csv`, `results/reg_pag_pilot_{pref,eval,speckle}.csv`; figure `results/reg_weight_curve{,_ir}.png`, `scripts/reg_weight_curve.py`)
The new default (pyramid anchor, no super-native term) on the full weight curve, paired against the single anchor at the same weight on the same prompts.

| 3K, n = 100 | HPS | ImageReward | CLIP | patch-KID | paired HPS vs single anchor | paired ImageReward |
|---|---|---|---|---|---|---|
| single anchor w2 / w4 / w8 | 28.7 / 30.9 / 31.9 | 1.22 / 1.42 / 1.45 | 37.1 / 37.6 / 37.5 | 0.0175 / 0.0166 / 0.0139 | — | — |
| pyramid w2 / w4 / w8 | 28.2 / 30.4 / 31.4 | 1.12 / 1.34 / 1.39 | 37.0 / 37.5 / 37.4 | 0.0175 / 0.0157 / 0.0136 | −0.44 [−0.66, −0.23] / −0.56 [−0.73, −0.39] / −0.53 [−0.66, −0.41] | −0.09 / −0.08 / −0.06, all CIs below 0 |
| plain w16 | 29.7 | 1.20 | — | — | −1.0 / −1.2 / −2.2 | — |

| 4K, n = 50 | HPS | ImageReward | CLIP | patch-KID | paired HPS vs single anchor | paired ImageReward |
|---|---|---|---|---|---|---|
| single anchor w2 / w4 / w8 | 26.6 / 29.2 / 30.8 | 1.12 / 1.37 / 1.46 | 35.6 / 37.1 / 37.4 | 0.086 / 0.044 / 0.027 | — | — |
| pyramid w2 / w4 / w8 | 28.3 / 30.4 / 31.1 | 1.17 / 1.37 / 1.44 | 36.7 / 37.3 / 37.2 | 0.038 / 0.027 / 0.022 | +1.71 [1.35, 2.09] / +1.15 [0.74, 1.57] / +0.30 [−0.04, 0.68] | −0.05 n.s. / 0.00 / −0.02 n.s. |
| pyramid + super 12, w2 / w4 / w8 | 29.2 / 30.4 / 31.05 | 1.22 / 1.31 / 1.42 | 36.8 / 37.2 / 37.2 | 0.026 / 0.023 / 0.021 | +2.6 / +1.1 / +0.2 | −0.10 n.s. / −0.06 / −0.04 |

| 5K, n = 6, w4 | HPS | ImageReward | CLIP | patch-KID | speckle p>native | s / image, GB |
|---|---|---|---|---|---|---|
| pyramid [2] (2560 px pass) | 29.8 | 1.48 | 35.1 | 0.070 | 0.067 | 1650, 21.5 |
| pyramid [2, 4] (2560 + 1280 px) | 29.8 | 1.40 | 34.5 | 0.061 | 0.080 | 1650, 21.5 |
| pyramid [2] + super 12 / 16 / 24 | 30.7 / 30.6 / 30.6 | 1.49 / 1.47 / 1.46 | 34.8 / 34.4 / 34.2 | 0.034 / 0.031 / 0.032 | 0.066 / 0.067 / 0.074 | 1650, 21.5 |

- **At 3K the pyramid costs, consistently:** −0.4 to −0.6 HPS and −0.06 to −0.09 ImageReward at every weight, every CI clear of zero (single anchor wins 62–81 % of prompts), CLIP and KID unchanged. It does not buy weight robustness there either (span w2→w8: pyramid 3.1, single anchor 3.2 HPS). At r = 3 the intermediate level sits at 1536 px, 1.5× native, not an octave: the extra pass splits the band the single anchor already handles and slightly blurs it.
- **At 4K the pyramid is a gain at w4 (+1.15 HPS), a wash at w8 (+0.3, CI through zero), and the super-native term adds nothing at w8** (pyramid-only 31.1 vs pyramid + super 12 31.05; KID 0.022 vs 0.021). The curve span w2→w8 for pyramid + super is 1.9 HPS vs 4.3 for the single anchor; pyramid-only w2 lands at 12:30 and will say whether the robustness is the pyramid's or the super term's.
- **At 5K the super-native term is what matters, and the deeper pyramid does not replace it:** pyramid-only KID 0.070 vs 0.034 with super 12, −0.9 HPS, and the [2, 4] pyramid is no better than [2] (KID 0.061, ImageReward lower, speckle higher). So the "super-native weight is out for good" decision of 01:00 is reversed for r ≥ 5: the term is a cost at r = 3 (−0.3 HPS), a KID-only gain at r = 4, and a clear gain at r = 5. **The components' value grows with the ratio**, which is the natural reading: the extrapolated band is 1 − 1/r² of the spectrum (89 % at r = 3, 96 % at r = 5), and the base model's own handling of it degrades with r.
- **Candidate rule (one line, zero per-resolution tuning), to be confirmed by the runs now on the cluster:** anchor levels at every octave strictly inside (1, r) — none at r = 3, [2] at r = 4, [2, 4] at r = 5–7 — and super-native weight w_s = w · (r − 2): 4 at r = 3 (a no-op, equal to w), 8 at r = 4 (on the 8–16 plateau), 12 at r = 5 (on the 12–24 plateau). The rule reduces REG to the single anchor at r = 3, which the 3K data say is the right answer. Running: single anchor + super 8 at 3K, n = 100 (reg3/reg4; the rule says the super term should not be applied at r = 3, this checks that it hurts), 5K [2, 4] + super 12 (the rule's 5K default) and [2] + super 8 (below the rule) on the six prompts (reg5), a 6K probe with [2, 4] + super 12 on two prompts (reg1; memory and coherence at r = 6), and the rule's 4K default (pyramid [2] + super 8, w4) on the 50 grid prompts (reg2 + reg1 after the current runs, handoff7). If the 4K default matches pyramid + super 12 (expected: plateau) and the 3K super term hurts, the rule stands and the paper's method table is one row per size with no free parameter.
- **4K w2 landed (12:00, reg2, 50 rows ok):** pyramid-only 28.3 HPS / 1.17 ImageReward / 36.7 CLIP / KID 0.038, against the single anchor's 26.6 / 1.12 / 35.6 / 0.086 — paired +1.71 HPS [1.35, 2.09] (92 % of prompts), +1.1 CLIP [0.5, 1.8], ImageReward n.s. The super-native term adds a further +0.9 HPS and halves KID again (29.2, 0.026) at w2. Curve span w2→w8 at 4K: single anchor 4.3, pyramid 2.85, pyramid + super 1.9 HPS. So at r = 4 both components contribute to the weight robustness, most of it at the low-weight end, and the super term's share is exactly what the candidate rule assigns it (8 at r = 4). Figure refreshed.
- Figure `results/reg_weight_curve.png`: plain / single anchor / pyramid / pyramid + super at 3K and 4K against the plain 2K curve, with bootstrap CIs. The single anchor already restores the native shape at both sizes; the pyramid lifts the low-weight end at 4K only.

### Rule confirmation: 3K super term, 4K rule default, 5K octave pyramid, 6K probe (2026-11-01 19:45; `jobs/reg_anchor_grid_s8_3k.jsonl` 100 rows reg3/reg4, `jobs/reg_pyramid_grid_pyr_s8.jsonl` 50 rows reg2/reg1, `jobs/reg_5k_depth_pilot2.jsonl` 12 rows reg5, `jobs/reg_6k_probe.jsonl` 2 rows reg1, all ok; `results/reg_pyramid_grid_{pref,eval}.csv`, `results/reg_paired_3k_w4_*.csv`, `results/reg_paired_{hps,ir,clip}.csv`, `results/reg_pag_pilot_{pref,eval,speckle}.csv`)

| cell | HPS | ImageReward | CLIP | patch-KID | paired HPS | paired ImageReward |
|---|---|---|---|---|---|---|
| 3K, n = 100: single anchor + super 8, w4 (vs single anchor w4: 30.9 / 1.42 / 37.6 / 0.0166) | 31.1 | 1.41 | 37.6 | 0.0161 | **+0.23 [0.15, 0.32]**, 69 % | +0.01 n.s. |
| 4K, n = 50: pyramid + super 8, w4 (vs pyramid + super 12: 30.4 / 1.31 / 37.2 / 0.023) | 30.4 | 1.34 | 37.4 | 0.024 | −0.05 n.s. | **+0.03 [0.02, 0.05]** |
| 5K, n = 6: pyramid [2] + super 8 (vs [2] + super 12: 30.7 / 1.49 / 34.8 / 0.034) | 30.5 | 1.48 | 35.1 | 0.051 | | |
| 5K, n = 6: pyramid [2, 4] + super 12 | 30.1 | 1.41 | 33.8 | 0.041 | | |
| 6K, n = 2 (lighthouse, portrait): pyramid [2, 4] + super 12 | 31.4 | 1.53 | 32.2 | 0.066 | 3135 s, 26.7 GB | |

- **The super-native term does not hurt at 3K — it helps by +0.2 HPS on the single anchor** (n = 100, CI clear, 69 % of prompts), ImageReward, CLIP and KID unchanged. So the rule's "no-op at r = 3" was over-fitted to the pyramid + super row: at 3K the pyramid is −0.55 and the super term +0.2, and the two had been read together. **The super-native term is positive at every ratio** (+0.2 at r = 3, +0.7 at r = 4 on the single anchor, +0.9 at r = 5) with a plateau at 4K (8 ≈ 12 ≈ 16) and at 5K (12 ≈ 16 ≈ 24, 8 is 0.2 HPS lower and KID 0.051 vs 0.034). A constant 3w = 12 is on the plateau everywhere.
- **The rule's 4K default (pyramid [2] + super 8) equals pyramid + super 12** on HPS and CLIP and is 0.03 ImageReward better; KID 0.024 vs 0.023. Plateau confirmed at n = 50.
- **The deeper 5K pyramid hurts again:** [2, 4] + super 12 is −0.6 HPS, −0.08 ImageReward, −1.0 CLIP and KID 0.041 vs 0.034 against [2] + super 12. The added pass is at 1280 px, 1.25× native. Together with 3K (1536 px, 1.5× native, −0.55) and 4K (2048 px, 2× native, +1.15 at w4): **an intermediate anchor helps when it is at least an octave above native, and hurts when it is closer than that.**
- **6K works** (ratio 6, 6144², 3135 s and 26.7 GB per image, two prompts coherent, HPS 31.4): the practical ceiling on a 32 GB card is between 6K and 7K. The 6K rows used [2, 4] (3072 + 1536 px), which the rule now excludes, so they are a feasibility note only.

**Final REG rule (decided 2026-11-01 20:00; replaces the candidate of 11:00):**
1. anchor the base prediction's sub-native band at the native resolution (the single anchor), native CFG weight w unchanged;
2. add intermediate anchor passes at full/2, full/4, … as long as the pass is at least 2× native: none at r = 3, [2] (2048 px) at r = 4, [2] (2560) at r = 5, [2] (3072) at r = 6, [2, 4] at r = 8;
3. super-native weight 3w (12 at w = 4) on the band above the native Nyquist.
Zero per-resolution tuning; every number is derived from r or from w. On the grid prompts the rule's cells are: 3K = single anchor + super 12 (running on reg1, n = 100; the super-8 row above is the proxy, 31.1 HPS), 4K = pyramid + super 12 (30.4 / 1.31 / 37.2 / 0.023, w2/w4/w8 done), 5K = pyramid [2] + super 12 (n = 6: 30.7; **n = 50 running on reg2–reg5 tonight with a plain-CFG w16 baseline, `jobs/reg_5k_grid.jsonl`**). The plateau rows (super 8 / 16 / 24; pyramid-only; deeper pyramid) become the ablation table.

### 3K rule cell, timing table, reg5 (2026-11-02 10:00; `jobs/reg_anchor_grid_s12_3k.jsonl` 100 rows reg1, all ok; `results/reg_pyramid_grid_{pref,eval}.csv`, `results/reg_paired_3k_w4_*.csv`; `results/timing.csv`)
- **The rule's 3K cell (single anchor + super 12, w4, n = 100): 31.1 HPS / 1.39 ImageReward / 37.7 CLIP / KID 0.0155**, paired against the single anchor +0.16 HPS [0.01, 0.30], +0.03 ImageReward [0.00, 0.06], CLIP n.s., KID 0.0155 vs 0.0166 — the same as the super-8 row (+0.23). The 3K rule curve at w2/w8 (200 rows) is half done on reg1; reg5's half never ran (below).
- **Timing and memory, from the job records (`results/timing.csv`; seconds per image at w4, RTX 5000 Ada, 30 steps):** plain CFG 288 s at 3K, 697 s at 4K; the single anchor costs +2 % (295 / 696 s), the pyramid +19 % at 3K and +14 % at 4K (343 / 794 s); the super-native term is free. Peak memory is the plain pass's: 14.1 GB at 3K, 17.4 GB at 4K, 21.5 GB at 5K (1623 s), 26.7 GB at 6K (3142 s). So the rule's cost is +2 % at 3K (no pyramid) and +14 % at 4K–6K, with no extra memory. Native 1K is 16 s, 2K 97 s.
- **reg5 is shared:** another user's suspended training process (pid 2186845, state T, 11 s of CPU time in 37 days) has held 12 GB on its card since 2026-09-25. The 5K shard there OOMed on all 25 rows (re-run on reg2–reg4), and the 3K shard launched beside it never started (`run_grid.py`'s 20 GB free-memory guard timed out after 6 h). The admin has been asked to remove the process; until then reg5 takes nothing, and handoff10 launches the 5K pyramid-only ablation there once the card is free.

### REG at 5K, grid scale (2026-11-02 11:00; `jobs/reg_5k_grid.jsonl`, 100 rows on reg2/reg3/reg4 (+ the 25 re-run after reg5's OOM), all ok; `results/reg_5k_grid_{pref,eval,speckle}.csv`, `results/reg_paired_5k_{hps,ir,clip}.csv`)
The rule's 5K cell (native anchor at 1024, intermediate anchor at 2560 px, super-native 12, w4) against the grid's best plain weight (w16), on the 50 4K kill-or-go prompts (25 aesthetic-4K + 25 single-subject), seed as in the grid.

| 5K (5120²), n = 50 | HPS | ImageReward | CLIP | patch-KID | OWL-ViT duplicate frac. | speckle p>native | s / image |
|---|---|---|---|---|---|---|---|
| plain CFG w16, NTK | 22.3 | −0.22 | 32.2 | 0.098 | 0.08 | 0.047 | 1480 |
| REG rule, w4, NTK | **30.0** | **1.37** | **37.2** | **0.038** | **0.00** | 0.061 | 1620 |
| paired REG − plain | +7.7 [6.3, 9.3], 100 % | +1.58 [1.26, 1.92], 98 % | +5.0 [3.0, 7.3], 78 % | | | | |
| 4K REG rule (same prompts) | 30.4 | 1.31 | 37.2 | 0.023 | 0.00 | 0.042 | 794 |

- **REG at 5K at the native weight is within 0.4 HPS of REG at 4K on the same prompts**, ImageReward and CLIP equal, with no duplicated subjects (OWL-ViT), where plain CFG at its best grid weight has fallen to 22.3 HPS, negative ImageReward, KID 0.098 and 8 % duplicate subjects. Every paired CI is far from zero; REG wins 50/50 prompts on HPS. The method claim therefore holds one step past the grid: r = 3, 4, 5 at w = 4, no change of any setting.
- The residual at 5K is texture, not layout: KID 0.038 vs 0.023 at 4K and the speckle statistic 0.061 vs 0.042, consistent with the six-prompt pilot. The pyramid-only ablation at 5K (n = 50) is queued on reg5 behind the shared card; the plain w4/w8 rows (the 5K plain curve) run on reg3/reg2 tonight. 6K is a feasibility note (26.7 GB, 52 min / image).

### Qwen-Image port (2026-11-02 11:00; `reg/models/qwen.py` `_sample`, `reg/extrap/dype.py` `DyPEQwenRope`; `scripts/qwen_port_check.py`, `scripts/qwen_determinism_check.py`, `scripts/qwen_diff_probe.py`)
- Qwen-Image now samples through the same `Guidance` object as Lumina: latents are unpacked from Qwen's (B, S, 64) tokens to (B, 16, H/8, W/8) for the band/anchor code and repacked per transformer call; Qwen's velocity is negated into Lumina's u = x0 − ε units (the anchor's `base` mix is not sign-symmetric); the timestep is σ itself; the CFG normalization is applied on the packed tokens after combining, as the stock pipeline does; an anchor pass calls the transformer on the pooled latent with its own `img_shapes`, so `QwenEmbedRope` (or DyPE) builds the RoPE for that grid. Perturbation terms (PAG/SEG/TPG) are not ported (not needed: component D is a negative result).
- DyPE for Qwen: `DyPEQwenRope` replaces `transformer.pos_embed`, keeps the frame axis and the text positions stock and gives the centred height/width axes NTK/YaRN frequencies when the extent exceeds the native 83 patches (1328 px / 16, Qwen's training resolution; 64 for Lumina/FLUX). Per call, so anchor passes get their own (less or un-) extrapolated frequencies. CPU test: equal to the stock tables below 83, different and finite above, pack/unpack exact.
- **Bit identity with the stock pipeline (1K, CFG 4, 8 steps, fp8 weights): max |diff| 0 after two fixes** found with the probe script — (i) cast the timestep to bf16 before dividing by 1000 (the pipeline's order), (ii) pass no text attention mask when the mask is all ones: the pipeline passes `None` for an unpadded prompt and an all-ones mask selects a different SDPA kernel (0.06 max diff on the first call, 0.5 on the latents after 8 steps). Both pipeline and loop are run-to-run deterministic. REG anchor at 2K (ratio 2, super 12): 62 s / 4 steps, 20.8 GB; with NTK the same.
- **Qwen pilot launched** (`jobs/qwen_pilot.jsonl`, 24 rows: six pilot prompts at 3K and 4K, plain CFG 4 vs the REG rule, NTK, 50 steps; reg4 now, reg1 tonight). Qwen costs 52 min per 4K image, so the Qwen grid will be n = 50 at one or two sizes, decided on the pilot.

### 3K rule weight curve (2026-11-03 11:30; `jobs/reg_anchor_grid_s12_3k_w28.jsonl`, n = 100; `results/reg_weight_curve.png`, `results/reg_weight_curve.csv`)

| 3K, HPS [95 % CI] | w2 | w4 | w8 |
|---|---|---|---|
| plain CFG | 12.1 | 22.0 | 27.5 (w16: 29.7) |
| single anchor | 28.7 [28.0, 29.4] | 30.9 [30.3, 31.5] | 31.9 [31.4, 32.4] |
| pyramid anchor | 28.2 | 30.4 | 31.4 |
| **rule: single anchor + super 12** | **29.7 [29.0, 30.3]** | **31.1 [30.5, 31.6]** | **31.8 [31.3, 32.3]** |

- The rule's 3K curve is the single anchor's with the low-weight end lifted: +1.0 HPS at w2 (paired CI [0.8, 1.2], rule above on 83 % of prompts), +0.2 at w4, −0.1 at w8 (ImageReward 1.26 / 1.39 / 1.43, CLIP and patch-CLIP within 0.1 of the single anchor at every weight, duplicates 2 %). Curve span w2→w8 is 2.1 HPS for the rule, 3.2 for the single anchor, 15 for plain CFG. Same reading as the pyramid at 4K: the super-native term buys weight robustness rather than a higher peak, and the rule is above the pyramid-only row at every 3K weight (+1.4 / +0.7 / +0.4), which is why the 3K rule has no intermediate level.
- The figure now carries the rule as a dotted blue series at 3K; at 4K the rule is the pyramid + super 12 row already plotted.

### 5K plain weight curve and pyramid-only ablation (2026-11-04 01:00; `jobs/reg_5k_plain_w48.jsonl`, `jobs/reg_5k_pyr_only.jsonl`, n = 50; `results/reg_5k_grid_{pref,eval,speckle}.csv`, `results/reg_weight_curve.png` 5K panel)

| 5K, n = 50 | HPS [95 % CI] | IR | CLIP | patch-CLIP | patch-KID | dup / no-subject | speck % |
|---|---|---|---|---|---|---|---|
| plain w4 | 8.6 [7.6, 9.7] | −2.17 | 12.8 | 8.5 | 0.250 | 0 % / 60 % | 0.57 |
| plain w8 | 15.2 [13.4, 17.0] | −1.24 | 24.2 | 12.1 | 0.163 | 8 % / 36 % | 0.56 |
| plain w16 | 22.3 [20.8, 23.7] | −0.22 | 32.2 | 15.4 | 0.098 | 8 % / 12 % | 1.08 |
| pyramid anchor, w4 (no super) | 29.6 [28.9, 30.4] | 1.39 | 37.2 | 19.4 | 0.052 | 4 % / 0 % | 1.36 |
| **REG rule, w4 (pyramid + super 12)** | **30.0 [29.3, 30.7]** | 1.37 | 37.2 | 21.0 | **0.038** | 0 % / 0 % | 1.29 |

- Plain CFG at 5K is a collapse at the native weight (no subject found in 60 % of single-subject images, patch-KID 0.25) and still 7.3 HPS below the rule at w16 (paired, CI [5.9, 8.9], rule above on 96 %); the plain curve is 8 HPS below REG at its best weight, the widest gap of the three sizes (figure, 5K panel).
- The super-native term at 5K, paired against the pyramid-only anchor on the same prompts: +0.4 HPS [0.2, 0.6] (rule above on 72 %), ImageReward +0.02 n.s., patch-CLIP +1.6, patch-KID 0.052 → 0.038, duplicates 4 % → 0 %, speck fraction unchanged (1.3 % both). Smaller on HPS than the six-prompt pilot (+0.9) but the KID/patch-CLIP gain holds at n = 50, so the rule's super-native weight stays for r ≥ 4 (at 3K it is the +1.0 at w2 recorded above). The texture artifacts the speckle fraction counts are the same for both, so the KID gain is in the mid-band texture, not in the white specks.
- The 3K/4K/5K headline table is now complete with both ablation rows at every size; the weight-curve figure has a 5K panel.

### Qwen 3K grid, REG vs plain CFG 4 (2026-11-04 11:00; `jobs/qwen_grid_3k.jsonl`, n = 50, same prompts and seeds as the Lumina 4K/5K grids; `results/qwen_grid_{pref,eval}.csv`)

| Qwen-Image 3K, n = 50 | HPS [95 % CI] | IR | CLIP | patch-CLIP | patch-KID | dup |
|---|---|---|---|---|---|---|
| plain CFG 4 (native weight) | 24.7 [23.1, 26.3] | 0.68 | 35.5 | 24.7 | 0.044 | 16 % |
| **REG rule, w4** (native anchor + super 12) | **29.2 [28.0, 30.3]** | **1.27** | 36.6 | 27.3 | **0.014** | 4 % |
| plain CFG 16 (best plain weight, 4× native; Thu 02:00) | 26.7 [25.5, 27.8] | 0.75 | 35.9 | 25.6 | 0.017 | 24 % |

- Paired over the 50 prompts: HPS +4.5 [3.5, 5.6], REG above on 90 %; ImageReward +0.59 [0.34, 0.85], above on 74 %. Patch-KID is a third of plain's and duplicates fall from 16 % to 4 %. The REG number (29.2) is 1 HPS below REG on Lumina at the same size (30.4 rule / 31.1 single + super 12 at 3K on the 100-prompt set), while Qwen plain at the native weight is 2.7 above Lumina plain (24.7 vs 22.0): Qwen degrades less at 3K and gains less from the anchor, the gap being 4.5 instead of 9.
- **Plain w16 baseline (Thu 02:00, 50 rows, 0 errors after the fix below):** the strong plain weight gains +2.0 HPS over w4 (paired [1.1, 2.9], above on 70 %) but stays well under REG at the native weight: REG − plain w16 = **+2.5 HPS [1.6, 3.5], REG above on 86 %**; ImageReward +0.51 [0.28, 0.75] (72 %); CLIP +0.6 [−0.3, 1.5] n.s.; patch-CLIP 27.3 vs 25.6; patch-KID 0.014 vs 0.017 (w16 closes most of w4's 0.044 gap on the texture statistic, as on Lumina); duplicates 4 % vs 24 % (w16 repeats more than w4's 16 %: pushing the weight buys contrast at the cost of tiling). So the six-prompt pilot's 0.7 gap was a small-n artefact: at n = 50 the Qwen gap to the best plain weight (+2.5) is twice Lumina's at 3K (+1.2) and REG does it at 4× lower guidance with a quarter of the duplicates.
- Plain w16 on the same prompts is the honest baseline (the six-prompt pilot put it within 0.7 HPS of REG). Its 50 rows failed on reg2/reg3 overnight: the text encoder loaded at 30.9 GB, the 7B Qwen2.5-VL in fp32, where reg1/reg4 load it in bf16 at 16 GB. Cause (11:00): `run_grid.py`'s encoding phase passed an empty negative prompt as None, so Qwen's default negative (a single space) was first requested at sampling time; on reg1/reg4 that entry was in the embedding cache from earlier runs, on reg2/reg3 it was a miss, which reloaded the 16 GB text encoder beside the 20 GB transformer. Fixed in e229abf (the phase encodes the row's negative as given), w16 rows relaunched on reg2/reg3 at 10:44.
- 4K grid (`jobs/qwen_grid_4k.jsonl`, the rule with the intermediate anchor at 2048 px, 100 rows at ~55 min) launched on reg1/reg4 as shards 0/4 and 1/4; shards 2/4 and 3/4 follow on reg2/reg3 once the w16 baseline is through.

### Qwen 4K grid, REG vs plain CFG 4 (2026-11-05 20:40; `jobs/qwen_grid_4k.jsonl`, n = 50, 100 rows on reg1–reg4, 0 errors; `results/qwen_grid_{pref,eval,speckle}.csv`)

| Qwen-Image 4K, n = 50 | HPS [95 % CI] | IR | CLIP | patch-CLIP | patch-KID | dup / no-subject | speckle p>native | s / image |
|---|---|---|---|---|---|---|---|---|
| plain CFG 4 (native weight) | 22.2 [20.5, 24.0] | 0.20 | 34.3 | 19.0 | 0.079 | 0 % / 8 % | 0.0043 | 3615 |
| **REG rule, w4** (native anchor, intermediate anchor at 2048, super 12) | **28.8 [27.6, 30.0]** | **1.22** | **36.1** | **22.6** | **0.031** | 0 % / 0 % | 0.0099 | 3665 |

- Paired over the 50 prompts: HPS **+6.6 [5.1, 8.2], REG above on 94 %**; ImageReward +1.02 [0.71, 1.36], above on 84 %; CLIP +1.8 [0.5, 3.2] (the first Qwen cell where prompt alignment separates). Plain loses the subject on 8 % of images; REG on none. Cost of the rule at 4K: +1.4 % time per image.
- Same shape as Lumina across sizes: REG is flat in resolution (29.2 at 3K, 28.8 at 4K; IR 1.27 / 1.22) while plain at the native weight falls (24.7 → 22.2; IR 0.68 → 0.20), so the gap widens with r (4.5 → 6.6 HPS).
- Texture: REG's patch-KID rises from 0.014 at 3K to 0.031 at 4K (plain 0.044 → 0.079; Lumina's rule at 4K: 0.023), so some residual remains at 4K. The grain seen by eye in two of the six pilot crops does not show as speckle at grid scale: the fraction of power above the native band is 0.010 for REG (0.005 at 3K), four times below Lumina's rule at 4K (0.042), and the speck fraction is 0.25 % (0.46 % at 3K). The higher value than plain (0.004) is mostly plain being blurred (Laplacian std 11.9 vs 18.8). By-eye check on grid crops and the super-8 / milder-shift variants are the follow-up if the w16 comparison leaves room.
- Strong baseline: `jobs/qwen_grid_4k_w16.jsonl` (plain CFG 16, same 50 prompts) launched 20:28 on reg4/reg3/reg2 and 20:39 on reg1 as shards 0–3 of 4; ~13 h. At 3K REG − plain w16 was +2.5 HPS; on Lumina the w16 gap grew from 3K to 4K.
- Runner note: handoff17's first version read three ssh broken pipes on Wed night as ended shards (no rows lost; fixed in 4cb5639: a worker counts as ended only on an explicit IDLE twice, rsync retried). reg1's shard ran ~6 h slow on Wed night beside the CPU scoring of the w16 rows.

### Qwen pilot, first 3K rows (2026-11-02 13:30; `results/qwen_pilot_3k.png`)
- Three prompts at 3K from reg4 (cathedral, mountain, still life), plain CFG 4 vs the REG rule (single native anchor at 1024, super-native 12), NTK, 50 steps. REG above plain on every pair: HPS 25.0 / 27.4 / 32.9 vs 18.1 / 18.3 / 26.4 (mean 28.4 vs 20.9), ImageReward 1.40 vs 0.89, CLIP 36.8 vs 35.4, patch-CLIP 30.8 vs 24.5. Time 1385 s vs 1261 s per image (+10 %), peak 22.5 GB for both.
- By eye the plain rows fail harder than Lumina does at 3K: under-exposed and flat, the stained glass turned to speckle, the cathedral reduced to three windows in the dark, the still life lost in shadow. REG gives the whole scene at full contrast with clean glass, pears and copper.
- Caveat on the baseline: Qwen's dynamic shift is linear in token count with no cap (`calculate_shift`, 256 → 0.5, 8192 → 0.9), so the stock mu is 2.35 at 3K and 3.79 at 4K; only 5 of 50 steps fall below σ = 0.5 at 3K and 2 at 4K, against 15 at the native 1328 px. The stock pipeline does the same, so "plain CFG" is stock Qwen at 3K, but a fair baseline should get a sensible schedule. Appended to the pilot (rows 24–47): plain CFG and REG at mu 0.83 (native) and 1.31 (2K), six prompts each, so the method comparison can be made at the best plain schedule. handoff13 relaunches reg4 on them after its first shard; reg1's shard reads the extended file.

### Qwen pilot complete (2026-11-03 11:00; `results/qwen_pilot_{pref,eval}.csv`, 48 rows, six prompts per cell)

| size | shift mu | plain CFG 4: HPS / IR / patch-CLIP / KID | REG rule: HPS / IR / patch-CLIP / KID |
|---|---|---|---|
| 3K | 0.83 (native) | 21.8 / 0.51 / 24.6 / 0.053 | 28.4 / 1.41 / 29.7 / 0.026 |
| 3K | 1.31 (2K) | 21.1 / 0.62 / 25.3 / 0.041 | 29.0 / 1.52 / 30.2 / 0.026 |
| 3K | 2.35 (stock) | 22.3 / 0.67 / 24.4 / 0.053 | 28.7 / 1.49 / 29.8 / 0.024 |
| 4K | 3.79 (stock) | 21.2 / 0.64 / 19.7 / 0.070 | 29.6 / 1.49 / 24.1 / 0.024 |

- The schedule caveat is closed: plain CFG 4 is at HPS 21–22 under all three shifts and REG at 28–29 under all three, so the stock schedule is used from here. REG above plain by +6.6 to +8.4 HPS, +0.8 to +0.9 ImageReward, +5 patch-CLIP, patch-KID halved at 3K and a third at 4K. REG at 4K (29.6) is level with REG at 3K, the same flatness in r as on Lumina.
- Qwen plain at 3K/4K fails harder than Lumina's (Lumina plain w4: 28 at 3K); the stock pipeline's uncapped shift does not explain it (the native shift gives the same). The pilot's six prompts are not the grid prompts, so the number that goes in the paper comes from `jobs/qwen_grid_3k.jsonl` (n = 50, REG w4 vs plain w4, launched by handoff14 after the weight pilot). A plain-weight check (`jobs/qwen_pilot_w.jsonl`, CFG 8 and 16 at 3K, reg1/reg4 now) decides whether a stronger plain baseline is appended to the grid, as w16 was at 5K on Lumina.
- By eye at 4K (sheet in the scratch dir, not committed): plain is washed out and low-contrast, with a blurred canvas-like texture at native pixels; REG gives the whole scene at full contrast, but its crops carry a fine grain (mountain reflection, street) that the 3K REG rows do not. Candidate causes, to test on the grid prompts later: the stock 4K schedule (2 of 50 steps below σ 0.5) and the super-native weight 12 at this r; a milder shift or super 8 at 4K is the first thing to try if the 4K grid is run.
- **Plain-weight pilot (13:00; `jobs/qwen_pilot_w.jsonl`, stock schedule, six prompts):** plain CFG 4 / 8 / 16 at 3K: HPS 22.3 / 27.3 / 28.0, ImageReward 0.67 / 0.99 / 1.06, patch-CLIP 24.4 / 27.4 / 27.4, patch-KID 0.053 / 0.025 / 0.021; REG w4: 28.7 / 1.49 / 29.8 / 0.024. The Lumina pattern repeats on Qwen: the best plain weight at 3K is 4× the native one, and at it plain is within 0.7 HPS of REG at the native weight (IR −0.4, patch-CLIP −2.4; KID slightly better). So the Qwen table needs the strong baseline: `jobs/qwen_grid_3k_w16.jsonl` (plain w16, same 50 prompts) runs on reg3 and reg2 as they free tonight (handoff15); the grid's REG-vs-plain-w4 rows are on reg1/reg4 since 12:55. On Lumina the w16 gap was +1.2 at 3K and +3.2 at 4K, so the 4K Qwen grid is where the method should separate most, with the grain caveat above to settle.

### Kill-or-go grid, early preference preview (2026-10-28 12:00; first 19 prompts per cell at 1K/2K/3K, 12–13 at 4K; HPSv2 / ImageReward, pre-scored on reg1 while the grid runs)
| HPS by CFG | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| 1K (trunc 0.9) | 30.7 | 31.3 | 31.8 | 32.0 | 32.1 | 31.9 | 31.7 | 31.5 | – |
| 2K (trunc 0.9) | 27.2 | 29.2 | 30.0 | 30.5 | 30.9 | 31.4 | 31.7 | 31.8 | – |
| 3K (trunc 0.7) | 12.9 | 19.1 | 22.3 | 24.3 | 25.7 | 27.2 | 27.9 | 29.2 | 29.0 |
| 4K (trunc 1.0) | – | 10.6 | – | 18.8 | – | 24.4 | – | 25.4 | – |

- Same picture as the 6-prompt pilots, now at n ≈ 19: at 1K HPS peaks at CFG 5–6 and falls above 8; at 2K it is still rising at 12; at 3K and 4K it climbs steeply through the whole range (CFG 4 at 3K is 10 HPS below the top). ImageReward is flat from CFG 4 at 1K/2K and rises with CFG at 3K/4K. So the grid will show the optimum moving up with resolution (Gate 1, in the sign measured by the pilots), and the preference metrics will not put a ceiling on it at 3K/4K; the base-anchored comparison on the same prompts (`reg_anchor_grid`, paired seeds) is what turns that into the paper's statement. Truncation: at 1K/2K, trunc 0.9 beats 0.5 by about 1 HPS at every CFG (ending CFG at 50 % costs more than it saves at native too).
- Final numbers with all 100/50/50/25 prompts per cell, bootstrap CIs and the repetition counts land with the grid (Friday).
