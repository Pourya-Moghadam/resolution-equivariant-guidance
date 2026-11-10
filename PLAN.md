# Project Plan: Resolution-Equivariant Guidance (REG) → ICML 2027

Approved 2026-10-24. Source of truth for phases, gates and deliverables.

## Context
We identified a gap (see `04-novelty-check.md`): resolution-extrapolation papers (DyPE, SEGA, UltraImage) keep guidance at native-resolution settings, and guidance papers never test beyond native resolution. Our method, REG (`05-method-and-contributions.md`), makes guidance resolution-equivariant with no new hyperparameters. This plan turns that into a submission.

**Decisions already made:** target ICML 2027 (late Jan 2027; confirm exact date on icml.cc, fallback NeurIPS 2027); 5 Linux servers, each one RTX 5000 32 GB; engineering by the user + Claude Code sessions; approach the SEGA/TPG authors only after the kill-or-go gate passes.

**Timeline:** 13 weeks, 2026-10-27 → ~2027-01-25. Two hard gates (end of Phase 1 and end of Phase 3) decide whether we continue toward ICML or re-target NeurIPS.

**Working principle:** every phase ends with an artifact committed to this repo (mirrored to the servers), so any Claude Code session can pick up from the repo alone.

---

## Repo layout (to be created in Phase 0)
```
resolution-equivariant-guidance/
  README.md, 01–06-*.md          # research notes
  PLAN.md                        # this plan
  reg/                           # python package
    models/   lumina.py flux.py qwen.py        # loaders, text-embedding cache, tiled VAE
    extrap/   dype.py sega.py ultraimage.py    # RoPE/attention extrapolation ports
    guidance/ cfg.py fdg.py pag.py seg.py tpg.py  base.py (GuidanceSignal Δ_t hooks)
    reg/      bandsplit.py schedule.py anchor.py calib.py   # components A, B, C, D
    diag/     spectrum.py entropy.py magnitude.py            # diagnostics
    eval/     patch_fid.py kid.py clip.py hps.py genEval.py repetition.py
  scripts/    run_grid.py run_matrix.py run_eval.py benchmark.py server_audit.sh
  configs/    *.yaml (model × extrap × guidance × resolution)
  jobs/       job list files; each server runs rows where idx % REG_NUM_SERVERS == REG_SERVER_ID
  results/    csv/json only (images live on server storage, not in git)
  paper/      ICML LaTeX template, figures/
```

---

## Phase 0 — Environment, baselines, measurement (Week 1: Oct 27 – Nov 2)
Goal: every later number rests on a working, timed pipeline.

1. **Server audit** (user, 30 min): run `scripts/server_audit.sh` on all 5 servers; paste outputs into `06-compute-and-datasets.md`. If GPU is Turing (16 GB) rather than Ada (32 GB), stop and re-plan models.
2. **Environment** (built on reg1, then on every worker): `.venv` with PyTorch 2.14 + CUDA 13, diffusers (from main), torchao, later FlashAttention-2; versions pinned in `requirements.lock`, built by `scripts/setup_env.sh` (no conda/uv on the servers). Deploy code with `scripts/push_code.sh`, then `fanout.sh --workers 'cd ~/reg && bash scripts/setup_env.sh'`. See `SERVER_SETUP.md`.
3. **Models running at native res:** Lumina-Image 2.0, FLUX.1-dev (bf16, text encoders offloaded), Qwen-Image FP8 (torchao or prequantized). Implement text-embedding cache + tiled VAE decode in `reg/models/`.
4. **Reproduce one baseline:** DyPE on FLUX at 2K using the authors' code → confirms RoPE extrapolation plumbing before porting.
5. **Benchmark** (`scripts/benchmark.py`): seconds/image for each model at 1K/2K/4K with CFG → replace the estimates table in `06-*.md`. Re-budget phases 1–4 from real numbers.
6. **Data:** download prompt sets (LAION-Aesthetics/HR captions or Aesthetic-4K eval split, GenEval, HPSv2, DrawBench), Unsplash Lite reference images (~10K ≥2K). Write ~200 single-subject prompts for the repetition metric. Store under `/data/reg/` on shared storage.
7. **Evaluation pipeline** (`reg/eval/`): patch-FID (16×1024² crops), KID, CLIP score, HPSv2, ImageReward, GenEval harness, repetition count via GroundingDINO. Unit-check on 50 native images.
8. **Job runner:** `scripts/run_grid.py --server-id k` consumes `jobs/*.jsonl`, skips done rows (resume-safe), writes one JSON per image; `rsync` results nightly to server 1. Split rows by `idx % REG_NUM_SERVERS`, not a hard-coded 5.
9. **Git:** init repo, `.gitignore` images/weights, first commit.

**Deliverable:** benchmark table + working eval on native-res images + 5 servers runnable from one job file.

---

## Phase 1 — Port + kill-or-go experiment (Weeks 2–3: Nov 3 – Nov 16)
Goal: evidence that optimal guidance shifts with resolution in the direction REG predicts.

1. **Port DyPE to Lumina-Image 2.0** (`reg/extrap/dype.py`): Lumina uses 2D RoPE in its DiT blocks; apply DyPE's per-step frequency rescaling to Lumina's RoPE. Validate: Lumina + DyPE at 2K/4K produces coherent images vs. vanilla (repetition visibly reduced). ~3–4 days.
2. **Guidance zoo** (`reg/guidance/`): a `GuidanceSignal` base class exposing Δ_t = v_cond − v_uncond (CFG) or v_clean − v_perturbed (PAG, SEG, TPG), with hooks for (a) band split, (b) per-band scale schedule and interval, (d) perturbation params. FDG implemented as CFG + band split. Verify at 1K each reproduces its paper's qualitative behavior. **Done 2026-10-27** (`reg/guidance/`, own sampling loop in `reg.models.lumina`, bit-identical to the pipeline for CFG; 1K validation `jobs/guidance_zoo_1k.jsonl`).
3. **Grid search** (`jobs/kill_or_go.jsonl`): Lumina + DyPE (dynamic NTK, per 07). **Decided 2026-10-26 from the pilot (08) and measured speeds (06):**
   - CFG scale {2,3,4,5,6,8,10,12} × interval end (`cfg_trunc_ratio`) {0.5,0.7,0.9,1.0} = 32 settings at 1K (200 prompts) and 2K (100 prompts); 18 settings (CFG + 16, × {0.7,1.0}) at 3K (100 prompts); CFG only {3 … 16} (8 settings) at 4K (50 prompts). **Launched 2026-10-27** (`scripts/make_kill_or_go.py`, 11,800 rows, ~330 GPU-h); CFG 16 added after sweep C (08) showed the 3K/4K CLIP optimum at 6–10 with decline beyond. ≈ 32 + 167 + 127 + 77 = **~400 GPU-h ≈ 4 days on 4 GPUs**. 3K replaces 4K as the main high resolution (clean, 2.4× cheaper; 4K keeps the speckle issue).
   - FDG w_low/w_high pairs and the PAG/SEG runs follow as a second job once the guidance zoo (step 2) exists.
   - Fixed seeds (one per prompt); prompts = 100 Aesthetic-4K eval captions + 100 own single-subject prompts (`prompts/`), see 06.
   - Metrics (step 7 decision): patch-FID + KID on Aesthetic-4K references, CLIP score (OpenCLIP ViT-L/14), repetition count (GroundingDINO, single-subject prompts). ImageReward parked (package incompatible with transformers 5, see 06); HPSv2 and GenEval are Phase 3 additions, not gate metrics.
4. **Analysis notebook** (`results/kill_or_go.ipynb`): best setting per resolution per metric; plot optimal low-band scale / interval end vs. r; compare against REG-(B) prediction (low-band guidance should end earlier at higher r; high-band roughly constant).

**GATE 1 (Nov 17):**
- **GO** if optima move monotonically with r in the predicted direction for ≥2 of 3 quality metrics, and the shift is larger than seed noise (bootstrap CI).
  - **Revised 2026-10-28 (decision made per the user's standing instruction):** "predicted direction" is dropped. The pilots (08) show the optimum *rises* with r (native ≈ 4, 3K ≈ 6–8, 4K ≈ 8–10) and that the rise sits in the coarse image-relative band only — the opposite sign to 05-(B)'s original prediction. The gate now asks only that the shift be monotone and larger than seed noise; the sign is a finding. 05-(B) is reformulated as band-wise magnitude matching, to be tested by the diagnostics (`jobs/reg_diag.jsonl`, pulled forward from Phase 2).
- **NO-GO** → write up negative result as internal note; switch to backup idea (joint space-time video extrapolation or prediction-target × resolution; see `03-*.md`), re-target NeurIPS.
- **On GO:** user emails the SEGA/TPG authors with the kill-or-go plot and a 1-page summary proposing collaboration (SEGA/TPG code access, co-authorship). Draft the email in `paper/outreach.md`.

---

## Phase 2 — Diagnostics + REG components A/B/D (Weeks 4–5: Nov 17 – Nov 30)
Goal: show *why* guidance breaks, and that REG recovers the grid optimum with zero tuning.

1. **Diagnostics** (`reg/diag/`, ~50 GPU-h): on 100 prompts × {1K,2K,4K} record per step: radial power spectrum of Δ_t (in cycles/image), ‖Δ_t‖/‖v_base‖, attention entropy of clean vs. perturbed branch. Produce the paper's Figure 2 (spectrum drift) and Figure 3 (entropy gap vs. r).
2. **Implement A** (`reg/reg/bandsplit.py`): band split with cutoff κ in cycles/image (FFT-based low-pass; κ chosen once at native res, e.g. from FDG's default).
3. **Implement B** (`reg/reg/schedule.py`): per-band time remap; low band uses SD3-style shift with α = r, high band identity. Guidance intervals defined on native axis.
4. **Implement D** (`reg/reg/calib.py`): SEG σ_tok = r·σ₀; PAG/TPG window = native receptive field; entropy-gap matching option.
5. **Validation run** (~150 GPU-h): Lumina + DyPE at 2K/4K with REG(A+B) for CFG/FDG and REG(D) for PAG/SEG/TPG, native-res defaults only. Compare to (i) native defaults unmodified, (ii) grid-search optimum from Phase 1. Success = REG within CI of the grid optimum.
6. **Port SEGA and UltraImage** to Lumina (`reg/extrap/sega.py`, `ultraimage.py`) in parallel (Claude Code session on a non-GPU task slot). If the SEGA authors share their code, use theirs.

**Deliverable:** diagnostics figures + validation table; update `05-*.md` with any formula changes.

---

## Phase 3 — Component C + main matrix (Weeks 6–8: Dec 1 – Dec 21)
Goal: the paper's main tables.

1. **Implement C** (`reg/reg/anchor.py`): extra forward pass on x_t downsampled by r (area average) at matched time t′; take low band of Δ_t from this pass, high band from full-res. Verify overhead ≈ 1/r² tokens.
2. **Main matrix 2K** (~470 GPU-h): Lumina × {DyPE, SEGA, UltraImage} × {CFG, FDG, PAG, SEG, TPG} × {baseline, REG-AB(D), REG-ABC(D)} × 1,000 prompts.
3. **Main matrix 4K** (~510 GPU-h): Lumina × 3 extrap × {CFG, TPG} × {baseline, REG} × 300 prompts.
4. **FLUX.1-dev case** (~150 GPU-h): distilled guidance; apply REG to the distilled guidance scale + PAG; 2K, 500 prompts.
5. **Ablations** (~150 GPU-h): A only / B only / C only / D only; κ sensitivity; wrong-direction shift (α = 1/r) as a sanity check.
6. **Eval** continuously on CPU/idle GPU as batches finish.

**GATE 2 (Dec 22):** REG beats baselines on patch-FID/KID *and* CLIP or HPS at 2K and 4K for ≥2 extrapolation methods, with CIs. If only partially → still submit to ICML but scope claims down; if not at all → NeurIPS with redesign.

---

## Phase 4 — Large model, benchmarks, theory (Weeks 8–10: Dec 15 – Jan 4, overlaps Phase 3)
1. **Qwen-Image FP8** (~830 GPU-h, start as soon as 2 GPUs free up): CFG ± REG, best extrapolation method (likely SEGA for comparability with the SEGA paper); 1,000 @2K, 200 @4K. Same FP8 for baseline and ours.
2. **GenEval** (553 prompts) and **HPSv2** (800 subset) at 2K for Lumina baseline vs REG.
3. **Repetition metric** full run (200 prompts × 3 res × baseline/REG).
4. **8K qualitative:** 10–20 images, Lumina + SEGA ± REG, for the teaser figure.
5. **Theory** (user + Claude Code, no GPU): Gaussian/linear-diffusion model of per-band SNR under r× resolution → closed-form band-wise time shift and, if tractable, w(r). Show FreCaS/RectifiedHR/URAE settings as special cases. Write `paper/theory.tex` appendix.
6. **Reruns buffer** (~750 GPU-h reserved).

---

## Phase 5 — Writing and submission (Weeks 11–13: Jan 5 – Jan ~25)
1. Jan 5: full draft skeleton in `paper/` (ICML 2027 template); figures auto-generated from `results/*.csv` by `scripts/make_figures.py`.
2. Jan 12: complete draft; internal review (collaborators if onboarded).
3. Jan 15–20: final reruns, limitations section, reproducibility checklist, code cleanup for anonymous release.
4. Jan 20: freeze results. Jan ~22–25: abstract + paper submission (check exact ICML dates).
5. Post-submission: arXiv preprint immediately after (to time-stamp against scooping).

---

## Ongoing tasks (every 2 weeks, 30 min)
- **Scoop check:** arXiv query set from `04-novelty-check.md` restricted to new dates; Semantic Scholar "cited by" for DyPE/SEGA/FDG/TPG; OpenReview. Log in `04-novelty-check.md` under a dated heading.
- **Backups:** `rsync` results and configs to server 1 + external disk; git push.
- **Update `06-*.md`** budget with actual GPU-hours consumed.

## Division of labor
| Claude Code sessions (on servers) | User |
|---|---|
| Env setup, ports (DyPE/SEGA/UltraImage), guidance zoo, REG components, eval pipeline, job runner, figures, LaTeX drafting | GPU audit, data downloads needing accounts, gate decisions, outreach email, theory direction, paper writing/editing, submission |

## Risks and mitigations
| Risk | Mitigation |
|---|---|
| Benchmarks 2× slower than estimated | Phase 0 timing feeds re-budget; drop Qwen 4K first, then 4K matrix breadth |
| DyPE port to Lumina fails/looks bad | Fall back to FLUX.1-dev (distilled CFG) + Chroma (real CFG) as main models |
| REG-(B) only approximates grid optimum | Lead with diagnostics + (C); frame (B) as explanatory |
| Scooped before January | Biweekly checks; arXiv preprint right after submission; collaboration with the SEGA/TPG group converts competitor into co-author |
| ICML slips | Everything carries to NeurIPS 2027 (May) with video extension added |

## Verification (how we know each phase worked)
- Phase 0: `scripts/benchmark.py` table committed; `run_eval.py` on 50 native images returns sane patch-FID/CLIP; a 5-row job file completes across 5 servers with no duplicates.
- Phase 1: side-by-side 4K grid (vanilla vs DyPE) shows reduced repetition; grid-search notebook plot with CIs; gate criteria evaluated explicitly in the notebook.
- Phase 2: REG-with-defaults row lands within CI of grid optimum; diagnostics figures reproducible from saved `.npz`.
- Phase 3–4: `results/main_2k.csv`, `main_4k.csv`, `ablations.csv`, `qwen.csv` regenerate all paper tables via `make_figures.py`; gate 2 criteria met.
- Phase 5: PDF compiles from template; all numbers in text traceable to a CSV row.
