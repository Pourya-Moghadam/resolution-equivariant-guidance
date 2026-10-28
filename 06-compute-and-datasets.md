# 06 — Compute, Models, Datasets, Timeline

All numbers are **estimates**. Benchmark them on day 1; real timings can be 2× off either way.

## Hardware
- **5 servers, each with one NVIDIA RTX 5000, 32 GB VRAM.** Assumed Ada generation; confirm with `nvidia-smi`, since the older 16 GB Turing card has a similar name.
- The workload is inference only. Each server takes the rows where `idx % REG_NUM_SERVERS == REG_SERVER_ID`; outputs are rsynced to reg1. No multi-node code needed. See `SERVER_SETUP.md`.
- An RTX 5000 Ada is roughly 0.4–0.5× an A100 for this workload. FlashAttention-2 and FP8 work on it; FlashAttention-3 doesn't (Hopper only).
- Confirmed 2026-10-26 on all 5 (reg1–5 = bri-p7-1..5): RTX 5000 Ada Generation, 32 GB (32,760 MiB), driver 595.x, Ubuntu 24.04, 11 TB `/data1` each, no shared filesystem.

## Models (must be RoPE-based DiTs, ideally with real CFG)

| Role | Model | 32 GB fit |
|---|---|---|
| Main development model | **Lumina-Image 2.0** (2.6B, RoPE, real CFG) | Easily |
| Large model for main results | **Qwen-Image** (20B, RoPE, real CFG; SEGA also evaluates on it) | FP8 weights (~20 GB) + precomputed text embeddings |
| Distilled-guidance case | FLUX.1-dev (DyPE/SEGA main model) | bf16 with text encoders offloaded |
| Optional | Chroma (FLUX-derived, real CFG) | Yes |

- **SD3.5 is not usable:** it has absolute position embeddings, not RoPE.
- **Porting:** DyPE/SEGA code targets FLUX/Qwen; porting to Lumina is ~1 week.

**Memory tips:** cache text embeddings and unload the encoders; run the CFG and perturbation branches one after another; tiled VAE decode at 4K; `torch.compile` + SDPA/flash. Use the same quantization and seeds for baseline and ours.

## Time per image (RTX 5000 Ada, with CFG)

Bold = **measured** on reg1 (2026-10-26, 30 steps, CFG 4, bf16, SDPA, tiled VAE; `results/benchmark_native.csv`, produced by `scripts/sample.py` + `scripts/benchmark.py`). Plain = estimate pending measurement.

| Model | 1K | 2K | 4K | peak VRAM |
|---|---|---|---|---|
| Lumina-Image 2.0 | **17.7 s** (est. 9) | **93 s** (est. 45) | **11.6 min** (est. 7) | **12.3 / 12.3 / 17.4 GB** |
| Qwen-Image (FP8 weight-only, 50 steps, true CFG 4) | **145 s** (est. 45) | **7.4 min** (est. 4–5) | **51.6 min** (est. 35–45) | **19.9 / 20.7 / 25.0 GB** |
| FLUX.1-dev (bf16, 28 steps, embedded guidance) | **20.4 s** (est. 20) | **100 s** (with DyPE schedule, +2% for DyPE) | bf16 **OOM** on reg1 (26 GB + fragmentation + another user's 1.3 GB); FP8 / allocator fix under test | **24.7 GB** at 1K and 2K |

- Lumina is ~2× the estimate at 1K/2K and 1.65× at 4K. 4K costs 7.4× 2K (quadratic attention dominates at 65K tokens); 2K costs 5.3× 1K. 3072² measured too: **282 s, 14.1 GB**.
- **Qwen-Image is 3.2× the estimate at 1K but only 1.6× at 2K** (20B parameters, 50 steps, two CFG passes, FP8 weight-only dequantizes on the fly; 2K costs 3.05× 1K, i.e. per-step overhead dominates at 1K). Measured 2K at 7.4 min and 4K at 51.6 min per image: Phase 3c as planned (1,000 @2K + 200 @4K, ×2 for baseline/REG) ≈ 250 GPU-h at 2K + 345 GPU-h at 4K ≈ 600 GPU-h, under the 830 budgeted. 4K Qwen is ~7 images per GPU-day, so its 4K set stays at a few hundred images. Levers if needed: 30 steps instead of 50 (Qwen's default is conservative), FP8 dynamic-activation kernels (fused fp8 matmul on Ada rather than weight-only dequant), fewer 4K prompts.
- FLUX bf16 needs ~27 GB at 4K; with a 1.3 GB foreign process on reg1 it does not fit. Workers have the whole card.
- **Timestep-schedule extrapolation per model** (matters for fair baselines, see 07): Lumina fixed `shift=6` at every resolution; FLUX dynamic, calibrated to 4096 tokens → mu 3.2 at 2K / 11.5 at 4K (DyPE clamps to 1.15); Qwen dynamic, calibrated to 8192 tokens, `max_shift` 0.9 → mu 1.31 at 2K, and clamping it to 0.9 changes the 2K image by 1.3 pixel levels on average (identical to the eye). Qwen's 4K mu would be ~3.0; test the clamp there before the Qwen 4K baseline.
- **Vanilla extrapolation (no DyPE/SEGA), same prompt and seed** (`$REG_DATA/outputs/smoke/`): 1K clean; 2K coherent subject with duplicated tail/body at the frame edges and smeared background; 3K subject stays ~native size in a small region, rest is a banded texture field; 4K collapses to a flat mean colour (pixel std 18). Ruled out as causes: scheduler shift (Lumina uses fixed `shift=6`, `use_dynamic_shifting=false`, so the sigma schedule is identical at every resolution; the token-count `mu` the pipeline computes is ignored), RoPE table range (positions ≤255 of 512), VAE tiling (active at 2K and fine), numerics (smooth output, no NaNs). Same prompt/seed on the other models at 4K without extrapolation: **FLUX.1-dev collapses to flat grey** even with the schedule clamped (see 07); **Qwen-Image does not collapse but duplicates the subject** (two identical foxes on a blurred background, 51.6 min). **Consequence:** there is no usable vanilla 4K baseline for any of the three models; every ≥2K cell in the grids is extrapolation-method + native guidance vs. + REG, as PLAN.md already assumes. Lumina's resolution-independent schedule is itself a lever for component (B) (FLUX, by contrast, shifts by token count).
- **Re-budget, Phase 1 kill-or-go as planned** (40 settings × 200 prompts at 1K and 2K; 20 × 100 at 4K): 39 + 207 + 385 ≈ **630 GPU-h ≈ 5.3 days on 5 GPUs** continuous (plan said ~350 GPU-h / 3–4 days). 4K is 60% of it: cut 4K to 20 settings × 50 prompts or use fewer steps there if the schedule bites. Later phases scale the same way (~1.8×); refine after FLUX/Qwen are measured.
- Speed-ups not yet tried: `torch.compile` on the transformer, FlashAttention-2 kernels (vs SDPA default), 20–25 steps for grid searches.
- Perturbation guidance adds a third branch (~1.5× cost).
- REG's native anchor (C) adds ~1/r² extra tokens per step (negligible).
- 8K: qualitative figures only (~10–20 images).

## Budget

| Phase | Plan | GPU-h | Time on 5 GPUs |
|---|---|---|---|
| Setup and porting | DyPE/SEGA → Lumina, pipeline | – | ~1 week |
| 1. Kill-or-go | Lumina + DyPE: ~40 settings × 200 prompts at 1K/2K; ~20 × 100 at 4K | ~350 | 3–4 days |
| 2. Diagnostics | Guidance spectra, size, entropy | ~50 | <1 day |
| 3a. Main 2K | Lumina: 3 extrapolation methods × 5 guidance methods × {base, ours} × 1,000 prompts | ~470 | ~4 days |
| 3b. Main 4K | Lumina: 12 cells × 300 prompts | ~510 | 4–5 days |
| 3c. Large model | Qwen-Image FP8: CFG ± ours; 1,000 @2K, 200 @4K | ~830 | ~7 days |
| 3d. FLUX, ablations, GenEval/HPS | | ~400 | 3–4 days |
| Reruns (~30%) | | ~750 | ~6 days |
| **Total** | | **~3,400** | **~4 weeks continuous; 6–8 weeks in practice** |

(The same plan on A100s would be ~2,500 A100-hours; cloud cost ≈ US$4–8K.)

## Datasets
No training data is needed: only prompts plus real high-resolution reference images.

**Decided 2026-10-26** (checked on the Hub: licence, gating, size):

| Purpose | Dataset | Size / status |
|---|---|---|
| Main quality (FID/KID, CLIP) and prompt pool | **Aesthetic-4K** (`zhang0jhon/Aesthetic-4K`, MIT, ungated; Diffusion-4K paper, arXiv 2503.18352): `eval/size_2048` and `eval/size_4096` images with captions | eval split downloaded to `$REG_DATA/datasets/aesthetic_4k` (whole repo 53.8 GB, 14,996 files) |
| Real reference images for patch-FID/KID | the same Aesthetic-4K eval images (native 2K/4K photos) — one dataset for prompts and references | see above |
| Fallback reference pool | `laion/relaion-high-resolution` (parquet, gated-auto, 24.6 GB metadata; images must be fetched) | not needed unless Aesthetic-4K is too small |
| Qualitative / quick | DrawBench (`sayakpaul/drawbench`, Apache-2.0) | 200 prompts, downloaded |
| Repetition metric | Own 100 single-subject prompts (`prompts/single_subject.jsonl`) + GroundingDINO counts (`IDEA-Research/grounding-dino-base`, in transformers 5.x) | to write |
| Grid search (kill-or-go) | 100 Aesthetic-4K captions + the 100 single-subject prompts | 200 |
| Human preference | **Working (2026-10-28)** in a separate venv: HPSv2 (v2.1) and ImageReward-v1.0 via `scripts/run_pref.py`, see [Human-preference scorer](#human-preference-scorer-2026-10-28). (The `image-reward` package breaks against transformers 5, so it cannot live in `.venv`; `reg/eval/image_reward.py` stays as the parked in-venv attempt.) The gate uses patch-FID, KID, CLIP, repetition. | run_pref |
| Prompt adherence | GenEval | Phase 3 only (needs its detector stack) |

Metric implementations: patch-FID/KID with `torch-fidelity`'s Inception features via `torchmetrics`; CLIP score with `open_clip_torch` (`ViT-L-14`, laion2B); `image-reward` package (installed with `--no-build-isolation`; its build metadata imports `pkg_resources`).

- **FID at 4K:** use **patch FID** (e.g. 300 images × 16 crops of 1024² ≈ 5K patches) plus KID; standard FID is unreliable with a few hundred images.

**Storage:** weights (Lumina ~6 GB, Qwen-Image ~40 GB bf16 / ~20 GB FP8, FLUX ~24 GB) plus outputs (~30K images @2K, ~5K @4K ≈ 300–500 GB as PNG). Plan for **1–2 TB**; keep full PNGs only for evaluation sets.

## Software
- Ubuntu with Python 3.12.3 on every server; `.venv` built per server by `scripts/setup_env.sh` from `requirements.lock` (no conda/uv available).
- PyTorch 2.14 (cu130), diffusers main, torchao (FP8), `torch.compile`; FlashAttention-2 to be added.

## Timeline (starting 2026-10-24)

| When | Milestone |
|---|---|
| Early November | Environment, porting DyPE/SEGA to Lumina, benchmarking timings |
| Mid-November | Kill-or-go result → go / stop decision |
| Late Nov – Dec | Diagnostics, REG (A)(B)(D), then (C) |
| December – early January | Main matrix, Qwen-Image, ablations |
| January | Writing |
| **Late January 2027** | **ICML 2027 submission** (fallback: NeurIPS 2027, May) |

## Human-preference scorer (2026-10-28)

HPSv2 and ImageReward run in a **separate venv**, `/data1/moghadamp/reg_pref_venv` (reg1 only so far), because both packages are written against transformers 4.x and the project venv has 5.x. Nothing in `.venv` or `requirements.lock` changed.

- **Installed:** torch 2.14.1+cu130 / torchvision 0.29.1 (same as `.venv`), transformers 4.39.3, tokenizers 0.15.2, huggingface_hub 0.36.2, timm 0.6.13 (pinned by image-reward), fairscale 0.4.13 (`--no-build-isolation`), hpsv2 1.2.0 and image-reward 1.5 (both `--no-deps`), plus diffusers 0.32.2 / accelerate 1.15 / datasets 5.0.1 only because `ImageReward/__init__` eagerly imports its ReFL training module. Recipe: `python3 -m venv reg_pref_venv; pip install --extra-index-url https://download.pytorch.org/whl/cu130 torch==2.14.1+cu130 torchvision==0.29.1+cu130; pip install transformers==4.39.3 timm==0.6.13 diffusers==0.32.2 accelerate datasets ftfy regex einops pandas braceexpand fsspec sentencepiece webdataset pyarrow clint; pip install --no-build-isolation fairscale==0.4.13; pip install --no-deps hpsv2==1.2.0 image-reward==1.5`.
- **Fixes needed:** the hpsv2 wheel on PyPI omits `bpe_simple_vocab_16e6.txt.gz` (copied from `.venv`'s open_clip into `hpsv2/src/open_clip/`; it is the stock OpenAI CLIP vocab). hpsv2 also pins `pytest==7.2.0`, `pytest-split`, `protobuf<4`; not installed (unused at inference). The package's own `hpsv2.score()` reloads the 1.9 GB checkpoint per call and first downloads LAION's ViT-H weights only to overwrite them, so `run_pref.py` builds the model once from the HPS checkpoint alone. ImageReward needs a stub `clip` module (same trick as `reg/eval/image_reward.py`).
- **Checkpoints:** HPSv2 `HPS_v2_compressed.pt` and `HPS_v2.1_compressed.pt` (1.9 GB each) in `$HF_HOME/hub/models--xswu--HPSv2`; ImageReward `ImageReward.pt` + `med_config.json` in `$REG_DATA/weights/imagereward` (already there from the earlier attempt); BERT tokenizer `bert-base-uncased` in `$HF_HOME`.
- **Cost:** model load 19 s (HPS) + 14 s (IR); 53 ms (HPS) + 43 ms (IR) per image on the GPU, plus 0.6 s to decode and downscale a 4096² PNG, so about 0.7 s/image wall. Peak VRAM 3.7 GB (HPS alone), 5.4 GB with both loaded; safe on a shared card.
- **Usage:** `REG_DATA=/data1/moghadamp/reg_data /data1/moghadamp/reg_pref_venv/bin/python scripts/run_pref.py --out $REG_DATA/outputs/<job> --group-by size cfg method --csv results/<job>_pref.csv [--metrics hps ir] [--hps-version v2.0]`. Per-image scores go to `<out>/pref/<id>.json` (resume-safe, like `run_eval.py`). HPS is stored as 100 × cosine (the paper's convention, typical 20–35); IR is the raw reward (−2..2). Images are downscaled to a 1024 longer side with PIL before the scorers' own 224 px resize + centre crop, so neither metric sees resolution, only content.
- **Test on `reg_anchor_pilot`** (39 images, `--group-by size cfg method`): collapsed Δ-only anchor rows at 4K score HPS 6.5 / 12.4 / 15.1 and IR −2.28 / −1.71 / −1.24 at w2 / w4 / w6, against the coherent base-anchored rows at 4K HPS 26.1 (w2, n=3) and 29.6 (w4), IR 1.02 and 1.43; the 2K reference is HPS 31.1, IR 1.49. Both metrics separate collapsed from coherent images by many standard errors, no NaNs.

