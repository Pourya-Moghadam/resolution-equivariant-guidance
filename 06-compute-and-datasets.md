# 06 — Compute, Models, Datasets, Timeline

All numbers are **estimates**. Benchmark them on day 1; real timings can be 2× off either way.

## Hardware
- **5 servers, each with one NVIDIA RTX 5000, 32 GB VRAM.** Assumed Ada generation; confirm with `nvidia-smi`, since the older 16 GB Turing card has a similar name.
- The workload is inference only. Each server takes every 5th prompt (`idx % 5`); results go to shared storage or rsync. No multi-node code needed.
- An RTX 5000 Ada is roughly 0.4–0.5× an A100 for this workload. FlashAttention-2 and FP8 work on it; FlashAttention-3 doesn't (Hopper only).
- Note: the laptop used for planning (Intel Arc 140V, no CUDA) is for development and writing only.

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

## Estimated time per image (RTX 5000 Ada, with CFG)

| Model | 1K | 2K | 4K |
|---|---|---|---|
| Lumina-Image 2.0 | ~9 s | ~45 s | ~7 min |
| Qwen-Image (FP8) | ~45 s | ~4–5 min | ~35–45 min |
| FLUX.1-dev | ~20 s | ~2 min | ~15–20 min |

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

| Purpose | Dataset | Size |
|---|---|---|
| Main quality (FID/KID, CLIP) | High-res captions: LAION-HR / LAION-Aesthetics, or the Aesthetic-4K eval split (check its count) | 1,000 prompts @2K; 300 @4K |
| Real reference images for FID/KID | Unsplash Lite (~25K free photos) or LAION-HR subset | ~10K images ≥2K (~50–150 GB) |
| Prompt adherence | GenEval | 553 prompts |
| Human preference | HPSv2 benchmark (3,200 prompts, 4 styles) | 800 subset |
| Qualitative / quick | DrawBench | 200 prompts |
| Repetition metric | Our own single-subject prompts + detector (e.g. GroundingDINO counts) | ~200 prompts |
| Grid search | Random subset of the main prompt set | 100–200 prompts |

- **FID at 4K:** use **patch FID** (e.g. 300 images × 16 crops of 1024² ≈ 5K patches) plus KID; standard FID is unreliable with a few hundred images.

**Storage:** weights (Lumina ~6 GB, Qwen-Image ~40 GB bf16 / ~20 GB FP8, FLUX ~24 GB) plus outputs (~30K images @2K, ~5K @4K ≈ 300–500 GB as PNG). Plan for **1–2 TB**; keep full PNGs only for evaluation sets.

## Software
- Linux (Ubuntu) preferred, or WSL2. FlashAttention on native Windows is painful.
- PyTorch 2.x, diffusers, FlashAttention-2, torchao (FP8), `torch.compile`.

## Timeline (starting 2026-10-24)

| When | Milestone |
|---|---|
| Early November | Environment, porting DyPE/SEGA to Lumina, benchmarking timings |
| Mid-November | Kill-or-go result → go / stop decision |
| Late Nov – Dec | Diagnostics, REG (A)(B)(D), then (C) |
| December – early January | Main matrix, Qwen-Image, ablations |
| January | Writing |
| **Late January 2027** | **ICML 2027 submission** (fallback: NeurIPS 2027, May) |
