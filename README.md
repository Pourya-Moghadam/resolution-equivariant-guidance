# REG research: Resolution-Equivariant Guidance for Diffusion Transformers

Research notes from the planning stage (2026-10-24). The goal is a top-tier paper (ICML/NeurIPS/ICLR/CVPR) in diffusion models, building on the research track of [name removed] (University of Toronto).

## Files

| File | Contents |
|---|---|
| [PLAN.md](PLAN.md) | **Approved project plan**: 5 phases, 2 gates, timeline to ICML 2027 (late Jan 2027) |
| [SERVER_SETUP.md](SERVER_SETUP.md) | How everything is run from reg1 (hub): code, data and results moving to and from workers reg2–5; how jobs are split across servers |
| [03-literature-review.md](03-literature-review.md) | Recent work on guidance and on resolution extrapolation, and the gap between them |
| [04-novelty-check.md](04-novelty-check.md) | Search log for the novelty check, detailed reading of the closest prior art, and the verdict |
| [05-method-and-contributions.md](05-method-and-contributions.md) | The proposed method (REG) and the paper's contributions |
| [06-compute-and-datasets.md](06-compute-and-datasets.md) | Model choice, compute budget on our 5× RTX 5000 servers, datasets, and timeline |

## Summary in one paragraph

Papers on resolution extrapolation (DyPE, SEGA, UltraImage) fix the model's RoPE and attention but keep guidance at its native-resolution settings. Papers on guidance (CFG variants, FDG, PAG/SEG/TPG) test only at native resolution. **Nobody has studied how guidance behaves at extrapolated resolution.** Our idea is **Resolution-Equivariant Guidance (REG)**. It moves guidance into image-relative frequency and band-wise matched signal-to-noise, uses an optional native-resolution anchor for the low-frequency guidance, and calibrates perturbation guidance to token count. It's training-free, adds no new hyperparameters, and plugs into any guidance method and any extrapolation method. The novelty check (arXiv, Semantic Scholar citation tracing, OpenReview ICLR 2027) found no prior work on the core idea. The closest prior art is FreCaS, RectifiedHR and ScaleCrafter, all hand-tuned and none about DiTs with RoPE.

## Status / next steps
**Current phase:** Phase 0 (environment, baselines, measurement), Oct 27 – Nov 2. See [PLAN.md](PLAN.md).

Decisions (2026-10-24): target ICML 2027; servers run Linux; work done by the user + Claude Code sessions; contact the [name removed]/[name removed] group after Gate 1 passes.

- [ ] From reg1, `bash scripts/fanout.sh 'bash ~/reg/scripts/server_audit.sh'` and paste the output into 06-compute-and-datasets.md (GPU confirmed RTX 5000 Ada, 32 GB on all 5)
- [ ] Read the FreCaS (SD3 setting) and RectifiedHR full texts once more before writing related work
- [ ] Port DyPE (and SEGA) to Lumina-Image 2.0 (~1 week)
- [ ] Kill-or-go experiment: guidance grid search at 1K / 2K / 4K (~3–4 days on 5 GPUs)
- [ ] Re-check arXiv every ~2 weeks for scooping (watch [name removed]/[name removed]/[name removed] and Sadat et al.)
- [ ] Target: ICML 2027 (late January 2027); fallback NeurIPS 2027 (May 2027)
