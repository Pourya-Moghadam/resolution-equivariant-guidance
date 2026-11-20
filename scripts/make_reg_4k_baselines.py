"""Two baselines at 4K on Lumina-Image 2.0 (dynamic NTK, the main-table setting), same prompts and seeds as the 4K row:

  hires fix   sample at 1K with the native weight, Lanczos-upscale the decoded image 4x, VAE-encode, noise to the
              sigma of step 30 - round(30 * strength) and run the remaining steps at 4K with plain CFG at the native
              weight (reg.models.lumina, ``hires``). The practitioners' two-stage pipeline and the simplest member
              of the latent-anchoring family. Strengths 0.35 and 0.6, 100 prompts each.
  APG         Adaptive Projected Guidance (eta 0, momentum -0.5) at the re-tuned plain weight w = 16: does a fix for
              high-weight oversaturation rescue the re-tuned weight? Without a norm threshold and with one of 120
              (SDXL's default 15 at equal per-element size: x2 for 16 latent channels, x4 for 4K), on the 50 prompts
              of the original 4K cells.
Output: jobs/reg_4k_baselines.jsonl (300 rows) -> outputs/reg_4k_baselines.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in (ROOT / "jobs/kill_or_go.jsonl").read_text().splitlines() if l.strip()]
src = {r["tags"]["prompt_id"]: r for r in rows if r["height"] == 3072 and r["guidance"] == 4.0 and r["cfg_trunc_ratio"] == 1.0}
refs = {r["tags"]["prompt_id"]: r["ref_file"] for r in rows if r.get("ref_file")}
first = {r["tags"]["prompt_id"] for r in rows if r["height"] == 4096}
assert len(src) == 100 and len(first) == 50
out = []
for p in sorted(src):
    s = src[p]
    base = {"model": "lumina", "prompt": s["prompt"], "height": 4096, "width": 4096, "seed": s["seed"], "extrap": "ntk",
            **({"ref_file": refs[p]} if p in refs else {})}
    tags = {"experiment": "reg_4k_baselines", "size": 4096, "prompt_id": p, "prompt_set": s["tags"]["prompt_set"]}
    for st, name in ((0.35, "hires_s035"), (0.6, "hires_s060")):
        out.append({"id": f"{name}_4096_w4_{p}", **base, "guidance": 4.0, "cfg_trunc_ratio": 1.0,
                    "hires": {"strength": st, "base_size": 1024, "base_guidance": 4.0}, "tags": {**tags, "cfg": 4.0, "method": name}})
    if p in first:
        for nt, name in ((0.0, "apg"), (120.0, "apg_nt120")):
            out.append({"id": f"{name}_4096_w16_{p}", **base, "guidance": 16.0, "cfg_trunc_ratio": 1.0,
                        "apg": {"eta": 0.0, "momentum": -0.5, "norm_threshold": nt}, "tags": {**tags, "cfg": 16.0, "method": name}})
(ROOT / "jobs/reg_4k_baselines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
import collections; print(len(out), collections.Counter(r["tags"]["method"] for r in out))
