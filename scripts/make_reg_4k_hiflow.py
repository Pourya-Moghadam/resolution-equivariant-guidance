"""HiFlow at 4K on Lumina-Image 2.0 (dynamic NTK, the main-table setting), same 100 prompts and seeds as the 4K row.

HiFlow (arXiv:2504.06232) ported from the official FLUX pipeline to our sampler (reg.models.lumina._hiflow): cascade
1K -> 2K -> 4K on one 30-step schedule, the last 16 and 10 steps at 2K and 4K, direction alignment (Butterworth low-pass,
cutoff 0.2, weights 1.0 and 0.25 decaying over the stage) and acceleration alignment (0.5, 0.5): the official script's
values. Two settings of the per-stage guidance weight, which the official script raises with resolution (FLUX guidance
embedding 3.5 -> 4.5 -> 6):
  hiflow_w4      the native weight 4 at every stage (no re-tuning, as REG)
  hiflow_scaled  the official ratios applied to the native weight: 4 -> 5.14 -> 6.86
Output: jobs/reg_4k_hiflow.jsonl (200 rows, shuffled with a fixed seed) -> outputs/reg_4k_hiflow.
"""
import collections, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in (ROOT / "jobs/kill_or_go.jsonl").read_text().splitlines() if l.strip()]
src = {r["tags"]["prompt_id"]: r for r in rows if r["height"] == 3072 and r["guidance"] == 4.0 and r["cfg_trunc_ratio"] == 1.0}
refs = {r["tags"]["prompt_id"]: r["ref_file"] for r in rows if r.get("ref_file")}
subj = {r["tags"]["prompt_id"]: r["subject"] for r in rows if r.get("subject")}
assert len(src) == 100
HF = {"sizes": [2048, 4096], "steps": [16, 10], "alphas": [1.0, 0.25], "betas": [0.5, 0.5], "filter_ratio": [0.2, 0.2], "base_size": 1024, "base_guidance": 4.0}
out = []
for p in sorted(src):
    s = src[p]
    base = {"model": "lumina", "prompt": s["prompt"], "height": 4096, "width": 4096, "seed": s["seed"], "extrap": "ntk",
            **({"ref_file": refs[p]} if p in refs else {}), **({"subject": subj[p]} if p in subj else {})}
    tags = {"experiment": "reg_4k_hiflow", "size": 4096, "prompt_id": p, "prompt_set": s["tags"]["prompt_set"], "cfg": 4.0}
    for name, w in (("hiflow_w4", [4.0, 4.0]), ("hiflow_scaled", [round(4.0 * 4.5 / 3.5, 2), round(4.0 * 6.0 / 3.5, 2)])):
        out.append({"id": f"{name}_4096_{p}", **base, "guidance": 4.0, "cfg_trunc_ratio": 1.0, "hiflow": {**HF, "guidance": w}, "tags": {**tags, "method": name}})
random.Random(0).shuffle(out)
(ROOT / "jobs/reg_4k_hiflow.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
print(len(out), collections.Counter(r["tags"]["method"] for r in out), out[0]["hiflow"])
