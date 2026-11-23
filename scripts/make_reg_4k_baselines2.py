"""Second baselines job at 4K on Lumina-Image 2.0 (dynamic NTK), after reg_4k_baselines (scripts/make_reg_4k_baselines.py):

  APG to n = 100      apg_proj and apg_full at w = 16 on the 50 prompts the first job did not cover (100 rows).
  re-tuned hires fix  the hires-fix pipeline at strength 0.35 with a stronger second stage, w = 8 and w = 16 (the 1K stage
                      stays at the native weight 4), on all 100 prompts (200 rows): does re-tuning the 4K stage give the
                      two-stage pipeline the native detail it lacks at w = 4 (detail 3.6 against 41.7 for REG)?
Rows are shuffled with a fixed seed so that the index-mod-4 shards mix the slow APG rows and the fast hires rows.
Output: jobs/reg_4k_baselines2.jsonl (300 rows) -> outputs/reg_4k_baselines2.
"""
import collections, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in (ROOT / "jobs/kill_or_go.jsonl").read_text().splitlines() if l.strip()]
src = {r["tags"]["prompt_id"]: r for r in rows if r["height"] == 3072 and r["guidance"] == 4.0 and r["cfg_trunc_ratio"] == 1.0}
refs = {r["tags"]["prompt_id"]: r["ref_file"] for r in rows if r.get("ref_file")}
subj = {r["tags"]["prompt_id"]: r["subject"] for r in rows if r.get("subject")}
first = {r["tags"]["prompt_id"] for r in rows if r["height"] == 4096}
assert len(src) == 100 and len(first) == 50
out = []
for p in sorted(src):
    s = src[p]
    base = {"model": "lumina", "prompt": s["prompt"], "height": 4096, "width": 4096, "seed": s["seed"], "extrap": "ntk",
            **({"ref_file": refs[p]} if p in refs else {}), **({"subject": subj[p]} if p in subj else {})}
    tags = {"experiment": "reg_4k_baselines2", "size": 4096, "prompt_id": p, "prompt_set": s["tags"]["prompt_set"]}
    for w in (8.0, 16.0):
        out.append({"id": f"hires_s035_4096_w{int(w)}_{p}", **base, "guidance": w, "cfg_trunc_ratio": 1.0,
                    "hires": {"strength": 0.35, "base_size": 1024, "base_guidance": 4.0}, "tags": {**tags, "cfg": w, "method": "hires_s035"}})
    if p not in first:
        for mom, nt, name in ((0.0, 0.0, "apg_proj"), (-0.5, 120.0, "apg_full")):
            out.append({"id": f"{name}_4096_w16_{p}", **base, "guidance": 16.0, "cfg_trunc_ratio": 1.0,
                        "apg": {"eta": 0.0, "momentum": mom, "norm_threshold": nt}, "tags": {**tags, "cfg": 16.0, "method": name}})
random.Random(0).shuffle(out)
(ROOT / "jobs/reg_4k_baselines2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
print(len(out), collections.Counter((r["tags"]["method"], r["tags"]["cfg"]) for r in out))
for k in range(4):
    print("shard", k, collections.Counter(r["tags"]["method"][:3] for r in out[k::4]))
