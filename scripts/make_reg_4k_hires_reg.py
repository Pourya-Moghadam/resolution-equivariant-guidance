"""Hires fix with REG in the 4K stage, Lumina-Image 2.0 (dynamic NTK), 100 prompts: the 1K stage is plain CFG at the native
weight, the 4K stage (last 35 %, 60 % or 80 % of the 30 steps) uses the REG guidance (anchor + pyramid + super-native 12,
w = 4) in place of plain CFG. Hires fix and HiFlow score slightly above REG on the downscaled scorers and have little or
no native-scale detail; does REG in the second stage add the detail while keeping the 1K layout? Strength 1.0 is REG itself.
Rows are shuffled with a fixed seed. Output: jobs/reg_4k_hires_reg.jsonl (300 rows) -> outputs/reg_4k_hires_reg.
"""
import collections, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in (ROOT / "jobs/kill_or_go.jsonl").read_text().splitlines() if l.strip()]
src = {r["tags"]["prompt_id"]: r for r in rows if r["height"] == 3072 and r["guidance"] == 4.0 and r["cfg_trunc_ratio"] == 1.0}
refs = {r["tags"]["prompt_id"]: r["ref_file"] for r in rows if r.get("ref_file")}
subj = {r["tags"]["prompt_id"]: r["subject"] for r in rows if r.get("subject")}
assert len(src) == 100
out = []
for p in sorted(src):
    s = src[p]
    base = {"model": "lumina", "prompt": s["prompt"], "height": 4096, "width": 4096, "seed": s["seed"], "extrap": "ntk",
            **({"ref_file": refs[p]} if p in refs else {}), **({"subject": subj[p]} if p in subj else {})}
    tags = {"experiment": "reg_4k_hires_reg", "size": 4096, "prompt_id": p, "prompt_set": s["tags"]["prompt_set"]}
    for st in (0.35, 0.60, 0.80):
        name = f"hires_reg_s{int(st * 100):03d}"
        out.append({"id": f"{name}_4096_w4_{p}", **base, "guidance": 4.0,
                    "anchor": {"scale": 4.0, "ratio": 4, "base": True, "levels": [2], "super_scale": 12.0},
                    "hires": {"strength": st, "base_size": 1024, "base_guidance": 4.0}, "tags": {**tags, "cfg": 4.0, "method": name}})
random.Random(0).shuffle(out)
(ROOT / "jobs/reg_4k_hires_reg.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
print(len(out), collections.Counter(r["tags"]["method"] for r in out))
