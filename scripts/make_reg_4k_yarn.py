"""REG against the re-tuned plain weight under DyPE in its authors' default configuration (YaRN frequency rescaling with
the kappa(t) = t^2 schedule, "dynamic YaRN"), Lumina-Image 2.0 at 4K, 100 prompts like the main 4K row.
reg_generality_grid already has REG w4 and plain w4 under YaRN on the first 50 prompts. This adds:
  plain CFG w16 on all 100 prompts (the comparison the paper's claim rests on), REG w4 and plain w4 on the other 50,
  plain CFG w8 on the first 50 (checks that w16 is still the best plain weight under YaRN).
Prompts, seeds and reference files as in scripts/make_reg_4k_ext.py. Output: jobs/reg_4k_yarn.jsonl -> outputs/reg_4k_yarn.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in (ROOT / "jobs/kill_or_go.jsonl").read_text().splitlines() if l.strip()]
src = {r["tags"]["prompt_id"]: r for r in rows if r["height"] == 3072 and r["guidance"] == 4.0 and r["cfg_trunc_ratio"] == 1.0}
refs = {r["tags"]["prompt_id"]: r["ref_file"] for r in rows if r.get("ref_file")}
first = {json.loads(l)["tags"]["prompt_id"] for l in (ROOT / "jobs/reg_generality_grid.jsonl").read_text().splitlines() if l.strip()}
assert len(src) == 100 and len(first) == 50 and first <= set(src)
out = []
for p in sorted(src):
    s = src[p]
    base = {"model": "lumina", "prompt": s["prompt"], "height": 4096, "width": 4096, "seed": s["seed"], "extrap": "yarn", "static_extrap": False,
            **({"ref_file": refs[p]} if p in refs else {})}
    tags = {"experiment": "reg_4k_yarn", "size": 4096, "prompt_id": p, "prompt_set": s["tags"]["prompt_set"]}
    plain = lambda w: {"id": f"cfgyarn_4096_w{w}_{p}", **base, "guidance": float(w), "cfg_trunc_ratio": 1.0,
                       "tags": {**tags, "cfg": float(w), "method": "cfg_yarn", "trunc": 1.0}}
    out.append(plain(16))
    if p in first:
        out.append(plain(8))
    else:
        out.append(plain(4))
        out.append({"id": f"regyarn_4096_w4_{p}", **base, "guidance": 4.0,
                    "anchor": {"scale": 4.0, "ratio": 4, "base": True, "levels": [2], "super_scale": 12.0},
                    "tags": {**tags, "cfg": 4.0, "method": "reg_yarn"}})
(ROOT / "jobs/reg_4k_yarn.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
import collections; print(len(out), "rows", collections.Counter((r["tags"]["method"], r["tags"]["cfg"]) for r in out))
