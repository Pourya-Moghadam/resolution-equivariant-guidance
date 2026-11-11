#!/usr/bin/env python
"""Write jobs/reg_6k_grid.jsonl (STATUS, 2026-11-11): REG under the rule at 6K (6144^2, r = 6) on the 50 4K/5K grid prompts,
with the best plain-CFG weight of the grid (w16) as the baseline. 100 rows, ~90 GPU-h (3142 s / 26.7 GB per REG image).

  REG rule: base anchored at native (ratio 6), one intermediate pass at full/2 = 3072 px (3x native; full/4 = 1536 is
  under 2x and left out), super-native weight 3w = 12, native CFG weight 4.   ids reg_pyr_s12_6144_w4_<pid>, method anchor_base_pyr_s12
  plain: CFG 16, NTK, no truncation.                                           ids cfg_6144_w16_<pid>, method cfg
Output dir reg_6k_grid. Rows sorted by (prompt, method) so every 4-way shard holds both methods for its prompts.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pool = [json.loads(l) for l in open(ROOT / "prompts" / "kill_or_go_200.jsonl")]
a4k = [p for p in pool if "ref_file" in p][:25]
single = [p for p in pool if "subject" in p][:25]
seed_of = {p["id"]: i for i, p in enumerate(pool)}
rows = []
for p in a4k + single:
    common = {"model": "lumina", "prompt": p["prompt"], "height": 6144, "width": 6144, "seed": seed_of[p["id"]], "extrap": "ntk"}
    for k in ("subject", "expected_count", "ref_file"):
        if k in p:
            common[k] = p[k]
    tags = {"experiment": "reg_6k_grid", "size": 6144, "prompt_id": p["id"], "prompt_set": "a4k" if "ref_file" in p else "single"}
    rows.append({"id": f"reg_pyr_s12_6144_w4_{p['id']}", **common, "guidance": 4.0,
                 "anchor": {"scale": 4.0, "ratio": 6, "base": True, "levels": [2], "super_scale": 12.0},
                 "tags": {**tags, "cfg": 4.0, "method": "anchor_base_pyr_s12"}})
    rows.append({"id": f"cfg_6144_w16_{p['id']}", **common, "guidance": 16.0, "cfg_trunc_ratio": 1.0,
                 "tags": {**tags, "cfg": 16.0, "trunc": 1.0, "method": "cfg"}})
out = ROOT / "jobs" / "reg_6k_grid.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out)
