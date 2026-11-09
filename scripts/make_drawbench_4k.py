#!/usr/bin/env python
"""Write jobs/drawbench_4k.jsonl (08, 2026-11-09): prompt adherence on a public prompt set. DrawBench (200 prompts, 11
categories) with Lumina at 4K under the three cells of the headline table: plain CFG 4 (native weight), plain CFG 16 (best
plain weight at 4K) and the REG rule at w4 (native anchor, intermediate anchor at 2048, super-native 12). Row fields copied
from the 4K grids (kill_or_go for plain, reg_pyramid_grid for the rule); seed = prompt index. 600 rows, ~11.6 min each
(~116 GPU-h), ordered by prompt so a partial run is balanced; output dir drawbench_4k. Scored with CLIP, HPSv2.1 and
ImageReward per category (no reference images).
"""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = Path(os.environ.get("REG_DATA", "/data1/moghadamp/reg_data")) / "datasets" / "drawbench" / "DrawBench Prompts - Sheet1.csv"
prompts = [(r["Prompts"].strip(), r["Category"].strip()) for r in csv.DictReader(open(src, newline="", encoding="utf-8"))]
rows = []
for i, (prompt, cat) in enumerate(prompts):
    pid = f"db{i:03d}"
    base = {"model": "lumina", "prompt": prompt, "height": 4096, "width": 4096, "seed": i, "extrap": "ntk"}
    tags = {"experiment": "drawbench_4k", "size": 4096, "prompt_id": pid, "prompt_set": "drawbench", "category": cat}
    for w in (4.0, 16.0):
        rows.append({**base, "id": f"db_cfg_4096_w{int(w)}_{pid}", "guidance": w, "cfg_trunc_ratio": 1.0,
                     "tags": {**tags, "cfg": w, "method": "cfg"}})
    rows.append({**base, "id": f"db_reg_4096_w4_{pid}", "guidance": 4.0,
                 "anchor": {"scale": 4.0, "ratio": 4, "base": True, "levels": [2], "super_scale": 12.0},
                 "tags": {**tags, "cfg": 4.0, "method": "reg"}})
out = ROOT / "jobs" / "drawbench_4k.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out, "|", len(prompts), "prompts,", len({c for _, c in prompts}), "categories")
