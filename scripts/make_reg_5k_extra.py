#!/usr/bin/env python
"""Write jobs/reg_5k_plain_w48.jsonl (plain CFG w4 and w8 at 5K, 100 rows, ~41 GPU-h) and jobs/reg_5k_pyr_only.jsonl
(the pyramid-only ablation at 5K, 50 rows, ~23 GPU-h) on the same 50 prompts as jobs/reg_5k_grid.jsonl (08, 2026-11-02).
With reg_5k_grid's REG rule cell and plain w16 these give the 5K weight curve (plain w4/8/16 vs REG w4) and the 5K ablation
row. Output dir reg_5k_grid for all of them.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
grid = [json.loads(l) for l in open(ROOT / "jobs" / "reg_5k_grid.jsonl")]
plain16 = [r for r in grid if r["tags"]["method"] == "cfg"]
reg = [r for r in grid if r["tags"]["method"] != "cfg"]
assert len(plain16) == 50 and len(reg) == 50
plain = []
for w in (4.0, 8.0):
    for r in plain16:
        pid = r["tags"]["prompt_id"]
        plain.append({**r, "id": f"cfg_5120_w{w:g}_{pid}", "guidance": w, "tags": {**r["tags"], "cfg": w}})
plain.sort(key=lambda r: (r["tags"]["prompt_id"], r["tags"]["cfg"]))
pyr = [{**r, "id": f"reg_pyr_5120_w4_{r['tags']['prompt_id']}", "anchor": {k: v for k, v in r["anchor"].items() if k != "super_scale"},
        "tags": {**r["tags"], "method": "anchor_base_pyr"}} for r in reg]
for name, rows in (("reg_5k_plain_w48", plain), ("reg_5k_pyr_only", pyr)):
    out = ROOT / "jobs" / f"{name}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(len(rows), "rows ->", out)
