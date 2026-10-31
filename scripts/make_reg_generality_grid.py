#!/usr/bin/env python
"""Write jobs/reg_generality_grid.jsonl (08, 2026-10-31): the YaRN half of the generality pilot at grid scale.

  reg_yarn / cfg_yarn   REG (pyramid [2], super 12) w4 and plain CFG w4 under dynamic YaRN on the 50 4K kill-or-go prompts
                        (same seeds as reg_pyramid_grid / kill_or_go), 100 rows, ~21 GPU-h  -> output dir reg_generality_grid
  (the reduced-PAG pilot is jobs/reg_pag_scale_pilot.jsonl from scripts/make_reg_pag_scale_pilot.py, 2026-10-30)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
reg = [json.loads(l) for l in open(ROOT / "jobs" / "reg_pyramid_grid.jsonl")]
assert len(reg) == 50 and all(r["tags"]["method"] == "anchor_base_pyr_s12" for r in reg)
rows = []
for r in reg:
    pid = r["tags"]["prompt_id"]
    tags = {**r["tags"], "experiment": "reg_generality_grid"}
    rows.append({**r, "id": f"regyarn_4096_w4_{pid}", "extrap": "yarn", "static_extrap": False, "tags": {**tags, "method": "reg_yarn"}})
    plain = {k: v for k, v in r.items() if k != "anchor"}
    rows.append({**plain, "id": f"cfgyarn_4096_w4_{pid}", "extrap": "yarn", "static_extrap": False, "tags": {**tags, "method": "cfg_yarn"}})
out = ROOT / "jobs" / "reg_generality_grid.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out)

