#!/usr/bin/env python
"""Write jobs/reg_5k_depth_pilot.jsonl (08, 2026-11-01 01:00): the 5K follow-up after the super-native sweep.

At 5K (ratio 5) the residual speckle (p>native 0.066 vs 0.042 at 4K) does not shrink with the super-native weight (12/16/24:
0.066/0.067/0.074), so the lever to test is pyramid depth, not weight: pyramid-only (no super-native term, the new default)
with levels [2] (2560 px pass) and levels [2, 4] (2560 + 1280 px passes), six prompts, w4 (12 rows, ~5.5 GPU-h on one card).
Output dir reg_pag_pilot (same prompts and seeds as the 5K sweep, so run_pref/run_eval score them together).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = [json.loads(l) for l in open(ROOT / "jobs" / "reg_anchor_pilot_b.jsonl")]
base4k = [r for r in src if r["height"] == 4096 and r["tags"]["cfg"] == 4.0 and r["anchor"].get("base")]
assert len(base4k) == 6
rows = []
for tag, levels in (("pyr", [2]), ("pyr24", [2, 4])):
    for r in base4k:
        name = r["tags"]["prompt_name"]
        rows.append({**r, "id": f"reg_{tag}4_5120_{name}", "height": 5120, "width": 5120,
                     "anchor": {"scale": 4.0, "ratio": 5, "base": True, "levels": levels},
                     "tags": {**r["tags"], "experiment": "reg_pag_pilot", "size": 5120, "method": f"anchor_base_{tag}"}})
out = ROOT / "jobs" / "reg_5k_depth_pilot.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out)
