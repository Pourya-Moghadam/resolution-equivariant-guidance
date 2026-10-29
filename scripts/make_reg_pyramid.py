#!/usr/bin/env python
"""Write jobs/reg_pyramid_pilot.jsonl: the pyramid anchor and the super-native weight at 4K on the six pilot prompts
(08, 2026-10-29), next to the stage B/C base-anchored rows (`reg_anchor_pilot`, w4 single level).

  variants at w4, base=True:  pyr   levels [2]                (1K + 2K passes)
                              sup   super_scale 1             (no guidance above the native Nyquist)
                              pyrs  levels [2] + super_scale 1
  python scripts/make_reg_pyramid.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = [json.loads(l) for l in open(ROOT / "jobs" / "reg_anchor_pilot_b.jsonl")]
base4k = [r for r in src if r["height"] == 4096 and r["tags"]["cfg"] == 4.0 and r["anchor"].get("base")]
assert len(base4k) == 6, len(base4k)
variants = {"pyr": {"levels": [2]}, "sup": {"super_scale": 1.0}, "pyrs": {"levels": [2], "super_scale": 1.0}}
rows = []
for name, extra in variants.items():
    for r in base4k:
        row = json.loads(json.dumps(r))
        row["anchor"] = {**row["anchor"], **extra}
        row["id"] = f"{name}4_4096_{row['tags']['prompt_name']}"
        row["tags"] = {**row["tags"], "experiment": "reg_pyramid_pilot", "method": f"anchor_base_{name}"}
        rows.append(row)
rows.sort(key=lambda r: (r["tags"]["prompt_name"], r["id"]))  # interleave variants so a partial run compares
out = ROOT / "jobs" / "reg_pyramid_pilot.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out, f"~{len(rows) * 760 / 3600:.1f} GPU-h")
