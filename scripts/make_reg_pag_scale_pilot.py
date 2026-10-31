#!/usr/bin/env python
"""Write jobs/reg_pag_scale_pilot.jsonl and jobs/reg_5k_pilot.jsonl (08, 2026-10-30 23:30), the follow-ups to the PAG / 5K pilot.

  reg_pag_scale_pilot: PAG at its native scale 2 on the REG base over-perturbs at 4K (-3.9 HPS on 6/6 prompts, illustration
    look), so the perturbation term does NOT stay native: sweep PAG scale {0.5, 1.0} on the REG base, 4K, six prompts (12 rows,
    ~3.8 GPU-h) to see whether a token-count-scaled PAG (REG component D) recovers the 1K behaviour. Output dir reg_pag_pilot.
  reg_5k_pilot: the r-dependence of the super-native weight — 5K (ratio 5), six prompts, super {12, 16, 24}, minus the three
    rows already in reg_pag_pilot (15 rows, ~7 GPU-h, 21.5 GB peak). Output dir reg_pag_pilot (same ids as the pilot would use).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = [json.loads(l) for l in open(ROOT / "jobs" / "reg_anchor_pilot_b.jsonl")]
base4k = [r for r in src if r["height"] == 4096 and r["tags"]["cfg"] == 4.0 and r["anchor"].get("base")]
assert len(base4k) == 6
REG = {"levels": [2], "super_scale": 12.0}
done = {json.loads(l)["id"] for l in open(ROOT / "jobs" / "reg_pag_pilot.jsonl")}

pag = []
for r in base4k:
    name = r["tags"]["prompt_name"]
    for s in (0.5, 1.0):
        pag.append({**r, "id": f"regpag{s:g}_4096_{name}", "anchor": {**r["anchor"], **REG}, "pag": {"scale": s, "layers": [15, 16, 17]},
                    "tags": {**r["tags"], "experiment": "reg_pag_pilot", "method": f"reg_pag{s:g}"}})
five = []
for sup in (12.0, 16.0, 24.0):
    for r in base4k:
        name = r["tags"]["prompt_name"]
        row = {**r, "id": f"reg_s{sup:g}4_5120_{name}", "height": 5120, "width": 5120,
               "anchor": {**r["anchor"], "ratio": 5, "levels": [2], "super_scale": sup},
               "tags": {**r["tags"], "experiment": "reg_pag_pilot", "size": 5120, "method": f"anchor_base_pyr_s{sup:g}"}}
        if row["id"] not in done:
            five.append(row)
for name, rows in (("reg_pag_scale_pilot", pag), ("reg_5k_pilot", five)):
    out = ROOT / "jobs" / f"{name}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(len(rows), "rows ->", out)
