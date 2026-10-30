#!/usr/bin/env python
"""Write jobs/reg_pag_pilot.jsonl (08, 2026-10-30): does "guidance stays native once the base is native" hold for
perturbation guidance too, and does the super-native weight keep growing with the ratio?

  4K, six pilot prompts, PAG at its calibrated native setting (blocks 15-17, scale 2; 08 calibration):
    cfgpag   plain CFG 4 + PAG 2                        (native settings under extrapolation)
    regpag   REG (base anchor, pyramid [2], super 12) w4 + PAG 2   (same PAG on the anchored base)
  5K (5120, ratio 5, pyramid level 2 -> 2560 px pass), REG w4 at super 12 and 24, two prompts:
    tests the r-dependence of the super-native weight (12-16 at 4K, native at 3K) and the memory ceiling.
Output dir reg_pag_pilot; the 5K rows are last so a memory failure there costs nothing else.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = [json.loads(l) for l in open(ROOT / "jobs" / "reg_anchor_pilot_b.jsonl")]
base4k = [r for r in src if r["height"] == 4096 and r["tags"]["cfg"] == 4.0 and r["anchor"].get("base")]
assert len(base4k) == 6
PAG = {"scale": 2.0, "layers": [15, 16, 17]}
REG = {"levels": [2], "super_scale": 12.0}
rows = []
for r in base4k:
    name = r["tags"]["prompt_name"]
    plain = {k: v for k, v in r.items() if k != "anchor"}
    rows.append({**plain, "id": f"cfgpag4_4096_{name}", "pag": PAG,
                 "tags": {**r["tags"], "experiment": "reg_pag_pilot", "method": "cfg_pag"}})
    rows.append({**r, "id": f"regpag4_4096_{name}", "anchor": {**r["anchor"], **REG}, "pag": PAG,
                 "tags": {**r["tags"], "experiment": "reg_pag_pilot", "method": "reg_pag"}})
for name in ("portrait", "lighthouse"):
    r = next(x for x in base4k if x["tags"]["prompt_name"] == name)
    for sup in (12.0, 24.0):
        if sup == 24.0 and name == "lighthouse":
            continue
        rows.append({**r, "id": f"reg_s{sup:g}4_5120_{name}", "height": 5120, "width": 5120,
                     "anchor": {**r["anchor"], "ratio": 5, "levels": [2], "super_scale": sup},
                     "tags": {**r["tags"], "experiment": "reg_pag_pilot", "size": 5120, "method": f"anchor_base_pyr_s{sup:g}"}})
out = ROOT / "jobs" / "reg_pag_pilot.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out)
