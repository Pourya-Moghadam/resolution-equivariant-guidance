#!/usr/bin/env python
"""Write jobs/reg_generality_pilot.jsonl (08, 2026-10-30): REG next to the other perturbation guidances and the
other extrapolation method, to test the "plugs into any guidance / any extrapolation" claim before the Qwen port.

  4K, six pilot prompts (reg_anchor_pilot_b), each guidance at its calibrated native setting (08: blocks 15-17):
    cfgseg / regseg   plain CFG 4 + SEG 1.5 (sigma 5)   vs   REG (base anchor, pyramid [2], super 12) w4 + the same SEG
    cfgtpg / regtpg   plain CFG 4 + TPG 1.5 (frac 0.5)  vs   REG w4 + the same TPG
    cfgyarn / regyarn plain CFG 4 under dynamic YaRN (07's other extrapolation)  vs  REG w4 under dynamic YaRN
Output dir reg_generality_pilot. 36 rows, ~8 GPU-h.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = [json.loads(l) for l in open(ROOT / "jobs" / "reg_anchor_pilot_b.jsonl")]
base4k = [r for r in src if r["height"] == 4096 and r["tags"]["cfg"] == 4.0 and r["anchor"].get("base")]
assert len(base4k) == 6
LAYERS = [15, 16, 17]
PERT = {"seg": {"scale": 1.5, "sigma": 5.0, "layers": LAYERS}, "tpg": {"scale": 1.5, "frac": 0.5, "layers": LAYERS}}
REG = {"levels": [2], "super_scale": 12.0}
rows = []
for r in base4k:
    name = r["tags"]["prompt_name"]
    plain = {k: v for k, v in r.items() if k != "anchor"}
    reg = {**r, "anchor": {**r["anchor"], **REG}}
    tags = {**r["tags"], "experiment": "reg_generality_pilot"}
    for key, p in PERT.items():
        rows.append({**plain, "id": f"cfg{key}4_4096_{name}", key: p, "tags": {**tags, "method": f"cfg_{key}"}})
        rows.append({**reg, "id": f"reg{key}4_4096_{name}", key: p, "tags": {**tags, "method": f"reg_{key}"}})
    rows.append({**plain, "id": f"cfgyarn4_4096_{name}", "extrap": "yarn", "static_extrap": False, "tags": {**tags, "method": "cfg_yarn"}})
    rows.append({**reg, "id": f"regyarn4_4096_{name}", "extrap": "yarn", "static_extrap": False, "tags": {**tags, "method": "reg_yarn"}})
out = ROOT / "jobs" / "reg_generality_pilot.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out)
