#!/usr/bin/env python
"""Write jobs/reg_pilot_fdg.jsonl: first direct test of REG-A (05) — FDG band cutoff fixed vs scaled by
the resolution ratio. Same 6 prompts and seed 0 as sweeps B/C, whose CFG rows are the baselines.

  1K: FDG (w_low, w_high) with down = 2 (native reference)
  4K: same pairs with down = 2 (fixed, i.e. FDG as published) and down = 8 (= 2 x 4096/1024, REG-A)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [(2, 6), (4, 8), (8, 4)]

prompts = {}
for l in open(ROOT / "jobs" / "pilot_cfg_sweep_c.jsonl"):
    r = json.loads(l)
    prompts.setdefault(r["tags"]["prompt_name"], r["prompt"])

rows = []
for name, prompt in prompts.items():
    for wl, wh in PAIRS:
        for size, downs in ((1024, [2]), (4096, [2, 8])):
            for down in downs:
                rows.append({
                    "id": f"fdg_{wl}_{wh}_d{down}_{size}_{name}", "model": "lumina", "prompt": prompt,
                    "height": size, "width": size, "seed": 0, "extrap": "none" if size == 1024 else "ntk",
                    "fdg": {"w_low": float(wl), "w_high": float(wh), "down": float(down)},
                    "tags": {"experiment": "reg_pilot_fdg", "prompt_name": name, "size": size, "down": down,
                             "pair": f"{wl}/{wh}"},
                })
rows.sort(key=lambda r: r["height"])  # 1K first: a quick sanity check before the 4K rows
out = ROOT / "jobs" / "reg_pilot_fdg.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out)

# Stage B (2026-10-27, after the first 4K rows): only w_low = 8 rescued 4K; w_low 2-4 collapse whatever the
# cutoff. So the 4K guidance need sits in the low band. Map w_low and w_high separately at 4K, and the cutoff
# at the one working pair.
STAGE_B = [(6, 4, 2), (12, 4, 2), (8, 2, 2), (8, 6, 2), (8, 4, 4), (8, 2, 8)]
rows_b = [{
    "id": f"fdg_{wl}_{wh}_d{d}_4096_{name}", "model": "lumina", "prompt": prompt, "height": 4096, "width": 4096,
    "seed": 0, "extrap": "ntk", "fdg": {"w_low": float(wl), "w_high": float(wh), "down": float(d)},
    "tags": {"experiment": "reg_pilot_fdg", "prompt_name": name, "size": 4096, "down": d, "pair": f"{wl}/{wh}"},
} for wl, wh, d in STAGE_B for name, prompt in prompts.items()]
out_b = ROOT / "jobs" / "reg_pilot_fdg_b.jsonl"
out_b.write_text("".join(json.dumps(r) + "\n" for r in rows_b))
print(len(rows_b), "rows ->", out_b)
