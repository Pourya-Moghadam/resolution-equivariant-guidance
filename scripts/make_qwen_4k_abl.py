#!/usr/bin/env python
"""Write jobs/qwen_4k_abl.jsonl (08, 2026-11-05 21:00): which component gives REG's painterly, haloed texture at 4K on Qwen.
The 4K grid's REG rows hold HPS (28.8) but their native-pixel crops are brush-like with edge halos, where the 3K REG rows are
clean (patch-KID 0.031 vs 0.014). Four one-change variants of the 4K rule row on 12 of the grid's 50 prompts (every 4th, same
seeds), paired against the grid's rule rows (qwen_grid, method reg):
  reg_s8     super-native weight 8 instead of 12
  reg_nolvl  no intermediate anchor at 2048 (levels [])
  reg_mu235  shift mu 2.35 (the stock 3K value) instead of the stock 4K 3.79 (2 of 50 steps below sigma 0.5)
  reg_mu131  shift mu 1.31 (the stock 2K value)
48 rows, ~61 min each, output dir qwen_4k_abl, ordered by prompt so a partial run is balanced across variants.
"""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = [r for r in map(json.loads, open(ROOT / "jobs" / "qwen_grid_4k.jsonl")) if r["tags"]["method"] == "reg"]
base = sorted(base, key=lambda r: r["tags"]["prompt_id"])[::4][:12]
VARIANTS = {
    "s8": lambda r: r["anchor"].update(super_scale=8.0),
    "nolvl": lambda r: r["anchor"].update(levels=[]),
    "mu235": lambda r: r.update(shift_mu=2.35),
    "mu131": lambda r: r.update(shift_mu=1.31),
}
rows = []
for b in base:
    for name, change in VARIANTS.items():
        r = copy.deepcopy(b)
        change(r)
        pid = r["tags"]["prompt_id"]
        r["id"] = f"qreg_4096_w4_{name}_{pid}"
        r["tags"] = {**r["tags"], "experiment": "qwen_4k_abl", "method": f"reg_{name}"}
        rows.append(r)
assert len(rows) == 48 and len({r["id"] for r in rows}) == 48
out = ROOT / "jobs" / "qwen_4k_abl.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out, "| prompts:", " ".join(b["tags"]["prompt_id"] for b in base))
