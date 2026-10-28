#!/usr/bin/env python
"""Write jobs/reg_anchor_grid.jsonl: base-anchored CFG at the native weight on the kill-or-go prompts, so
the headline comparison (anchored w4 vs plain CFG at every weight) has the grid's own baselines — same
prompts, seeds, sampler — at 3K (100 prompts) and 4K (50 prompts). ~18 GPU-h on one card.

  python scripts/make_reg_anchor_grid.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PER_HALF = {3072: 50, 4096: 25}  # as in make_kill_or_go.py
W = 4.0

pool = [json.loads(l) for l in open(ROOT / "prompts" / "kill_or_go_200.jsonl")]
a4k = [p for p in pool if "ref_file" in p]
single = [p for p in pool if "subject" in p]
seed_of = {p["id"]: i for i, p in enumerate(pool)}

rows = []
for size, n in PER_HALF.items():
    for p in a4k[:n] + single[:n]:
        row = {
            "id": f"anchor_{size}_w{W:g}_{p['id']}", "model": "lumina", "prompt": p["prompt"], "height": size, "width": size,
            "seed": seed_of[p["id"]], "guidance": W, "anchor": {"scale": W, "ratio": size // 1024, "base": True},
            "extrap": "ntk",
            "tags": {"experiment": "reg_anchor_grid", "size": size, "cfg": W, "method": "anchor_base",
                     "prompt_id": p["id"], "prompt_set": "a4k" if "ref_file" in p else "single"},
        }
        for k in ("subject", "expected_count", "ref_file"):
            if k in p:
                row[k] = p[k]
        rows.append(row)
rows.sort(key=lambda r: (r["tags"]["prompt_id"], r["height"]))  # interleave sizes so partial runs cover both
out = ROOT / "jobs" / "reg_anchor_grid.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
h = sum({3072: 290, 4096: 700}[r["height"]] for r in rows) / 3600
print(len(rows), "rows ->", out, f"~{h:.0f} GPU-h")
