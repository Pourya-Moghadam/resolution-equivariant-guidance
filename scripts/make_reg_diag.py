#!/usr/bin/env python
"""Write jobs/reg_diag.jsonl: guidance diagnostics (PLAN Phase 2 step 1, pulled forward 2026-10-28).

Six pilot prompts, seed 0, 30 steps, with reg.diag recording at every step. Rows:
  CFG 4 (the native default) at 1K / 2K / 3K / 4K  — what native guidance looks like under extrapolation
  CFG 8 at 3K / 4K                                 — the measured optimum there, for comparison
Band cutoffs recorded: down {2, 4, 6, 8, 12, 16} (2·r is FDG's native cutoff in cycles per image).
~3.5 GPU-h on one card.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
prompts = {}
for l in open(ROOT / "jobs" / "pilot_cfg_sweep_c.jsonl"):
    r = json.loads(l)
    prompts.setdefault(r["tags"]["prompt_name"], r["prompt"])

cells = [(s, 4.0) for s in (1024, 2048, 3072, 4096)] + [(3072, 8.0), (4096, 8.0)]
rows = []
for size, cfg in cells:
    for name, prompt in prompts.items():
        rows.append({
            "id": f"diag_cfg{cfg:g}_{size}_{name}", "model": "lumina", "prompt": prompt, "height": size, "width": size,
            "seed": 0, "guidance": cfg, "extrap": "none" if size == 1024 else "ntk",
            "diag": {"downs": [2, 4, 6, 8, 12, 16]},
            "tags": {"experiment": "reg_diag", "prompt_name": name, "size": size, "cfg": cfg},
        })
rows.sort(key=lambda r: (r["height"], r["guidance"]))
out = ROOT / "jobs" / "reg_diag.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out)
