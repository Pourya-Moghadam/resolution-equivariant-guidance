#!/usr/bin/env python
"""Write jobs/reg_anchor_pilot.jsonl: first test of REG (C), the native-resolution anchor (05, 08).

Six pilot prompts, seed 0, dynamic NTK. Anchored CFG with the *native* weight 4 at 2K / 3K / 4K, plus
weights 2 and 6 at 4K to see whether the anchored band behaves as it does at native resolution
(CFG 2 works at 1K and collapses at 4K). Baselines exist: sweep B CFG 4/8 and the FDG rows.
~4 GPU-h on one card.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
prompts = {}
for l in open(ROOT / "jobs" / "pilot_cfg_sweep_c.jsonl"):
    r = json.loads(l)
    prompts.setdefault(r["tags"]["prompt_name"], r["prompt"])

cells = [(2048, 4.0), (3072, 4.0), (4096, 4.0), (4096, 2.0), (4096, 6.0)]
rows = []
for size, w in cells:
    for name, prompt in prompts.items():
        rows.append({
            "id": f"anchor{w:g}_{size}_{name}", "model": "lumina", "prompt": prompt, "height": size, "width": size,
            "seed": 0, "guidance": w, "anchor": {"scale": w, "ratio": size // 1024}, "extrap": "ntk",
            "tags": {"experiment": "reg_anchor_pilot", "prompt_name": name, "size": size, "cfg": w, "method": "anchor"},
        })
out = ROOT / "jobs" / "reg_anchor_pilot.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out)

# Stage B (2026-10-28 08:30): the Δ-only anchor failed at 4K (08). base=true also takes the coarse band of
# the base prediction from the native pass. 4K w2/w4 (collapse points of plain CFG) and 3K w4.
rows_b = []
for size, w in [(4096, 4.0), (4096, 2.0), (3072, 4.0)]:
    for name, prompt in prompts.items():
        rows_b.append({
            "id": f"anchorbase{w:g}_{size}_{name}", "model": "lumina", "prompt": prompt, "height": size, "width": size,
            "seed": 0, "guidance": w, "anchor": {"scale": w, "ratio": size // 1024, "base": True}, "extrap": "ntk",
            "tags": {"experiment": "reg_anchor_pilot", "prompt_name": name, "size": size, "cfg": w, "method": "anchor_base"},
        })
out_b = ROOT / "jobs" / "reg_anchor_pilot_b.jsonl"
out_b.write_text("".join(json.dumps(r) + "\n" for r in rows_b))
print(len(rows_b), "rows ->", out_b)

# Stage C (2026-10-28 09:30): does the CFG optimum return to native with the base anchor? Complete the
# base-anchored weight sweep: 4K w6/w8 (B has w2/w4), 3K w2/w8 (B has w4). With the plain-CFG sweeps B/C
# this gives optimum-vs-weight curves with and without the anchor at 3K and 4K.
rows_c = []
for size, w in [(4096, 6.0), (4096, 8.0), (3072, 2.0), (3072, 8.0)]:
    for name, prompt in prompts.items():
        rows_c.append({
            "id": f"anchorbase{w:g}_{size}_{name}", "model": "lumina", "prompt": prompt, "height": size, "width": size,
            "seed": 0, "guidance": w, "anchor": {"scale": w, "ratio": size // 1024, "base": True}, "extrap": "ntk",
            "tags": {"experiment": "reg_anchor_pilot", "prompt_name": name, "size": size, "cfg": w, "method": "anchor_base"},
        })
out_c = ROOT / "jobs" / "reg_anchor_pilot_c.jsonl"
out_c.write_text("".join(json.dumps(r) + "\n" for r in rows_c))
print(len(rows_c), "rows ->", out_c)
