#!/usr/bin/env python
"""Write jobs/qwen_pilot_w.jsonl (2026-11-03): plain-CFG weight check on Qwen-Image at 3K. The Qwen pilot (jobs/qwen_pilot.jsonl)
showed plain CFG 4 at HPS 21-22 under three schedules against REG at 28-29; on Lumina the best plain weight rose with
resolution (w16 at 5K), so before the Qwen grid the plain baseline gets w8 and w16 on the six pilot prompts at 3K (stock
schedule, NTK, 50 steps). 12 rows, ~4 GPU-h. Output dir qwen_pilot (scored together with the pilot).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = [json.loads(l) for l in open(ROOT / "jobs" / "reg_anchor_pilot_b.jsonl")]
prompts = {}
for r in src:
    prompts.setdefault(r["tags"]["prompt_name"], r["prompt"])
assert len(prompts) == 6
rows = []
for w in (8.0, 16.0):
    for name, prompt in prompts.items():
        rows.append({"id": f"qcfg{int(w)}_3072_{name}", "model": "qwen", "prompt": prompt, "height": 3072, "width": 3072, "seed": 0,
                     "guidance": w, "extrap": "ntk",
                     "tags": {"experiment": "qwen_pilot", "prompt_name": name, "size": 3072, "cfg": w, "method": "cfg"}})
out = ROOT / "jobs" / "qwen_pilot_w.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out)
