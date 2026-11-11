#!/usr/bin/env python
"""Write jobs/qwen_drawbench_3k.jsonl (STATUS, 2026-11-11): DrawBench prompt adherence on the second model. The 200 DrawBench
prompts on Qwen-Image at 3K (r = 3072 / 1328 = 2.3), plain CFG at the best swept weight (16) against the REG rule at the
native weight (single native anchor, super-native 12; no intermediate pass, as in the Qwen 3K grid). 400 rows, ~150 GPU-h
(1261 / 1390 s per image on an RTX 5000 Ada). Stock schedule (mu 2.35 at 3K, the value the 4K cells are capped to).
Rows ordered by prompt so a partial run is balanced; seed = prompt index as in jobs/drawbench_4k.jsonl. Output dir
qwen_drawbench_3k; scored per category like the Lumina run.
"""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = Path(os.environ.get("REG_DATA", "/data1/moghadamp/reg_data")) / "datasets" / "drawbench" / "DrawBench Prompts - Sheet1.csv"
prompts = [(r["Prompts"].strip(), r["Category"].strip()) for r in csv.DictReader(open(src, newline="", encoding="utf-8"))]
rows = []
for i, (prompt, cat) in enumerate(prompts):
    pid = f"db{i:03d}"
    base = {"model": "qwen", "prompt": prompt, "height": 3072, "width": 3072, "seed": i, "extrap": "ntk"}
    tags = {"experiment": "qwen_drawbench_3k", "size": 3072, "prompt_id": pid, "prompt_set": "drawbench", "category": cat}
    rows.append({**base, "id": f"qdb_cfg_3072_w16_{pid}", "guidance": 16.0, "tags": {**tags, "cfg": 16.0, "method": "cfg"}})
    rows.append({**base, "id": f"qdb_reg_3072_w4_{pid}", "guidance": 4.0,
                 "anchor": {"scale": 4.0, "ratio": 3, "base": True, "super_scale": 12.0},
                 "tags": {**tags, "cfg": 4.0, "method": "reg"}})
out = ROOT / "jobs" / "qwen_drawbench_3k.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out, "|", len(prompts), "prompts")
