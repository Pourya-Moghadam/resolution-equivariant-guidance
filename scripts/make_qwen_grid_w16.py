#!/usr/bin/env python
"""Write jobs/qwen_grid_3k_w16.jsonl (08, 2026-11-03 13:00): the strong plain baseline for the Qwen 3K grid. The weight pilot
(jobs/qwen_pilot_w.jsonl) put plain CFG 16 at HPS 28.0 / patch-KID 0.021 on the six pilot prompts, close to REG w4 (28.7 /
0.024) and far above plain w4 (22.3 / 0.053), the Lumina pattern (best plain weight rises with resolution). Same 50 prompts
and seeds as jobs/qwen_grid_3k.jsonl, ids qcfg_3072_w16_<pid>, output dir qwen_grid. 50 rows, ~18 GPU-h.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = [json.loads(l) for l in open(ROOT / "jobs" / "qwen_grid_3k.jsonl")]
rows = []
for r in base:
    if r["tags"]["method"] != "cfg":
        continue
    rows.append({**r, "id": f"qcfg_3072_w16_{r['tags']['prompt_id']}", "guidance": 16.0, "tags": {**r["tags"], "cfg": 16.0}})
assert len(rows) == 50
out = ROOT / "jobs" / "qwen_grid_3k_w16.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out)
