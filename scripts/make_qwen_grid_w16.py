#!/usr/bin/env python
"""Write jobs/qwen_grid_{3k|4k}_w16.jsonl (08, 2026-11-03 13:00; --size 4096 added 2026-11-05): the strong plain baseline for the Qwen grids. The weight pilot
(jobs/qwen_pilot_w.jsonl) put plain CFG 16 at HPS 28.0 / patch-KID 0.021 on the six pilot prompts, close to REG w4 (28.7 /
0.024) and far above plain w4 (22.3 / 0.053), the Lumina pattern (best plain weight rises with resolution). Same 50 prompts
and seeds as jobs/qwen_grid_{3k|4k}.jsonl, ids qcfg_<size>_w16_<pid>, output dir qwen_grid. 50 rows, ~18 GPU-h at 3K, ~50 at 4K.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=3072, choices=[3072, 4096])
SIZE = ap.parse_args().size
K = f"{SIZE // 1024}k"
base = [json.loads(l) for l in open(ROOT / "jobs" / f"qwen_grid_{K}.jsonl")]
rows = []
for r in base:
    if r["tags"]["method"] != "cfg":
        continue
    rows.append({**r, "id": f"qcfg_{SIZE}_w16_{r['tags']['prompt_id']}", "guidance": 16.0, "tags": {**r["tags"], "cfg": 16.0}})
assert len(rows) == 50
out = ROOT / "jobs" / f"qwen_grid_{K}_w16.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out)
