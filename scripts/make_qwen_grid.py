#!/usr/bin/env python
"""Write jobs/qwen_grid_3k.jsonl (08, 2026-11-03; --size 4096 -> jobs/qwen_grid_4k.jsonl, 2026-11-04): REG on Qwen-Image at 3K on the 50 grid prompts (the 25 a4k + 25 single-subject
prompts of the 4K/5K Lumina grids, same seeds), REG rule vs plain CFG at the native weight 4; stock schedule, NTK, 50 steps.
The pilot (jobs/qwen_pilot.jsonl) put plain CFG 4 at HPS 21-22 under three schedules and REG at 28-29. 100 rows, ~37 GPU-h
(1261 / 1385 s per 3K image). Rows sorted by (method, prompt) so a 2-way shard gets both methods. Output dir qwen_grid.
Plain CFG at a higher weight is appended if jobs/qwen_pilot_w.jsonl says it beats w4 (ids qcfg_3072_w<w>_<pid>).
At 4K the REG rule adds the intermediate anchor at full/2 = 2048 px (levels [2]); 100 rows, ~90 GPU-h (52 / 57 min per image).
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=3072)
SIZE = ap.parse_args().size
R = SIZE // 1024
LEVELS = [2] if SIZE // 2 >= 2048 else []

ROOT = Path(__file__).resolve().parents[1]
pool = [json.loads(l) for l in open(ROOT / "prompts" / "kill_or_go_200.jsonl")]
a4k = [p for p in pool if "ref_file" in p][:25]
single = [p for p in pool if "subject" in p][:25]
seed_of = {p["id"]: i for i, p in enumerate(pool)}
rows = []
for p in a4k + single:
    common = {"model": "qwen", "prompt": p["prompt"], "height": SIZE, "width": SIZE, "seed": seed_of[p["id"]], "extrap": "ntk"}
    for k in ("subject", "expected_count", "ref_file"):
        if k in p:
            common[k] = p[k]
    tags = {"experiment": "qwen_grid", "size": SIZE, "prompt_id": p["id"], "prompt_set": "a4k" if "ref_file" in p else "single"}
    rows.append({"id": f"qcfg_{SIZE}_w4_{p['id']}", **common, "guidance": 4.0, "tags": {**tags, "cfg": 4.0, "method": "cfg"}})
    anchor = {"scale": 4.0, "ratio": R, "base": True, "super_scale": 12.0}
    if LEVELS:
        anchor["levels"] = LEVELS
    rows.append({"id": f"qreg_{SIZE}_w4_{p['id']}", **common, "guidance": 4.0, "anchor": anchor,
                 "tags": {**tags, "cfg": 4.0, "method": "reg"}})
rows.sort(key=lambda r: (r["tags"]["method"], r["tags"]["prompt_id"]))
out = ROOT / "jobs" / f"qwen_grid_{SIZE // 1024}k.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out)
