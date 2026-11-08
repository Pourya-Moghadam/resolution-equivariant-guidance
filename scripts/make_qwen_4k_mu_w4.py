#!/usr/bin/env python
"""Write jobs/qwen_4k_mu_w4.jsonl (08, 2026-11-07 22:00): plain CFG 4 at the capped shift (mu 2.35) on the 50 Qwen 4K grid
prompts, so the capped-schedule 4K cell has its native-weight plain row too (REG and plain w16 are in jobs/qwen_4k_mu.jsonl).
Ids qcfg_4096_w4_mu235_<pid>, method cfg_mu235, output dir qwen_4k_mu_w4 (separate from qwen_4k_abl so handoff21's scoring of
that dir is not held up). 50 rows, ~52 min each.
"""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
grid = [json.loads(l) for l in open(ROOT / "jobs" / "qwen_grid_4k.jsonl")]
rows = []
for b in sorted((r for r in grid if r["tags"]["method"] == "cfg"), key=lambda r: r["tags"]["prompt_id"]):
    r = copy.deepcopy(b); r.update(shift_mu=2.35, id=f"qcfg_4096_w4_mu235_{r['tags']['prompt_id']}")
    r["tags"] = {**r["tags"], "experiment": "qwen_4k_abl", "method": "cfg_mu235"}
    rows.append(r)
assert len(rows) == 50
out = ROOT / "jobs" / "qwen_4k_mu_w4.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out)
