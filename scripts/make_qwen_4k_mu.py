#!/usr/bin/env python
"""Write jobs/qwen_4k_mu.jsonl (08, 2026-11-06 23:50): the Qwen 4K cell under a capped dynamic shift. The texture ablation
(jobs/qwen_4k_abl.jsonl) found shift mu 2.35 (the stock 3K value) in place of the stock 4K 3.79 gives REG photographic
texture at 4K (patch-KID 0.019 vs 0.028 on the same 12 prompts, HPS unchanged). Rows, output dir qwen_4k_abl (the 12
existing reg_mu235 rows are skipped by the runner):
  qreg_4096_w4_mu235_<pid>        REG rule at mu 2.35, all 50 grid prompts (38 new)        method reg_mu235
  qcfg_4096_w16_mu235_<pid>       plain CFG 16 at mu 2.35, all 50 (the strong baseline under the same schedule)  cfg_mu235
  qreg_4096_w4_mu235nolvl_<pid>   mu 2.35 without the 2048 anchor, the ablation's 12 prompts (the two helpful changes together)
112 rows, ~61 min each, ordered by prompt so a partial run is balanced.
"""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
grid = [json.loads(l) for l in open(ROOT / "jobs" / "qwen_grid_4k.jsonl")]
reg = sorted((r for r in grid if r["tags"]["method"] == "reg"), key=lambda r: r["tags"]["prompt_id"])
cfg = {r["tags"]["prompt_id"]: r for r in grid if r["tags"]["method"] == "cfg"}
abl_pids = {json.loads(l)["tags"]["prompt_id"] for l in open(ROOT / "jobs" / "qwen_4k_abl.jsonl")}
rows = []
for b in reg:
    pid = b["tags"]["prompt_id"]
    r = copy.deepcopy(b); r.update(shift_mu=2.35, id=f"qreg_4096_w4_mu235_{pid}")
    r["tags"] = {**r["tags"], "experiment": "qwen_4k_abl", "method": "reg_mu235"}
    rows.append(r)
    c = copy.deepcopy(cfg[pid]); c.update(shift_mu=2.35, guidance=16.0, id=f"qcfg_4096_w16_mu235_{pid}")
    c["tags"] = {**c["tags"], "experiment": "qwen_4k_abl", "method": "cfg_mu235", "cfg": 16.0}
    rows.append(c)
    if pid in abl_pids:
        n = copy.deepcopy(r); n["anchor"]["levels"] = []; n["id"] = f"qreg_4096_w4_mu235nolvl_{pid}"
        n["tags"] = {**n["tags"], "method": "reg_mu235nolvl"}
        rows.append(n)
assert len(rows) == 112 and len({r["id"] for r in rows}) == 112
out = ROOT / "jobs" / "qwen_4k_mu.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out)
