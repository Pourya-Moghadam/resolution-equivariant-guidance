#!/usr/bin/env python
"""Write jobs/qwen_4k_mu_nolvl.jsonl (08, 2026-11-09): the Qwen 4K rule cell without the 2048 anchor, at the capped shift
(mu 2.35), on the 38 grid prompts the ablation did not cover (the other 12 are in qwen_4k_abl already). With Qwen's
native side of 1328 px the rule's condition "intermediate anchors at full/2^k while >= 2x native" excludes 2048 at 4K
(2 x 1328 = 2656), so this, not the levels-[2] row copied from Lumina, is the rule as written; on the 12 ablation prompts
it has the best patch-KID (0.015) at equal HPS and is 12 % faster. Ids qreg_4096_w4_mu235nolvl_<pid>, method
reg_mu235nolvl, output dir qwen_4k_abl. 38 rows, ~54 min each.
"""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
have = {json.loads(l)["tags"]["prompt_id"] for l in open(ROOT / "jobs" / "qwen_4k_abl.jsonl")}
rows = []
for l in open(ROOT / "jobs" / "qwen_4k_mu.jsonl"):
    b = json.loads(l)
    if b["tags"]["method"] != "reg_mu235" or b["tags"]["prompt_id"] in have:
        continue
    r = copy.deepcopy(b)
    r["anchor"]["levels"] = []
    r["id"] = f"qreg_4096_w4_mu235nolvl_{r['tags']['prompt_id']}"
    r["tags"] = {**r["tags"], "method": "reg_mu235nolvl"}
    rows.append(r)
assert len(rows) == 38
out = ROOT / "jobs" / "qwen_4k_mu_nolvl.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out)
