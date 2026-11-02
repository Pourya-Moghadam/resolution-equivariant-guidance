#!/usr/bin/env python
"""Write jobs/qwen_pilot.jsonl (2026-11-02): first REG run on Qwen-Image. The six pilot prompts of jobs/reg_anchor_pilot_b.jsonl
at 3K and 4K, plain CFG 4 (the native weight) against the REG rule (native anchor at 1024, intermediate anchor at full/2 when
>= 2x native, super-native weight 12), both under NTK extrapolation (DyPE, base 83 patches), 50 steps, seed 0.
24 rows, ~18 GPU-h (Qwen 4K: 52 min / image plain). Output dir qwen_pilot.
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
for size in (3072, 4096):
    r = size // 1024
    levels = [2] if size // 2 >= 2048 else []
    for name, prompt in prompts.items():
        common = {"model": "qwen", "prompt": prompt, "height": size, "width": size, "seed": 0, "guidance": 4.0, "extrap": "ntk"}
        tags = {"experiment": "qwen_pilot", "prompt_name": name, "size": size, "cfg": 4.0}
        rows.append({"id": f"qcfg4_{size}_{name}", **common, "tags": {**tags, "method": "cfg"}})
        anchor = {"scale": 4.0, "ratio": r, "base": True, "super_scale": 12.0}
        if levels:
            anchor["levels"] = levels
        rows.append({"id": f"qreg4_{size}_{name}", **common, "anchor": anchor, "tags": {**tags, "method": "reg"}})
rows.sort(key=lambda r: (r["height"], r["tags"]["method"], r["tags"]["prompt_name"]))  # so a 2-way shard gets both methods
out = ROOT / "jobs" / "qwen_pilot.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out)
