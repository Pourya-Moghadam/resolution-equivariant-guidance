"""GenEval (553 prompts, 6 task types: single/two objects, counting, colors, position, colour attribution) at 4K on
Lumina-Image 2.0: plain CFG at the best swept weight (w = 16) and the REG rule at the native weight (w = 4), one image per
prompt, seed = prompt index. Rows carry tags.category = the GenEval tag and tags.prompt_id = gev<idx:03d>, so the official
evaluator can be run through scripts/run_geneval.py (which lays the images out as GenEval expects, by prompt index).
Prompt file: $REG_DATA/datasets/geneval/evaluation_metadata.jsonl (from github.com/djghosh13/geneval).
Output: jobs/geneval_4k.jsonl (1106 rows) -> outputs/geneval_4k.
"""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = Path(os.environ.get("REG_DATA", "/data1/moghadamp/reg_data")) / "datasets/geneval/evaluation_metadata.jsonl"
prompts = [json.loads(l) for l in META.read_text().splitlines() if l.strip()]
out = []
for i, m in enumerate(prompts):
    pid = f"gev{i:03d}"
    base = {"model": "lumina", "prompt": m["prompt"], "height": 4096, "width": 4096, "seed": i, "extrap": "ntk"}
    tags = {"experiment": "geneval_4k", "size": 4096, "category": m["tag"], "prompt_id": pid, "prompt_set": "geneval", "geneval_index": i}
    out.append({"id": f"gev_cfg_4096_w16_{pid}", **base, "guidance": 16.0, "cfg_trunc_ratio": 1.0, "tags": {**tags, "cfg": 16.0, "method": "cfg", "trunc": 1.0}})
    out.append({"id": f"gev_reg_4096_w4_{pid}", **base, "guidance": 4.0,
                "anchor": {"scale": 4.0, "ratio": 4, "base": True, "levels": [2], "super_scale": 12.0},
                "tags": {**tags, "cfg": 4.0, "method": "anchor_base_pyr_s12"}})
(ROOT / "jobs/geneval_4k.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
print(len(prompts), "prompts,", len(out), "rows")
