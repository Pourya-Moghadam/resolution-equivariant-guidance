"""GenEval at the native 1K resolution on Lumina-Image 2.0, plain CFG at the native weight (w = 4): the reference the
4K GenEval rows (jobs/geneval_4k.jsonl) are compared against. Same prompts, seeds (= prompt index) and tags.
Output: jobs/geneval_1k.jsonl (553 rows) -> outputs/geneval_1k; scored by scripts/run_geneval.py like the 4K set.
"""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = Path(os.environ.get("REG_DATA", "/data1/moghadamp/reg_data")) / "datasets/geneval/evaluation_metadata.jsonl"
prompts = [json.loads(l) for l in META.read_text().splitlines() if l.strip()]
out = []
for i, m in enumerate(prompts):
    pid = f"gev{i:03d}"
    tags = {"experiment": "geneval_1k", "size": 1024, "category": m["tag"], "prompt_id": pid, "prompt_set": "geneval", "geneval_index": i,
            "cfg": 4.0, "method": "cfg", "trunc": 1.0}
    out.append({"id": f"gev_cfg_1024_w4_{pid}", "model": "lumina", "prompt": m["prompt"], "height": 1024, "width": 1024, "seed": i,
                "extrap": "none", "guidance": 4.0, "cfg_trunc_ratio": 1.0, "tags": tags})
(ROOT / "jobs/geneval_1k.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
print(len(out), "rows")
