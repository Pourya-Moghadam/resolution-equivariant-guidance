"""Extend the 5K and 6K main cells on Lumina (plain CFG at the re-tuned weight w = 16, REG rule at w = 4) from 50 to 100
prompts, as scripts/make_reg_4k_ext.py did at 4K: the 50 prompts of the 3K cell that the 5K/6K grids did not run,
with their seeds, prompt sets and Aesthetic-4K reference files. Rows alternate 5K/6K so every shard gets a mix.
Output: jobs/reg_5k6k_ext.jsonl (200 rows) -> outputs/reg_5k6k_ext. Ids: cfg_<size>_w16_<pid>, reg_pyr_s12_<size>_w4_<pid>.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in (ROOT / "jobs/kill_or_go.jsonl").read_text().splitlines() if l.strip()]
done = {json.loads(l)["tags"]["prompt_id"] for l in (ROOT / "jobs/reg_5k_grid.jsonl").read_text().splitlines() if l.strip()}
src = {r["tags"]["prompt_id"]: r for r in rows if r["height"] == 3072 and r["guidance"] == 4.0 and r["cfg_trunc_ratio"] == 1.0}
refs = {r["tags"]["prompt_id"]: r["ref_file"] for r in rows if r.get("ref_file")}
pids = sorted(p for p in src if p not in done)
out = []
for p in pids:
    s = src[p]
    for size in (5120, 6144):
        base = {"model": "lumina", "prompt": s["prompt"], "height": size, "width": size, "seed": s["seed"], "extrap": "ntk", **({"ref_file": refs[p]} if p in refs else {})}
        tags = {"experiment": "reg_5k6k_ext", "size": size, "prompt_id": p, "prompt_set": s["tags"]["prompt_set"]}
        out.append({"id": f"cfg_{size}_w16_{p}", **base, "guidance": 16.0, "cfg_trunc_ratio": 1.0, "tags": {**tags, "cfg": 16.0, "method": "cfg", "trunc": 1.0}})
        out.append({"id": f"reg_pyr_s12_{size}_w4_{p}", **base, "guidance": 4.0,
                    "anchor": {"scale": 4.0, "ratio": size // 1024, "base": True, "levels": [2], "super_scale": 12.0},
                    "tags": {**tags, "cfg": 4.0, "method": "anchor_base_pyr_s12"}})
(ROOT / "jobs/reg_5k6k_ext.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
print(len(pids), "prompts,", len(out), "rows:", pids[0], "..", pids[-1])
