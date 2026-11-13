"""Extend the three main 4K cells on Lumina (plain CFG w4, plain CFG w16, REG rule w4) from 50 to 100 prompts.

The kill-or-go grid and the pyramid grid ran the 4K cells on 50 of the 100 prompts of the 3K cell (25 Aesthetic-4K captions + 25 single-subject prompts); this writes the same
three configurations for the other 50 (prompt text, seed and prompt set taken from the 3K kill-or-go rows; the Aesthetic-4K reference file from the 1K rows), so that the
4K row of the main table can be reported at n = 100 like the 3K row. Output: jobs/reg_4k_ext.jsonl -> outputs/reg_4k_ext.
Ids: cfg_4096_w4_<pid>, cfg_4096_w16_<pid>, reg_pyr_s12_4096_w4_<pid>; tags carry method/cfg/size like the 5K/6K grids.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in (ROOT / "jobs/kill_or_go.jsonl").read_text().splitlines() if l.strip()]
done4k = {r["tags"]["prompt_id"] for r in rows if r["height"] == 4096}
src = {r["tags"]["prompt_id"]: r for r in rows if r["height"] == 3072 and r["guidance"] == 4.0 and r["cfg_trunc_ratio"] == 1.0}
refs = {r["tags"]["prompt_id"]: r["ref_file"] for r in rows if r.get("ref_file")}  # the 3K rows carry no reference; the 1K/2K rows do
pids = sorted(p for p in src if p not in done4k)
out = []
for p in pids:
    s = src[p]
    base = {"model": "lumina", "prompt": s["prompt"], "height": 4096, "width": 4096, "seed": s["seed"], "extrap": "ntk", **({"ref_file": refs[p]} if p in refs else {})}
    for w in (4.0, 16.0):
        out.append({"id": f"cfg_4096_w{int(w)}_{p}", **base, "guidance": w, "cfg_trunc_ratio": 1.0,
                    "tags": {"experiment": "reg_4k_ext", "size": 4096, "cfg": w, "method": "cfg", "trunc": 1.0, "prompt_id": p, "prompt_set": s["tags"]["prompt_set"]}})
    out.append({"id": f"reg_pyr_s12_4096_w4_{p}", **base, "guidance": 4.0,
                "anchor": {"scale": 4.0, "ratio": 4, "base": True, "levels": [2], "super_scale": 12.0},
                "tags": {"experiment": "reg_4k_ext", "size": 4096, "cfg": 4.0, "method": "anchor_base_pyr_s12", "prompt_id": p, "prompt_set": s["tags"]["prompt_set"]}})
(ROOT / "jobs/reg_4k_ext.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
print(len(pids), "prompts,", len(out), "rows:", pids[0], "..", pids[-1])
