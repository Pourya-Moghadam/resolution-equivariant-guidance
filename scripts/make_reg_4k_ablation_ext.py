"""Extend the three 4K ablation cells on Lumina (single anchor, + pyramid, + super-native 3w; all w4) from 50 to 100
prompts: the same other-50 prompts as scripts/make_reg_4k_ext.py, so that the ablation table can be reported at n = 100
like the main table. Output: jobs/reg_4k_ablation_ext.jsonl -> outputs/reg_4k_ablation_ext (150 rows).
Ids: anchor_4096_w4_<pid>, reg_pyr_4096_w4_<pid>, reg_s12_4096_w4_<pid>; methods anchor_base / anchor_base_pyr / anchor_base_s12.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in (ROOT / "jobs/kill_or_go.jsonl").read_text().splitlines() if l.strip()]
done4k = {r["tags"]["prompt_id"] for r in rows if r["height"] == 4096}
src = {r["tags"]["prompt_id"]: r for r in rows if r["height"] == 3072 and r["guidance"] == 4.0 and r["cfg_trunc_ratio"] == 1.0}
refs = {r["tags"]["prompt_id"]: r["ref_file"] for r in rows if r.get("ref_file")}
subj = {r["tags"]["prompt_id"]: r["subject"] for r in rows if r.get("subject")}
CELLS = [("anchor", "anchor_base", {"scale": 4.0, "ratio": 4, "base": True}),
         ("reg_pyr", "anchor_base_pyr", {"scale": 4.0, "ratio": 4, "base": True, "levels": [2]}),
         ("reg_s12", "anchor_base_s12", {"scale": 4.0, "ratio": 4, "base": True, "super_scale": 12.0})]
pids = sorted(p for p in src if p not in done4k)
out = []
for p in pids:
    s = src[p]
    base = {"model": "lumina", "prompt": s["prompt"], "height": 4096, "width": 4096, "seed": s["seed"], "extrap": "ntk",
            **({"ref_file": refs[p]} if p in refs else {}), **({"subject": subj[p]} if p in subj else {})}
    for stem, method, anchor in CELLS:
        out.append({"id": f"{stem}_4096_w4_{p}", **base, "guidance": 4.0, "anchor": anchor,
                    "tags": {"experiment": "reg_4k_ablation_ext", "size": 4096, "cfg": 4.0, "method": method, "prompt_id": p, "prompt_set": s["tags"]["prompt_set"]}})
(ROOT / "jobs/reg_4k_ablation_ext.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
print(len(pids), "prompts,", len(out), "rows:", pids[0], "..", pids[-1], "| subjects:", sum("subject" in r for r in out))
