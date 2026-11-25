"""Token Perturbation Guidance at 4K on Lumina-Image 2.0 (dynamic NTK), 100 prompts, with the setting calibrated at 1K
(scale 1.5, half of the tokens shuffled, layers 15-17; jobs/pert_calib_1k.jsonl), as in the 6-prompt generality pilot:
  cfg_tpg     plain CFG at the native weight 4 + TPG: does a perturbation guidance rescue the native setting at 4K?
  cfg16_tpg   plain CFG at the re-tuned weight 16 + TPG
  reg_tpg     REG (w = 4) + TPG: does the 1K gain of TPG carry over once the base prediction is anchored?
Rows are shuffled with a fixed seed. Output: jobs/reg_4k_tpg.jsonl (300 rows) -> outputs/reg_4k_tpg.
"""
import collections, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in (ROOT / "jobs/kill_or_go.jsonl").read_text().splitlines() if l.strip()]
src = {r["tags"]["prompt_id"]: r for r in rows if r["height"] == 3072 and r["guidance"] == 4.0 and r["cfg_trunc_ratio"] == 1.0}
refs = {r["tags"]["prompt_id"]: r["ref_file"] for r in rows if r.get("ref_file")}
subj = {r["tags"]["prompt_id"]: r["subject"] for r in rows if r.get("subject")}
assert len(src) == 100
TPG = {"scale": 1.5, "frac": 0.5, "layers": [15, 16, 17]}
out = []
for p in sorted(src):
    s = src[p]
    base = {"model": "lumina", "prompt": s["prompt"], "height": 4096, "width": 4096, "seed": s["seed"], "extrap": "ntk", "tpg": TPG,
            **({"ref_file": refs[p]} if p in refs else {}), **({"subject": subj[p]} if p in subj else {})}
    tags = {"experiment": "reg_4k_tpg", "size": 4096, "prompt_id": p, "prompt_set": s["tags"]["prompt_set"]}
    out.append({"id": f"cfgtpg_4096_w4_{p}", **base, "guidance": 4.0, "tags": {**tags, "cfg": 4.0, "method": "cfg_tpg"}})
    out.append({"id": f"cfgtpg_4096_w16_{p}", **base, "guidance": 16.0, "tags": {**tags, "cfg": 16.0, "method": "cfg_tpg"}})
    out.append({"id": f"regtpg_4096_w4_{p}", **base, "guidance": 4.0,
                "anchor": {"scale": 4.0, "ratio": 4, "base": True, "levels": [2], "super_scale": 12.0}, "tags": {**tags, "cfg": 4.0, "method": "reg_tpg"}})
random.Random(0).shuffle(out)
(ROOT / "jobs/reg_4k_tpg.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
print(len(out), collections.Counter((r["tags"]["method"], r["tags"]["cfg"]) for r in out))
