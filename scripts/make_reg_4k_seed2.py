"""Second seed for the main 4K comparison on Lumina-Image 2.0 (dynamic NTK): the same 100 prompts as Table tab:main with
seed + 1000, for four cells: REG (anchor + pyramid + super-native 12), plain w = 16, hires fix (strength 0.35, w = 4) and
HiFlow with the official stage ratios. Are the paired differences of the first seed reproduced on fresh noise?
Rows are shuffled with a fixed seed so that the index-mod-4 shards mix slow and fast rows.
Output: jobs/reg_4k_seed2.jsonl (400 rows) -> outputs/reg_4k_seed2.
"""
import collections, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in (ROOT / "jobs/kill_or_go.jsonl").read_text().splitlines() if l.strip()]
src = {r["tags"]["prompt_id"]: r for r in rows if r["height"] == 3072 and r["guidance"] == 4.0 and r["cfg_trunc_ratio"] == 1.0}
refs = {r["tags"]["prompt_id"]: r["ref_file"] for r in rows if r.get("ref_file")}
subj = {r["tags"]["prompt_id"]: r["subject"] for r in rows if r.get("subject")}
assert len(src) == 100
out = []
for p in sorted(src):
    s = src[p]
    base = {"model": "lumina", "prompt": s["prompt"], "height": 4096, "width": 4096, "seed": s["seed"] + 1000, "extrap": "ntk",
            **({"ref_file": refs[p]} if p in refs else {}), **({"subject": subj[p]} if p in subj else {})}
    tags = {"experiment": "reg_4k_seed2", "size": 4096, "prompt_id": p, "prompt_set": s["tags"]["prompt_set"]}
    out.append({"id": f"reg_pyr_s12_4096_w4_s2_{p}", **base, "guidance": 4.0,
                "anchor": {"scale": 4.0, "ratio": 4, "base": True, "levels": [2], "super_scale": 12.0},
                "tags": {**tags, "cfg": 4.0, "method": "anchor_base_pyr_s12"}})
    out.append({"id": f"cfg_4096_w16_s2_{p}", **base, "guidance": 16.0, "cfg_trunc_ratio": 1.0, "tags": {**tags, "cfg": 16.0, "method": "cfg"}})
    out.append({"id": f"hires_s035_4096_w4_s2_{p}", **base, "guidance": 4.0, "cfg_trunc_ratio": 1.0,
                "hires": {"strength": 0.35, "base_size": 1024, "base_guidance": 4.0}, "tags": {**tags, "cfg": 4.0, "method": "hires_s035"}})
    out.append({"id": f"hiflow_scaled_4096_s2_{p}", **base, "guidance": 4.0, "cfg_trunc_ratio": 1.0,
                "hiflow": {"sizes": [2048, 4096], "steps": [16, 10], "alphas": [1.0, 0.25], "betas": [0.5, 0.5], "filter_ratio": [0.2, 0.2],
                           "base_size": 1024, "base_guidance": 4.0, "guidance": [5.14, 6.86]}, "tags": {**tags, "cfg": 4.0, "method": "hiflow_scaled"}})
random.Random(0).shuffle(out)
(ROOT / "jobs/reg_4k_seed2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
print(len(out), collections.Counter((r["tags"]["method"], r["tags"]["cfg"]) for r in out))
for k in range(4):
    print("shard", k, collections.Counter(r["tags"]["method"][:6] for r in out[k::4]))
