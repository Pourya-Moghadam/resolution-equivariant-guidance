#!/usr/bin/env python
"""Write jobs/reg_anchor_grid.jsonl: base-anchored CFG at the native weight on the kill-or-go prompts, so
the headline comparison (anchored w4 vs plain CFG at every weight) has the grid's own baselines — same
prompts, seeds, sampler — at 3K (100 prompts) and 4K (50 prompts). ~18 GPU-h on one card per weight.

  python scripts/make_reg_anchor_grid.py                          # w4 -> jobs/reg_anchor_grid.jsonl
  python scripts/make_reg_anchor_grid.py --weights 2 8 --name reg_anchor_grid_w28   # the curve's ends (stage C follow-up)
  python scripts/make_reg_anchor_grid.py --levels 2 --super-scale 12 --sizes 4096 --name reg_pyramid_grid   # REG pyramid + super-native weight (08, 2026-10-29)

Every job writes to the same output dir (tags.experiment = reg_anchor_grid) so one run_eval/run_pref call scores the curve.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PER_HALF = {3072: 50, 4096: 25}  # as in make_kill_or_go.py
ap = argparse.ArgumentParser()
ap.add_argument("--weights", type=float, nargs="+", default=[4.0])
ap.add_argument("--name", default="reg_anchor_grid")
ap.add_argument("--levels", type=int, nargs="*", default=[], help="pyramid levels (intermediate anchor ratios), e.g. 2")
ap.add_argument("--super-scale", type=float, default=None, help="CFG weight above the native Nyquist")
ap.add_argument("--sizes", type=int, nargs="*", default=[3072, 4096])
args = ap.parse_args()
PER_HALF = {s: PER_HALF[s] for s in args.sizes}
extra = {}
if args.levels:
    extra["levels"] = args.levels
if args.super_scale is not None:
    extra["super_scale"] = args.super_scale
method = "anchor_base" + ("_pyr" if args.levels else "") + (f"_s{args.super_scale:g}" if args.super_scale is not None else "")
prefix = "anchor" if not extra else method.replace("anchor_base", "reg")

pool = [json.loads(l) for l in open(ROOT / "prompts" / "kill_or_go_200.jsonl")]
a4k = [p for p in pool if "ref_file" in p]
single = [p for p in pool if "subject" in p]
seed_of = {p["id"]: i for i, p in enumerate(pool)}

rows = []
for W in args.weights:
  for size, n in PER_HALF.items():
    for p in a4k[:n] + single[:n]:
        row = {
            "id": f"{prefix}_{size}_w{W:g}_{p['id']}", "model": "lumina", "prompt": p["prompt"], "height": size, "width": size,
            "seed": seed_of[p["id"]], "guidance": W, "anchor": {"scale": W, "ratio": size // 1024, "base": True, **extra},
            "extrap": "ntk",
            "tags": {"experiment": args.name if extra else "reg_anchor_grid", "size": size, "cfg": W, "method": method,
                     "prompt_id": p["id"], "prompt_set": "a4k" if "ref_file" in p else "single"},
        }
        for k in ("subject", "expected_count", "ref_file"):
            if k in p:
                row[k] = p[k]
        rows.append(row)
rows.sort(key=lambda r: (r["tags"]["prompt_id"], r["height"], r["tags"]["cfg"]))  # interleave sizes/weights so partial runs cover all
out = ROOT / "jobs" / f"{args.name}.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
h = sum({3072: 340, 4096: 790}[r["height"]] if args.levels else {3072: 290, 4096: 700}[r["height"]] for r in rows) / 3600
print(len(rows), "rows ->", out, f"~{h:.0f} GPU-h")
