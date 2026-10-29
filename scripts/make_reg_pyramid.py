#!/usr/bin/env python
"""Write the pyramid-anchor pilot job files (08, 2026-10-29), next to the stage B/C base-anchored rows
(`reg_anchor_pilot`, w4 single level). Both stages write to the same output dir, reg_pyramid_pilot.

Stage 1 (jobs/reg_pyramid_pilot.jsonl, 4K, w4, base=True):
    pyr   levels [2]                (1K + 2K passes)
    sup   super_scale 1             (no guidance above the native Nyquist)
    pyrs  levels [2] + super_scale 1
Stage 2 (jobs/reg_pyramid_pilot2.jsonl), after stage 1 showed pyr > single level and sup < single level
(guidance above the native Nyquist suppresses speckle, so the rule's sign was wrong):
    pyr_s8, pyr_s12   4K pyramid with super_scale 8 / 12 (w4 below the native Nyquist, more above)
    pyr (3K)          the pyramid at 3K (ratio 3, level 2 -> 1536 px pass), vs anchorbase4_3072

  python scripts/make_reg_pyramid.py [--stage 2]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
args = ap.parse_args()
src = [json.loads(l) for l in open(ROOT / "jobs" / "reg_anchor_pilot_b.jsonl")]


def base_rows(size):
    rows = [r for r in src if r["height"] == size and r["tags"]["cfg"] == 4.0 and r["anchor"].get("base")]
    assert len(rows) == 6, len(rows)
    return rows


def make(name, size, extra):
    out = []
    for r in base_rows(size):
        row = json.loads(json.dumps(r))
        row["anchor"] = {**row["anchor"], **extra}
        row["id"] = f"{name}4_{size}_{row['tags']['prompt_name']}"
        row["tags"] = {**row["tags"], "experiment": "reg_pyramid_pilot", "method": f"anchor_base_{name}"}
        out.append(row)
    return out


if args.stage == 1:
    variants = {"pyr": {"levels": [2]}, "sup": {"super_scale": 1.0}, "pyrs": {"levels": [2], "super_scale": 1.0}}
    rows = [row for name, extra in variants.items() for row in make(name, 4096, extra)]
    fname = "reg_pyramid_pilot.jsonl"
else:
    rows = make("pyr_s8", 4096, {"levels": [2], "super_scale": 8.0}) + make("pyr_s12", 4096, {"levels": [2], "super_scale": 12.0}) \
        + make("pyr", 3072, {"levels": [2]})
    fname = "reg_pyramid_pilot2.jsonl"
rows.sort(key=lambda r: (r["tags"]["prompt_name"], -r["height"], r["id"]))  # interleave variants so a partial run compares
out = ROOT / "jobs" / fname
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", out, f"~{sum(790 if r['height'] == 4096 else 330 for r in rows) / 3600:.1f} GPU-h")
