#!/usr/bin/env python
"""Aggregate scripts/sample.py sidecars into the seconds-per-image table (Phase 0 step 5).

Reads every *.json under the given directories (default $REG_DATA/outputs/smoke), groups by
model x resolution x steps x guidance, and writes results/benchmark_native.csv plus a
markdown table on stdout to paste into 06-compute-and-datasets.md.

  source env.sh
  python scripts/benchmark.py                       # smoke dir
  python scripts/benchmark.py $REG_DATA/outputs/bench_lumina $REG_DATA/outputs/bench_flux
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", default=[os.path.join(os.environ.get("REG_DATA", "."), "outputs", "smoke")])
    ap.add_argument("--out", default=str(REPO / "results" / "benchmark_native.csv"))
    args = ap.parse_args()

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in args.dirs:
        for p in sorted(Path(d).rglob("*.json")):
            try:
                r = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            if "seconds_per_image" not in r:
                continue
            key = (r["model"], r.get("quant", "bf16"), r.get("extrap", "none"), r["height"], r["width"], r["steps"], r["guidance"])
            groups[key].append(r)

    rows = []
    for (model, quant, extrap, h, w, steps, g), rs in sorted(groups.items()):
        secs = [r["seconds_per_image"] for r in rs]
        rows.append({
            "model": model,
            "quant": quant,
            "extrap": extrap,
            "height": h,
            "width": w,
            "steps": steps,
            "guidance": g,
            "n": len(rs),
            "sec_per_image": round(statistics.mean(secs), 1),
            "sec_min": round(min(secs), 1),
            "peak_vram_gb": round(max(r["peak_vram_gb"] for r in rs), 1),
            "gpu": rs[0].get("gpu", ""),
            "git": rs[-1].get("git", ""),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w") as f:
            f.write(",".join(rows[0]) + "\n")
            for r in rows:
                f.write(",".join(str(v) for v in r.values()) + "\n")

    print("| Model | Quant | Extrap | Resolution | Steps | CFG | n | s/image (mean) | min | peak VRAM GB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['quant']} | {r['extrap']} | {r['height']}x{r['width']} | {r['steps']} | {r['guidance']} | {r['n']} | "
              f"{r['sec_per_image']} | {r['sec_min']} | {r['peak_vram_gb']} |")
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
