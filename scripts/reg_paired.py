#!/usr/bin/env python
"""Paired comparison of REG (and its component ablations) against the single-level anchor and plain CFG on the
same 4K kill-or-go prompts and seeds (08, 2026-10-30).

  python scripts/reg_paired.py [--metric hps|ir|clip] [--size 3072] [--ref anchor_base --ref-cfg 8] [--csv results/reg_paired.csv]

Reads the per-image preference records (<out>/pref/<id>.json) or eval records (<out>/eval/<id>.json, --metric clip) of
$REG_DATA/outputs/reg_pyramid_grid (tags.method = anchor_base_pyr_s12 | anchor_base_pyr | anchor_base_s12, w4),
reg_anchor_grid (anchor_base at tags.cfg) and kill_or_go (plain CFG at tags.cfg, trunc 1.0). For every method/weight
cell it prints the mean with a bootstrap 95 % CI over prompts, and for every cell the paired difference REG minus
cell over the prompts present in both, with CI and win rate.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(os.environ.get("REG_DATA", "/data1/moghadamp/reg_data")) / "outputs"
rng = np.random.default_rng(0)


def load(name, metric, size, any_trunc=False):
    sub, key = ("eval", "clip") if metric == "clip" else ("pref", metric)
    out = {}
    for f in glob.glob(str(R / name / "*.json")):
        r = json.loads(Path(f).read_text())
        if r.get("status") != "ok" or int(r["height"]) != size:
            continue
        p = R / name / sub / f"{r['id']}.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        if key not in s or s[key] is None:
            continue
        t = r["tags"]
        if not any_trunc and float(t.get("trunc", r.get("cfg_trunc_ratio", 1.0))) != 1.0:
            continue
        out[r["id"]] = dict(method=t.get("method", "cfg"), cfg=float(t["cfg"]), prompt=(t["prompt_id"], int(r["seed"])), v=float(s[key]))
    return out


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="hps", choices=["hps", "ir", "clip", "hps_crop", "ir_crop"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--ref", default="anchor_base_pyr_s12", help="method that the paired differences are taken against")
    ap.add_argument("--ref-cfg", type=float, default=4.0)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--any-trunc", action="store_true", help="keep cells whose guidance interval is not 1.0 (e.g. the 1K-then-upscale rows, trunc 0.9)")
    ap.add_argument("--dirs", nargs="*", default=["kill_or_go", "reg_anchor_grid", "reg_pyramid_grid"],
                    help="output dirs under $REG_DATA/outputs to read (add e.g. reg_generality_grid)")
    a = ap.parse_args()
    cells = defaultdict(dict)  # (method, cfg) -> prompt -> value
    for name in a.dirs:
        for r in load(name, a.metric, a.size, a.any_trunc).values():
            cells[(r["method"], r["cfg"])][r["prompt"]] = r["v"]
    ref = cells.get((a.ref, a.ref_cfg), {})
    order = sorted(cells, key=lambda k: (k[0] != a.ref, k[0].startswith("anchor"), k[0], k[1]))
    rows = []
    print(f"{a.metric}: {a.size // 1024}K kill-or-go prompts, trunc 1.0; paired against {a.ref} w{a.ref_cfg:g} (n={len(ref)})")
    print(f"{'method':>22} {'w':>4} {'n':>3} {'mean':>7} {'CI95':>16} | {'n':>3} {'ref-cell':>8} {'CI95':>16} {'win':>5}")
    for key in order:
        method, cfg = key
        sc = cells[key]
        vals = list(sc.values())
        lo, hi = boot_ci(vals)
        row = dict(method=method, cfg=cfg, n=len(vals), mean=float(np.mean(vals)), lo=lo, hi=hi)
        line = f"{method:>22} {cfg:>4g} {len(vals):>3} {np.mean(vals):>7.3f} [{lo:>6.3f}, {hi:>6.3f}]"
        common = sorted(set(sc) & set(ref))
        if key != (a.ref, a.ref_cfg) and common:
            d = np.array([ref[k] - sc[k] for k in common])
            dlo, dhi = boot_ci(d)
            row.update(n_pair=len(d), diff=float(d.mean()), diff_lo=dlo, diff_hi=dhi, win=float((d > 0).mean()))
            line += f" | {len(d):>3} {d.mean():>8.3f} [{dlo:>6.3f}, {dhi:>6.3f}] {100 * (d > 0).mean():>4.0f}%"
        print(line)
        rows.append(row)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["method", "cfg", "n", "mean", "lo", "hi", "n_pair", "diff", "diff_lo", "diff_hi", "win"])
            w.writeheader()
            w.writerows(rows)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
