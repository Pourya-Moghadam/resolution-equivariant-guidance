#!/usr/bin/env python
"""Weight-curve figure: HPSv2 against CFG weight for plain CFG, the single-level anchor and the pyramid anchor, at 3K
(n = 100 prompts) and 4K (n = 50), all on the same kill-or-go prompts and seeds.

  python scripts/reg_weight_curve.py [--metric hps|ir] [--fig results/reg_weight_curve.png] [--csv results/reg_weight_curve.csv]

Reads per-image preference records (<out>/pref/<id>.json) of kill_or_go (plain, trunc 1.0), reg_anchor_grid
(tags.method anchor_base) and reg_pyramid_grid (anchor_base_pyr; anchor_base_pyr_s12 as a dashed optional row).
Points are cell means with bootstrap 95 % CIs over prompts. The 2K plain curve is drawn as the native reference.
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
SERIES = [  # key, label, colour (fixed order; one hue per method), linestyle, marker
    ("cfg", "plain CFG", "#5b6b7a", "-", "o"),
    ("anchor_base", "single anchor (native band)", "#1b7fd1", "-", "s"),
    ("anchor_base_pyr", "pyramid anchor (REG)", "#d1431b", "-", "D"),
    ("anchor_base_pyr_s12", "pyramid + super-native 12", "#d1431b", ":", "^"),
]


def load(metric):
    cells = defaultdict(dict)  # (size, method, cfg) -> prompt -> value
    for name in ("kill_or_go", "reg_anchor_grid", "reg_pyramid_grid"):
        for f in glob.glob(str(R / name / "*.json")):
            r = json.loads(Path(f).read_text())
            if r.get("status") != "ok":
                continue
            p = R / name / "pref" / f"{r['id']}.json"
            if not p.exists():
                continue
            s = json.loads(p.read_text())
            t = r["tags"]
            if float(t.get("trunc", r.get("cfg_trunc_ratio", 1.0))) != 1.0 or s.get(metric) is None:
                continue
            cells[(int(r["height"]), t.get("method", "cfg"), float(t["cfg"]))][(t["prompt_id"], int(r["seed"]))] = float(s[metric])
    return cells


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="hps", choices=["hps", "ir"])
    ap.add_argument("--fig", default="results/reg_weight_curve.png")
    ap.add_argument("--csv", default="results/reg_weight_curve.csv")
    a = ap.parse_args()
    cells = load(a.metric)
    rows = []
    for (size, method, cfg), sc in sorted(cells.items()):
        if size == 1024:
            continue
        v = list(sc.values())
        lo, hi = boot_ci(v)
        rows.append(dict(size=size, method=method, cfg=cfg, n=len(v), mean=float(np.mean(v)), lo=lo, hi=hi))
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = {"hps": "HPSv2.1 ×100", "ir": "ImageReward"}[a.metric]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True, facecolor="white")
    for ax, size in zip(axes, (3072, 4096)):
        ref = [r for r in rows if r["size"] == 2048 and r["method"] == "cfg"]
        if ref:
            ref.sort(key=lambda r: r["cfg"])
            ax.plot([r["cfg"] for r in ref], [r["mean"] for r in ref], color="#b5bdc4", lw=1.6, ls="--", label="plain CFG at 2K (native reference)")
        for key, name, col, ls, mk in SERIES:
            pts = sorted((r for r in rows if r["size"] == size and r["method"] == key), key=lambda r: r["cfg"])
            if not pts:
                continue
            x = [r["cfg"] for r in pts]
            y = [r["mean"] for r in pts]
            ax.plot(x, y, color=col, lw=2, ls=ls, marker=mk, ms=6, mec="white", mew=1, label=name, zorder=3)
            ax.fill_between(x, [r["lo"] for r in pts], [r["hi"] for r in pts], color=col, alpha=0.12, lw=0)
        n = next((r["n"] for r in rows if r["size"] == size and r["method"] == "anchor_base"), 0)
        ax.set_title(f"{size // 1024}K ({size}², n = {n} prompts)", fontsize=10.5, loc="left")
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16])
        ax.set_xticklabels(["2", "4", "8", "16"])
        ax.set_xlabel("CFG weight")
        ax.grid(True, color="#e8eaec", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(label)
    axes[1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(a.fig, dpi=160)
    print("wrote", a.fig, "and", a.csv)
    for r in rows:
        if r["size"] >= 3072 and (r["method"] != "cfg" or r["cfg"] in (2.0, 4.0, 8.0, 16.0)):
            print(f"{r['size']} {r['method']:>20} w{r['cfg']:<4g} n={r['n']:<3} {r['mean']:6.2f} [{r['lo']:.2f}, {r['hi']:.2f}]")


if __name__ == "__main__":
    main()
