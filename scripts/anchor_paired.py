#!/usr/bin/env python
"""Paired comparison of base-anchored CFG against the kill-or-go grid's plain CFG on the same prompts and seeds.

  python scripts/anchor_paired.py [--csv results/anchor_paired.csv] [--fig results/anchor_paired.png]

Reads per-image preference records (<out>/pref/<id>.json from scripts/run_pref.py) of
$REG_DATA/outputs/reg_anchor_grid (anchored, tags.cfg = weight) and $REG_DATA/outputs/kill_or_go (plain CFG,
tags.cfg, tags.trunc). For every size and plain cell (cfg, trunc) it reports the anchored-minus-plain difference
per anchored weight over the prompts present in both, with a bootstrap 95 % CI over prompts and the win rate;
and per-cell means with bootstrap CIs. The figure plots HPS against weight: plain (trunc 1.0 solid, 0.7 dotted)
with CIs, anchored as filled squares.
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
HUE = {3072: "#1baf7a", 4096: "#eda100"}
rng = np.random.default_rng(0)


def load(name):
    """id -> (size, cfg, trunc, method, prompt_id, seed, hps, ir) for finished rows with a preference record."""
    out = {}
    for f in glob.glob(str(R / name / "*.json")):
        r = json.loads(Path(f).read_text())
        if r.get("status") != "ok":
            continue
        p = R / name / "pref" / f"{r['id']}.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        if "hps" not in s:
            continue
        t = r["tags"]
        out[r["id"]] = dict(size=int(r["height"]), cfg=float(t["cfg"]), trunc=float(t.get("trunc", r.get("cfg_trunc_ratio", 1.0))),
                            method=t.get("method", "cfg"), prompt=t["prompt_id"], seed=int(r["seed"]), hps=float(s["hps"]), ir=float(s.get("ir", np.nan)))
    return out


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/anchor_paired.csv")
    ap.add_argument("--fig", default="results/anchor_paired.png")
    ap.add_argument("--metric", default="hps", choices=["hps", "ir"])
    a = ap.parse_args()
    anch, grid = load("reg_anchor_grid"), load("kill_or_go")
    m = a.metric

    cells = defaultdict(dict)  # (size, kind, cfg, trunc) -> prompt -> score
    for r in anch.values():
        cells[(r["size"], "anchored", r["cfg"], r["trunc"])][(r["prompt"], r["seed"])] = r[m]
    for r in grid.values():
        if r["size"] in HUE:
            cells[(r["size"], "plain", r["cfg"], r["trunc"])][(r["prompt"], r["seed"])] = r[m]

    rows = []
    print(f"{'size':>5} {'cell':>22} {'n':>4} {'mean':>7} {'CI95':>16} | vs anchored w4: {'n':>3} {'diff':>6} {'CI95':>16} {'win':>5}")
    for key in sorted(cells):
        size, kind, cfg, trunc = key
        sc = cells[key]
        vals = list(sc.values())
        lo, hi = boot_ci(vals)
        row = dict(size=size, kind=kind, cfg=cfg, trunc=trunc, n=len(vals), mean=float(np.mean(vals)), lo=lo, hi=hi)
        line = f"{size:>5} {kind + ' w' + f'{cfg:g}' + ' t' + f'{trunc:g}':>22} {len(vals):>4} {np.mean(vals):>7.2f} [{lo:>6.2f}, {hi:>6.2f}]"
        ref = cells.get((size, "anchored", 4.0, 1.0), {})
        common = sorted(set(sc) & set(ref))
        if kind == "plain" and common:
            d = np.array([ref[k] - sc[k] for k in common])
            dlo, dhi = boot_ci(d)
            row.update(n_pair=len(d), diff=float(d.mean()), diff_lo=dlo, diff_hi=dhi, win=float((d > 0).mean()))
            line += f" | {len(d):>18} {d.mean():>6.2f} [{dlo:>6.2f}, {dhi:>6.2f}] {100 * (d > 0).mean():>4.0f}%"
        print(line)
        rows.append(row)

    with open(a.csv, "w", newline="") as f:
        keys = ["size", "kind", "cfg", "trunc", "n", "mean", "lo", "hi", "n_pair", "diff", "diff_lo", "diff_hi", "win"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.csv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    label = {"hps": "HPSv2.1 (×100)", "ir": "ImageReward"}[m]
    for size in sorted(HUE):
        for trunc, ls in ((1.0, "--"), (0.7, ":")):
            pts = sorted((r["cfg"], r) for r in rows if r["size"] == size and r["kind"] == "plain" and r["trunc"] == trunc)
            if not pts:
                continue
            w = [p[0] for p in pts]
            y = [p[1]["mean"] for p in pts]
            err = [[p[1]["mean"] - p[1]["lo"] for p in pts], [p[1]["hi"] - p[1]["mean"] for p in pts]]
            ax.errorbar(w, y, yerr=err, color=HUE[size], ls=ls, lw=1.6, marker="o", ms=5, mfc="#fcfcfb", capsize=2,
                        label=f"{size // 1024}K plain CFG" + (" (end 0.7)" if trunc != 1.0 else ""))
        pts = sorted((r["cfg"], r) for r in rows if r["size"] == size and r["kind"] == "anchored")
        if pts:
            w = [p[0] for p in pts]
            y = [p[1]["mean"] for p in pts]
            err = [[p[1]["mean"] - p[1]["lo"] for p in pts], [p[1]["hi"] - p[1]["mean"] for p in pts]]
            ax.errorbar(w, y, yerr=err, color=HUE[size], ls="-", lw=2.2, marker="s", ms=7, capsize=2, label=f"{size // 1024}K anchored (n={pts[0][1]['n']})")
            ax.annotate(f"{size // 1024}K anchored", (w[-1], y[-1]), xytext=(7, 6 if size == 3072 else -9), textcoords="offset points", fontsize=8, fontweight="bold", color="#0b0b0b", va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 3, 4, 5, 6, 8, 10, 12, 16])
    ax.set_xticklabels(["2", "3", "4", "5", "6", "8", "10", "12", "16"])
    ax.set_xlabel("CFG weight w")
    ax.set_ylabel(label)
    ax.set_title("Kill-or-go prompts, same seeds: anchored vs plain CFG (bootstrap 95 % CI)", fontsize=9.5, loc="left")
    ax.grid(True, color="#e6e5e1", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(a.fig, dpi=150)
    print("wrote", a.fig)


if __name__ == "__main__":
    main()
