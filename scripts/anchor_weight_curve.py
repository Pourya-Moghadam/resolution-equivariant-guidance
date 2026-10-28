#!/usr/bin/env python
"""Human-preference score versus CFG weight, plain CFG against base-anchored CFG, per resolution (08, stage C).

  python scripts/anchor_weight_curve.py [--fig results/anchor_weight_curve.png]

Reads results/pilot_cfg_baseline_pref.csv (plain 1K-3K), results/reg_pilot_cfg4k_baseline_pref.csv (plain 4K)
and results/reg_anchor_pilot_pref.csv (anchored, method anchor_base). Six prompts per point; bars are ±1 s.e.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HUE = {1024: "#2a78d6", 2048: "#eb6834", 3072: "#1baf7a", 4096: "#eda100"}  # fixed order, validated (dataviz)


def load(path, **fixed):
    out = {}
    for r in csv.DictReader(open(path)):
        size = int(r.get("size", fixed.get("size", 0)))
        if "method" in r and r["method"] != "anchor_base":
            continue
        out.setdefault(size, []).append((float(r["cfg"]), float(r["hps"]), float(r["hps_se"]), float(r["ir"]), float(r["ir_se"])))
    return {k: sorted(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig", default="results/anchor_weight_curve.png")
    a = ap.parse_args()
    plain = load("results/pilot_cfg_baseline_pref.csv")
    plain.update(load("results/reg_pilot_cfg4k_baseline_pref.csv", size=4096))
    anch = load("results/reg_anchor_pilot_pref.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor="#fcfcfb")
    for ax, (j, name, lo, hi) in zip(axes, ((1, "HPSv2.1 (×100)", 10, 34.5), (3, "ImageReward", -1.5, 1.8))):
        ax.set_facecolor("#fcfcfb")
        for size, rows in sorted(plain.items()):
            w = [r[0] for r in rows]
            ax.errorbar(w, [r[j] for r in rows], yerr=[r[j + 1] for r in rows], color=HUE[size], ls="--", lw=1.6, marker="o", ms=5,
                        mfc="#fcfcfb", capsize=2, label=f"{size // 1024}K CFG")
        for size, rows in sorted(anch.items()):
            w = [r[0] for r in rows]
            ax.errorbar(w, [r[j] for r in rows], yerr=[r[j + 1] for r in rows], color=HUE[size], ls="-", lw=2.2, marker="s", ms=6,
                        capsize=2, label=f"{size // 1024}K anchored")
            if j == 1:  # direct labels on the anchored lines only (the legend names the rest)
                ax.annotate(f"{size // 1024}K anchored", (w[-1], rows[-1][j]), xytext=(6, 6 if size == 3072 else -8), textcoords="offset points", fontsize=8, color="#0b0b0b", va="center", fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 6, 8, 10])
        ax.set_xticklabels(["2", "4", "6", "8", "10"])
        ax.set_xlim(1.7, 15)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("CFG weight w")
        ax.set_ylabel(name)
        ax.grid(True, color="#e6e5e1", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors="#52514e")
    axes[0].set_title("Base-anchored CFG (solid) returns the weight curve to the native one", fontsize=10, loc="left")
    axes[0].legend(fontsize=7, ncol=2, frameon=False, loc="lower right")
    fig.tight_layout()
    Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.fig, dpi=150)
    print("wrote", a.fig)


if __name__ == "__main__":
    main()
