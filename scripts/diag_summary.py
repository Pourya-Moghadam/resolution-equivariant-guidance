#!/usr/bin/env python
"""Summarise reg.diag recordings of a job: the relative size of the guidance signal per band and per
resolution, and the (B′) prediction for the coarse-band weight (05, revision 2026-10-28).

  source env.sh
  python scripts/diag_summary.py --out $REG_DATA/outputs/reg_diag [--csv results/reg_diag.csv] [--fig results/reg_diag.png]

For every (size, cfg) cell: ρ_band = ‖Δ_low‖/‖v_cond,low‖ per step, averaged over prompts, for each
recorded cutoff; also the global ratio ‖Δ‖/‖v_cond‖. ρ falls by two orders of magnitude along the
trajectory, so cells are compared step by step (the sigma schedule is the same at every resolution)
and summarised by the median over steps 1..n-1 (step 0, pure noise, is excluded). The prediction
    w_low(r) = w_native · median_t[ ρ_native,t(cutoff 2) / ρ_r,t(cutoff 2r) ]
compares the native cutoff at 1K with the same cutoff in cycles per image at ratio r.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

NATIVE = 1024


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--fig")
    ap.add_argument("--w-native", type=float, default=4.0)
    a = ap.parse_args()
    out = Path(a.out)

    cells = collections.defaultdict(list)  # (size, cfg) -> list of (name, npz)
    for js in sorted(out.glob("*.json")):
        r = json.loads(js.read_text())
        if r.get("status") != "ok" or "diag" not in r:
            continue
        d = np.load(r["diag"])
        cells[(int(r["height"]), float(r["tags"]["cfg"]))].append((r["tags"].get("prompt_name", r["id"]), d))
    if not cells:
        raise SystemExit("no finished rows with diagnostics")

    rows = []
    per_step = {}  # (size, cfg) -> (sigma, ratio[step, down]) averaged over prompts
    for (size, cfg), items in sorted(cells.items()):
        downs = items[0][1]["downs"]
        ratios = np.stack([d["low_rms_delta"] / d["low_rms_cond"] for _, d in items])  # prompt, step, down
        glob = np.stack([d["rms_delta"] / d["rms_cond"] for _, d in items])
        cos = np.stack([d["cos_low"] for _, d in items])
        sigma = items[0][1]["sigma"]
        per_step[(size, cfg)] = (sigma, np.nanmean(ratios, axis=0), downs)
        sl = slice(1, None)  # drop step 0
        row = {"size": size, "cfg": cfg, "n": len(items), "ratio_global": float(np.nanmedian(np.nanmean(glob, 0)[sl]))}
        for j, dn in enumerate(downs):
            row[f"ratio_d{dn:g}"] = float(np.nanmedian(np.nanmean(ratios[:, :, j], 0)[sl]))
            row[f"cos_d{dn:g}"] = float(np.nanmean(cos[:, :, j]))
        rows.append(row)

    # (B') prediction: native cutoff 2 at 1K vs cutoff 2r at size r, same cfg
    nat = {cfg: v for (size, cfg), v in per_step.items() if size == NATIVE}
    for r in rows:
        ratio = r["size"] / NATIVE
        cell = per_step[(r["size"], r["cfg"])]
        downs = list(cell[2])
        r["w_low_pred"] = float("nan")
        if r["cfg"] in nat and 2.0 * ratio in downs:
            rho_nat = nat[r["cfg"]][1][1:, downs.index(2.0)]
            rho_r = cell[1][1:, downs.index(2.0 * ratio)]
            r["w_low_pred"] = float(a.w_native * np.nanmedian(rho_nat / rho_r))

    keys = ["size", "cfg", "n", "ratio_global"] + [k for k in rows[0] if k.startswith("ratio_d")] + ["w_low_pred"]
    print("  ".join(f"{k:>12}" for k in keys))
    for r in rows:
        print("  ".join(f"{r[k]:>12.4g}" if isinstance(r[k], float) else f"{r[k]:>12}" for k in keys))

    if a.csv:
        import csv

        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", a.csv)

    if a.fig:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for (size, cfg), (sigma, ratio, downs) in sorted(per_step.items()):
            r_ = size / NATIVE
            j_native = list(downs).index(2.0)
            j_rel = list(downs).index(2.0 * r_) if 2.0 * r_ in list(downs) else j_native
            axes[0].plot(sigma, ratio[:, j_native], label=f"{size} cfg{cfg:g}")
            axes[1].plot(sigma, ratio[:, j_rel], label=f"{size} cfg{cfg:g} (cutoff {2 * r_:g})")
        for ax, title in zip(axes, ("low band, grid cutoff 2", "low band, cutoff 2·r (same cycles/image)")):
            ax.set_xlabel("sigma (1 = noise)")
            ax.set_ylabel("‖Δ_low‖ / ‖v_cond,low‖")
            ax.set_title(title)
            ax.invert_xaxis()
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(a.fig, dpi=130)
        print("wrote", a.fig)


if __name__ == "__main__":
    main()
