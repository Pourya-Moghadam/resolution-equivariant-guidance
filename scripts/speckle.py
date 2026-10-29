#!/usr/bin/env python
"""Fine-scale energy of job outputs at full resolution, the measurement the preference scorers cannot make
(they downscale to 1K first): per image, the std of the Laplacian over a centre crop (1/4 of the side) of
the luminance, and the fraction of pixel-spectrum power above the native Nyquist (cycles/image > 512),
grouped by row/tag keys. The 4K speckle shows up as a high Laplacian std with ordinary contrast.

  source env.sh
  python scripts/speckle.py --out $REG_DATA/outputs/reg_pyramid_pilot [--out $REG_DATA/outputs/reg_anchor_pilot ...] \\
      --group-by size method cfg [--filter size=4096] [--csv results/speckle.csv]
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
NATIVE = 1024


def val(r, k):
    return r.get("tags", {}).get(k, r.get(k))


def measure(path: str) -> dict:
    im = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    h, w = im.shape
    c = h // 4
    crop = im[h // 2 - c // 2: h // 2 + c // 2, w // 2 - c // 2: w // 2 + c // 2]
    lap = -4 * crop[1:-1, 1:-1] + crop[:-2, 1:-1] + crop[2:, 1:-1] + crop[1:-1, :-2] + crop[1:-1, 2:]
    # spectrum of the full image (downsampled 2x by area to keep the FFT cheap; keeps cycles/image up to h/4)
    small = im.reshape(h // 2, 2, w // 2, 2).mean((1, 3))
    f = np.abs(np.fft.rfft2(small - small.mean())) ** 2
    fy = np.fft.fftfreq(small.shape[0]) * small.shape[0]  # cycles per image
    fx = np.fft.rfftfreq(small.shape[1]) * small.shape[1]
    rad = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    above = float(f[rad > NATIVE / 2].sum() / f.sum())
    return {"std": float(crop.std()), "lap_std": float(lap.std()), "p_above_native": above}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", nargs="+", required=True)
    ap.add_argument("--group-by", nargs="*", default=["size", "method", "cfg"])
    ap.add_argument("--filter", nargs="*", default=[])
    ap.add_argument("--csv")
    a = ap.parse_args()
    filters = dict(f.split("=", 1) for f in a.filter)
    groups = defaultdict(list)
    for out in a.out:
        out = Path(out)
        cache = out / "speckle"
        cache.mkdir(exist_ok=True)
        for js in sorted(glob.glob(str(out / "*.json"))):
            r = json.loads(Path(js).read_text())
            if r.get("status") != "ok" or "image" not in r or not Path(r["image"]).exists():
                continue
            if not all(str(val(r, k)) == v for k, v in filters.items()):
                continue
            cf = cache / f"{r['id']}.json"
            m = json.loads(cf.read_text()) if cf.exists() else measure(r["image"])
            cf.write_text(json.dumps(m))
            groups[tuple(val(r, k) for k in a.group_by)].append(m)
    rows = []
    print("  ".join(f"{k:>14}" for k in a.group_by) + f"  {'n':>3}  {'std':>6}  {'lap_std':>8}  {'±':>5}  {'p>native':>8}")
    for key, ms in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        lap = np.array([m["lap_std"] for m in ms])
        row = dict(zip(a.group_by, key))
        row.update(n=len(ms), std=float(np.mean([m["std"] for m in ms])), lap_std=float(lap.mean()), lap_se=float(lap.std(ddof=1) / np.sqrt(len(lap))) if len(lap) > 1 else 0.0,
                   p_above_native=float(np.mean([m["p_above_native"] for m in ms])))
        rows.append(row)
        print("  ".join(f"{str(v):>14}" for v in key) + f"  {row['n']:>3}  {row['std']:>6.1f}  {row['lap_std']:>8.2f}  {row['lap_se']:>5.2f}  {row['p_above_native']:>8.4f}")
    if a.csv:
        import csv

        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
