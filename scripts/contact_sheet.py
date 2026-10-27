#!/usr/bin/env python
"""Contact sheets from a job's outputs: one sheet per value of --per, rows × cols from two row/tag keys.

  source env.sh
  python scripts/contact_sheet.py --out $REG_DATA/outputs/pilot_cfg_sweep_b --per prompt_name --rows size --cols cfg --filter seed=0

Sheets go to <out>/sheets/<per>_<filters>.jpg. Keys are looked up in row['tags'] first, then the row.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def val(r: dict, k: str):
    return r.get("tags", {}).get(k, r.get(k))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", default="prompt_name", help="one sheet per value of this key")
    ap.add_argument("--rows", default="size")
    ap.add_argument("--cols", default="cfg")
    ap.add_argument("--filter", nargs="*", default=[], help="key=value pairs a row must match (string compare)")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--dest", default=None, help="sheet directory (default <out>/sheets)")
    args = ap.parse_args()

    out = Path(args.out)
    filters = dict(f.split("=", 1) for f in args.filter)
    rows = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(out / "*.json")))]
    rows = [r for r in rows if r.get("status") == "ok" and "image" in r and all(str(val(r, k)) == v for k, v in filters.items())]
    by_sheet: dict = defaultdict(dict)
    for r in rows:
        by_sheet[val(r, args.per)][(val(r, args.rows), val(r, args.cols))] = r
    dest = Path(args.dest or out / "sheets")
    dest.mkdir(parents=True, exist_ok=True)
    label_h, t = 28, args.tile
    suffix = "".join(f"_{k}{v}" for k, v in filters.items())
    for name, cells in sorted(by_sheet.items(), key=lambda kv: str(kv[0])):
        rv = sorted({k[0] for k in cells})
        cv = sorted({k[1] for k in cells})
        sheet = Image.new("RGB", (label_h + len(cv) * t, label_h + len(rv) * t), "white")
        d = ImageDraw.Draw(sheet)
        for j, c in enumerate(cv):
            d.text((label_h + j * t + 4, 8), f"{args.cols}={c}", fill="black")
        for i, rr in enumerate(rv):
            d.text((2, label_h + i * t + 4), str(rr), fill="black")
            for j, c in enumerate(cv):
                r = cells.get((rr, c))
                if r is None:
                    continue
                im = Image.open(r["image"]).convert("RGB")
                im.thumbnail((t, t), Image.LANCZOS)
                sheet.paste(im, (label_h + j * t, label_h + i * t))
        f = dest / f"{name}{suffix}.jpg"
        sheet.save(f, quality=85)
        print(f, sheet.size)


if __name__ == "__main__":
    main()
