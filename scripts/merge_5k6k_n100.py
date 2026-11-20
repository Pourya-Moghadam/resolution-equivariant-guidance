"""Merge the 5K and 6K Lumina cells to 100 prompts, as scripts/merge_4k_n100.py did at 4K: every ok record of
reg_5k_grid (5K: plain w4/w8/w16 and REG variants, 50 prompts) or reg_6k_grid (6K: plain w16, REG, 50 prompts) plus
the same plain w16 and REG cells on the other 50 prompts (reg_5k6k_ext). Records are copied, images symlinked.
Output: $REG_DATA/outputs/lumina_{5k,6k}_n100. Plain w4/w8 at 5K stay at 50 prompts (not extended).
Then: run_pref / run_eval --refs / speckle with --group-by size method cfg -> results/lumina_{5k,6k}_n100_*.csv.
"""
import json, os
from pathlib import Path

D = Path(os.environ.get("REG_DATA", "/data1/moghadamp/reg_data")) / "outputs"
for size, grid in ((5120, "reg_5k_grid"), (6144, "reg_6k_grid")):
    out = D / f"lumina_{size // 1024}k_n100"; out.mkdir(exist_ok=True); n = 0
    for src in (grid, "reg_5k6k_ext"):
        for j in sorted((D / src).glob("*.json")):
            r = json.loads(j.read_text())
            if r.get("status") != "ok" or int(r.get("height", 0)) != size:
                continue
            png = j.with_suffix(".png")
            if not png.exists():
                continue
            (out / j.name).write_text(json.dumps(r))
            if not (out / png.name).exists():
                (out / png.name).symlink_to(png)
            n += 1
    print("merged", n, "records into", out)
