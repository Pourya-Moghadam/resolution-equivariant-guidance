"""Merge the three 4K ablation cells on Lumina to 100 prompts: single anchor (reg_anchor_grid, anchor_base w4), + pyramid
and + super-native (reg_pyramid_grid, anchor_base_pyr / anchor_base_s12 w4) on the first 50 prompts, and the same cells
on the other 50 (reg_4k_ablation_ext). Records copied, images symlinked. Output: $REG_DATA/outputs/lumina_4k_ablation_n100.
Plain CFG and REG at n = 100 are in lumina_4k_n100; reg_paired reads both directories.
"""
import json, os
from pathlib import Path

D = Path(os.environ.get("REG_DATA", "/data1/moghadamp/reg_data")) / "outputs"
out = D / "lumina_4k_ablation_n100"; out.mkdir(exist_ok=True); n = 0
WANT = {"reg_anchor_grid": {"anchor_base"}, "reg_pyramid_grid": {"anchor_base_pyr", "anchor_base_s12"},
        "reg_4k_ablation_ext": {"anchor_base", "anchor_base_pyr", "anchor_base_s12"}}
for src, methods in WANT.items():
    for j in sorted((D / src).glob("*.json")):
        r = json.loads(j.read_text()); t = r.get("tags", {})
        if r.get("status") != "ok" or int(r.get("height", 0)) != 4096 or t.get("method") not in methods or float(t.get("cfg", 0)) != 4.0:
            continue
        png = j.with_suffix(".png")
        if not png.exists():
            continue
        (out / j.name).write_text(json.dumps(r))
        if not (out / png.name).exists():
            (out / png.name).symlink_to(png)
        n += 1
print("merged", n, "records into", out)
