"""Merge the three main 4K cells on Lumina into one output directory with 100 prompts each, for scoring as a unit:
plain CFG w4 and w16 (kill_or_go, 4096, trunc 1.0: 50 prompts), REG rule w4 (reg_pyramid_grid, anchor_base_pyr_s12: 50)
and the same three cells on the other 50 prompts (reg_4k_ext). Records are copied with tags.method set ("cfg" for the
kill-or-go rows, which carry none) and images are symlinked. Output: $REG_DATA/outputs/lumina_4k_n100 (300 records).
Then: run_pref / run_eval --refs / speckle with --group-by size method cfg -> results/lumina_4k_n100_*.csv.
"""
import json, os
from pathlib import Path

D = Path(os.environ.get("REG_DATA", "/data1/moghadamp/reg_data")) / "outputs"
OUT = D / "lumina_4k_n100"
OUT.mkdir(exist_ok=True)
n = 0
for src in ("kill_or_go", "reg_pyramid_grid", "reg_4k_ext"):
    for j in sorted((D / src).glob("*.json")):
        r = json.loads(j.read_text())
        t = r.get("tags", {})
        if r.get("status") != "ok" or int(r.get("height", 0)) != 4096:
            continue
        if src == "kill_or_go":
            if float(t.get("trunc", r.get("cfg_trunc_ratio", 1.0))) != 1.0 or float(t["cfg"]) not in (4.0, 16.0):
                continue
            t["method"] = "cfg"
        elif src == "reg_pyramid_grid" and t.get("method") != "anchor_base_pyr_s12" or float(t.get("cfg", 0)) != 4.0 and t.get("method") != "cfg":
            continue
        png = j.with_suffix(".png")
        if not png.exists():
            continue
        r["tags"] = t
        (OUT / j.name).write_text(json.dumps(r))
        link = OUT / png.name
        if not link.exists():
            link.symlink_to(png)
        n += 1
print("merged", n, "records into", OUT)
