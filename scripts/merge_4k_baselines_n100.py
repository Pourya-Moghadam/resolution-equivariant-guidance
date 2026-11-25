"""Merge the 4K baseline rows on Lumina into one directory at n = 100: hires fix (strength 0.35 at second-stage weights 4,
8, 16; strength 0.60 at 4) and APG (projection only, full) from reg_4k_baselines and reg_4k_baselines2, and the two HiFlow
variants from reg_4k_hiflow. Records copied (subject added from lumina_4k_n100 where missing), images symlinked.
Output: $REG_DATA/outputs/lumina_4k_baselines_n100 (700 records). Plain CFG and REG at n = 100 are in lumina_4k_n100.
"""
import collections, json, os
from pathlib import Path

D = Path(os.environ.get("REG_DATA", "/data1/moghadamp/reg_data")) / "outputs"
out = D / "lumina_4k_baselines_n100"; out.mkdir(exist_ok=True); c = collections.Counter()
subj = {}
for j in (D / "lumina_4k_n100").glob("*.json"):
    r = json.loads(j.read_text())
    if r.get("subject"):
        subj[r["tags"]["prompt_id"]] = r["subject"]
for src in ("reg_4k_baselines", "reg_4k_baselines2", "reg_4k_hiflow"):
    for j in sorted((D / src).glob("*.json")):
        r = json.loads(j.read_text()); t = r.get("tags", {}); png = j.with_suffix(".png")
        if r.get("status") != "ok" or not png.exists():
            continue
        if "subject" not in r and t["prompt_id"] in subj:
            r["subject"] = subj[t["prompt_id"]]
        (out / j.name).write_text(json.dumps(r))
        if not (out / png.name).exists():
            (out / png.name).symlink_to(png)
        c[(t["method"], float(t["cfg"]))] += 1
print(sum(c.values()), dict(c))
