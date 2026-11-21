"""Merge the 4K DyPE-YaRN cells to 100 prompts: the reg_yarn / cfg_yarn records of reg_generality_grid (REG w4 and
plain w4, first 50 prompts) plus reg_4k_yarn (plain w16 on 100 prompts, plain w8 on the first 50, REG w4 and plain w4
on the other 50). Records are copied, images symlinked. Output: $REG_DATA/outputs/lumina_4k_yarn_n100.
"""
import json, os
from pathlib import Path

D = Path(os.environ.get("REG_DATA", "/data1/moghadamp/reg_data")) / "outputs"
out = D / "lumina_4k_yarn_n100"; out.mkdir(exist_ok=True); n = 0
# reg_4k_yarn rows carry no "subject" field; take it from the NTK n = 100 records of the same prompt (subject counts)
subj = {}
for j in (D / "lumina_4k_n100").glob("*.json"):
    r = json.loads(j.read_text())
    if r.get("subject"):
        subj[r["tags"]["prompt_id"]] = r["subject"]
for src in ("reg_generality_grid", "reg_4k_yarn"):
    for j in sorted((D / src).glob("*.json")):
        r = json.loads(j.read_text())
        if r.get("status") != "ok" or not r.get("id", "").startswith(("regyarn_", "cfgyarn_")) or int(r.get("height", 0)) != 4096:
            continue
        png = j.with_suffix(".png")
        if not png.exists():
            continue
        if not r.get("subject") and r["tags"].get("prompt_id") in subj:
            r["subject"] = subj[r["tags"]["prompt_id"]]
        (out / j.name).write_text(json.dumps(r))
        if not (out / png.name).exists():
            (out / png.name).symlink_to(png)
        n += 1
print("merged", n, "records into", out)
