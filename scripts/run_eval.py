#!/usr/bin/env python
"""Evaluate a job's outputs (Phase 0 step 7): per-image CLIP / ImageReward / repetition count,
plus patch-FID and patch-KID per setting against a reference image set.

  source env.sh
  python scripts/run_eval.py --out $REG_DATA/outputs/pilot_cfg_sweep_b --refs $REG_DATA/datasets/aesthetic_4k/eval/size_2048/images \
      --group-by size cfg --csv results/pilot_cfg_sweep_b_eval.csv

Reads every <id>.json the runner wrote (prompt, image path, tags). Per-image scores go to
<out>/eval/<id>.json (resume-safe); the set-level FID/KID is computed for each group of rows
that share the values of --group-by (taken from row['tags'] or the row itself). The
repetition count is computed only for rows that carry a 'subject' (single-subject prompts).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image

from reg.eval.patches import iter_patches


def row_key(r: dict, keys: list[str]) -> tuple:
    return tuple(r.get("tags", {}).get(k, r.get(k)) for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="job output dir with <id>.json/<id>.png from run_grid.py")
    ap.add_argument("--refs", default=None, help="directory of reference images for patch-FID/KID (optional)")
    ap.add_argument("--max-refs", type=int, default=400)
    ap.add_argument("--group-by", nargs="*", default=["size", "cfg"], help="row/tag keys that define a setting")
    ap.add_argument("--patch", type=int, default=1024)
    ap.add_argument("--crops", type=int, default=16)
    ap.add_argument("--csv", default=None, help="write the per-setting table here")
    ap.add_argument("--skip", nargs="*", default=[], choices=["clip", "ir", "rep", "fid"], help="metrics to skip")
    ap.add_argument("--with-ir", action="store_true", help="also compute ImageReward (package is incompatible with transformers 5; parked, see 06)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out)
    rows = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(out / "*.json")))]
    rows = [r for r in rows if r.get("status") == "ok" and "image" in r]
    missing = [r for r in rows if not os.path.exists(r["image"])]
    if missing:  # a merged copy of a running grid: the row's json may arrive before its png
        print(f"skipping {len(missing)} rows whose image is not here yet (e.g. {missing[0]['id']})", flush=True)
        rows = [r for r in rows if os.path.exists(r["image"])]
    evdir = out / "eval"
    evdir.mkdir(exist_ok=True)
    print(f"{len(rows)} finished rows in {out}", flush=True)

    import torch

    clip = ir = rep = inc = None
    if "clip" not in args.skip:
        from reg.eval.clip_score import ClipScorer

        clip = ClipScorer(args.device)
    if args.with_ir and "ir" not in args.skip:
        from reg.eval.image_reward import ImageRewardScorer

        ir = ImageRewardScorer(args.device)
    if "rep" not in args.skip and any("subject" in r for r in rows):
        from reg.eval.repetition import RepetitionCounter

        rep = RepetitionCounter(args.device)
    if "fid" not in args.skip and args.refs:
        from reg.eval.fid import InceptionFeatures, fid_from_features, kid_from_features, reference_features

        inc = InceptionFeatures(args.device)
        ref_paths = sorted(p for p in glob.glob(os.path.join(args.refs, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg")))[: args.max_refs]
        ref_feats = reference_features(inc, ref_paths, args.patch, args.crops)
        print(f"reference features: {tuple(ref_feats.shape)} from {len(ref_paths)} images", flush=True)

    # --- per image -------------------------------------------------------------------------
    feats_by_group: dict[tuple, list[torch.Tensor]] = defaultdict(list)
    per_image: dict[str, dict] = {}
    for i, r in enumerate(rows):
        f = evdir / f"{r['id']}.json"
        rec = json.loads(f.read_text()) if f.exists() else {"id": r["id"]}
        img = None
        need = [m for m, obj in (("clip", clip), ("ir", ir), ("rep", rep)) if obj is not None and m not in rec]
        if inc is not None and "feats" not in rec:
            need.append("fid")
        if need:
            img = Image.open(r["image"]).convert("RGB")
            patches = list(iter_patches(img, args.patch, args.crops)) if (clip or inc) else []
            if "clip" in need:
                rec["clip"] = clip.score(img, r["prompt"])
                rec["clip_patch"] = clip.score_patches(patches, r["prompt"])
            if "ir" in need:
                rec["ir"] = ir.score(img, r["prompt"])
            if "rep" in need and "subject" in r:
                d = rep.count(img, r["subject"])
                rec["rep_count"] = d["count"]
                rec["rep_scores"] = d["scores"]
            if "fid" in need:
                feats = inc(patches)
                torch.save(feats, evdir / f"{r['id']}.feats.pt")
                rec["feats"] = str(evdir / f"{r['id']}.feats.pt")
            f.write_text(json.dumps(rec))
        if inc is not None:
            feats_by_group[row_key(r, args.group_by)].append(torch.load(rec["feats"]))
        per_image[r["id"]] = rec
        if i % 20 == 0:
            print(f"  {i}/{len(rows)} {r['id']}", flush=True)

    # --- per setting -----------------------------------------------------------------------
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[row_key(r, args.group_by)].append(per_image[r["id"]])
    table = []
    for key, recs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        row = dict(zip(args.group_by, key))
        row["n"] = len(recs)
        for m in ("clip", "clip_patch", "ir"):
            vals = [x[m] for x in recs if m in x]
            if vals:
                row[m] = round(statistics.mean(vals), 4)
                row[m + "_se"] = round(statistics.stdev(vals) / len(vals) ** 0.5, 4) if len(vals) > 1 else 0.0
        reps = [x["rep_count"] for x in recs if "rep_count" in x]
        if reps:
            row["rep_mean"] = round(statistics.mean(reps), 3)
            row["rep_dup_frac"] = round(sum(c > 1 for c in reps) / len(reps), 3)
            row["rep_zero_frac"] = round(sum(c == 0 for c in reps) / len(reps), 3)
        if inc is not None and feats_by_group.get(key):
            g = torch.cat(feats_by_group[key])
            row["patch_fid"] = round(fid_from_features(g, ref_feats), 3)
            k, ks = kid_from_features(g, ref_feats)
            row["patch_kid"] = round(k, 5)
            row["patch_kid_std"] = round(ks, 5)
            row["n_patches"] = len(g)
        table.append(row)
        print(row, flush=True)
    if args.csv and table:
        keys = sorted({k for r in table for k in r}, key=lambda k: (k not in args.group_by, k))
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
            w.writeheader()
            w.writerows(table)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
