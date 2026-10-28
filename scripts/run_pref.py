#!/usr/bin/env python
"""Human-preference scores (HPSv2, ImageReward) for a job's outputs.

Runs in the SEPARATE preference venv, not the project venv (the ``image-reward`` package and
HPSv2's vendored open_clip are written against transformers 4.x; the project venv has 5.x):

  /data1/moghadamp/reg_pref_venv/bin/python scripts/run_pref.py --out $REG_DATA/outputs/reg_anchor_pilot \
      --group-by size cfg method --csv results/reg_anchor_pilot_pref.csv [--metrics hps ir]

Mirrors scripts/run_eval.py: reads every <id>.json in --out with "status": "ok" (prompt, image,
tags), writes per-image scores to <out>/pref/<id>.json (resume-safe: an existing record is only
extended with metrics it lacks), prints a table of mean and standard error per group of
--group-by keys (looked up in row["tags"] first, then the row) and optionally writes it as CSV.

Metrics
  hps  HPSv2 (Wu et al. 2023; ``hpsv2`` package, ViT-H-14 fine-tuned on HPD v2). Reported as
       100 x the image-text cosine similarity, the convention of the HPSv2 paper and leaderboard
       (typical range 20-35). Default checkpoint v2.1 (``--hps-version v2.0`` for the original).
  ir   ImageReward-v1.0 (Xu et al. 2023; ``image-reward`` package, BLIP-based). Raw reward,
       roughly -2..2.

Both scorers see a 224x224 centre crop of the (shorter-side) resized image, so resolution is not
being scored, only content, and a 4096^2 PNG is first downscaled with PIL (LANCZOS, longer side
--max-side, default 1024) before the scorer's own 224 px transform; the extra resize step is
negligible against the final 18x reduction and keeps the per-image memory small. Scores are
therefore comparable across sizes but blind to detail finer than ~1/224 of the image.

Checkpoints: HPSv2 from the Hugging Face repo ``xswu/HPSv2`` into $HF_HOME; ImageReward into
$REG_DATA/weights/imagereward (same location as the project venv's ImageRewardScorer). Set
REG_DATA (or source env.sh) so both land on the data disk.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

from PIL import Image

REG_DATA = os.environ.get("REG_DATA", "/data1/moghadamp/reg_data")
os.environ.setdefault("HF_HOME", os.path.join(REG_DATA, "weights", "hf"))
os.environ.setdefault("HPS_ROOT", os.path.join(REG_DATA, "weights", "hpsv2"))

HPS_FILES = {"v2.0": "HPS_v2_compressed.pt", "v2.1": "HPS_v2.1_compressed.pt"}


def row_key(r: dict, keys: list[str]) -> tuple:
    return tuple(r.get("tags", {}).get(k, r.get(k)) for k in keys)


def load_image(path: str, max_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


class HPSScorer:
    """HPSv2 via the ``hpsv2`` package's vendored open_clip. The package's ``hpsv2.score`` reloads
    the 4 GB checkpoint on every call and first downloads the LAION ViT-H weights it immediately
    overwrites; this builds the model once, from the HPS checkpoint only."""

    def __init__(self, device: str = "cuda", version: str = "v2.1"):
        import huggingface_hub
        import torch
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

        self.torch = torch
        self.device = torch.device(device)
        model, _, self.preprocess = create_model_and_transforms(
            "ViT-H-14", None, precision="amp", device=self.device, output_dict=True, light_augmentation=True, aug_cfg={}
        )
        cp = huggingface_hub.hf_hub_download("xswu/HPSv2", HPS_FILES[version])
        self.checkpoint = cp
        state = torch.load(cp, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        self.model = model.to(self.device).eval()
        self.tokenizer = get_tokenizer("ViT-H-14")

    def score(self, image: Image.Image, prompt: str) -> float:
        torch = self.torch
        with torch.inference_mode(), torch.autocast("cuda", enabled=self.device.type == "cuda"):
            x = self.preprocess(image).unsqueeze(0).to(self.device)
            t = self.tokenizer([prompt]).to(self.device)
            out = self.model(x, t)
            s = (out["image_features"] @ out["text_features"].T).squeeze()
        return float(s) * 100.0


class IRScorer:
    """ImageReward-v1.0 via the ``image-reward`` package (needs transformers 4.x). The package
    imports OpenAI's ``clip`` for its CLIP/aesthetic scorers, which we do not use; a stub module
    satisfies the import."""

    def __init__(self, device: str = "cuda"):
        import torch

        os.environ.setdefault("TORCH_HOME", os.path.join(REG_DATA, "weights", "torch"))
        sys.modules.setdefault("clip", types.ModuleType("clip"))
        import ImageReward as RM

        self.torch = torch
        self.root = os.path.join(REG_DATA, "weights", "imagereward")
        self.model = RM.load("ImageReward-v1.0", device=device, download_root=self.root)

    def score(self, image: Image.Image, prompt: str) -> float:
        with self.torch.inference_mode():
            return float(self.model.score(prompt, image))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="job output dir with <id>.json/<id>.png from run_grid.py")
    ap.add_argument("--group-by", nargs="*", default=["size", "cfg"], help="row/tag keys that define a setting")
    ap.add_argument("--csv", default=None, help="write the per-setting table here")
    ap.add_argument("--metrics", nargs="*", default=["hps", "ir"], choices=["hps", "ir"])
    ap.add_argument("--hps-version", default="v2.1", choices=sorted(HPS_FILES))
    ap.add_argument("--max-side", type=int, default=1024, help="downscale larger images to this longer side before scoring")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out)
    rows = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(out / "*.json")))]
    rows = [r for r in rows if r.get("status") == "ok" and "image" in r]
    pdir = out / "pref"
    pdir.mkdir(exist_ok=True)
    print(f"{len(rows)} finished rows in {out}", flush=True)

    # Only load a scorer if some row still needs it.
    recs = {}
    for r in rows:
        f = pdir / f"{r['id']}.json"
        recs[r["id"]] = json.loads(f.read_text()) if f.exists() else {"id": r["id"]}
    scorers = {}
    for m in args.metrics:
        if any(m not in rec for rec in recs.values()):
            t0 = time.time()
            scorers[m] = HPSScorer(args.device, args.hps_version) if m == "hps" else IRScorer(args.device)
            print(f"loaded {m} in {time.time() - t0:.1f}s", flush=True)

    timings: dict[str, list[float]] = defaultdict(list)
    for i, r in enumerate(rows):
        rec = recs[r["id"]]
        need = [m for m in scorers if m not in rec]
        if need:
            img = load_image(r["image"], args.max_side)
            for m in need:
                t0 = time.time()
                v = scorers[m].score(img, r["prompt"])
                timings[m].append(time.time() - t0)
                if v != v:
                    raise RuntimeError(f"{m} returned NaN for {r['id']}")
                rec[m] = v
            if "hps" in need:
                rec["hps_version"] = args.hps_version
            (pdir / f"{r['id']}.json").write_text(json.dumps(rec))
        if i % 20 == 0:
            print(f"  {i}/{len(rows)} {r['id']}", flush=True)
    for m, ts in timings.items():
        print(f"{m}: {len(ts)} images, {1000 * statistics.mean(ts):.0f} ms/image (scoring only, excl. PNG decode)", flush=True)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[row_key(r, args.group_by)].append(recs[r["id"]])
    table = []
    for key, grecs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        row = dict(zip(args.group_by, key))
        row["n"] = len(grecs)
        for m in args.metrics:
            vals = [x[m] for x in grecs if m in x]
            if vals:
                row[m] = round(statistics.mean(vals), 4)
                row[m + "_se"] = round(statistics.stdev(vals) / len(vals) ** 0.5, 4) if len(vals) > 1 else 0.0
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
