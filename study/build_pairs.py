#!/usr/bin/env python
"""Human study, step 1: build the blind image pairs from the finished grids.

Every pair is REG against one other method on the same prompt and seed at the same output size. Pairs are taken by rule
(every prompt that both methods have), never by hand. For each image of a pair two files are written under hashed names
that carry no method information: <hash>_fit.jpg (1400 px, the whole image) and <hash>_full.jpg (full resolution, JPEG
quality 95 without chroma subsampling, for the 1:1 view). The private manifest maps hashes back to methods.

  source env.sh
  python study/build_pairs.py [--out $REG_DATA/human_study/v1] [--limit 3] [--workers 8]
"""
from __future__ import annotations

import argparse, hashlib, json, os, secrets
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
REG = "anchor_base_pyr_s12"
# key, output size, REG directory, other directory, other method, other weight, description
COMPARISONS = [
    ("4k_cfg16", 4096, "lumina_4k_n100", "lumina_4k_n100", "cfg", 16.0, "REG vs plain CFG at the re-tuned weight 16, 4K"),
    ("4k_esrgan", 4096, "lumina_4k_n100", "upscale_4k", "up_realesrgan", 4.0, "REG vs 1K image upscaled 4x by Real-ESRGAN, 4K"),
    ("4k_hires", 4096, "lumina_4k_n100", "reg_4k_baselines", "hires_s035", 4.0, "REG vs hires fix (strength 0.35, native weight), 4K"),
    ("6k_cfg16", 6144, "lumina_6k_n100", "lumina_6k_n100", "cfg", 16.0, "REG vs plain CFG at the re-tuned weight 16, 6K"),
]


def index(d: Path, size: int, method: str, cfg: float) -> dict[str, dict]:
    out = {}
    for j in sorted(d.glob("*.json")):
        r = json.loads(j.read_text()); t = r.get("tags", {})
        if r.get("status") == "ok" and int(r.get("height", 0)) == size and t.get("method") == method and float(t.get("cfg", -1)) == cfg \
                and Path(r.get("image", "")).exists():
            out[t["prompt_id"]] = r
    return out


def convert(job: tuple[str, str, str]) -> tuple[str, int]:
    src, fit, full = job
    if not (Path(fit).exists() and Path(full).exists()):
        im = Image.open(src).convert("RGB")
        im.save(full + ".tmp", "JPEG", quality=95, subsampling=0, optimize=True); os.replace(full + ".tmp", full)
        s = im.resize((1400, round(1400 * im.height / im.width)), Image.LANCZOS)
        s.save(fit + ".tmp", "JPEG", quality=92, subsampling=0); os.replace(fit + ".tmp", fit)
    return full, os.path.getsize(full)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = Path(os.environ.get("REG_DATA", "/data1/moghadamp/reg_data"))
    ap.add_argument("--out", default=str(data / "human_study" / "v1"))
    ap.add_argument("--limit", type=int, default=0, help="pairs per comparison (0 = all); for a quick test build")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); img = out / "img"; img.mkdir(parents=True, exist_ok=True)
    mf = out / "manifest.json"
    old = json.loads(mf.read_text()) if mf.exists() else {}
    salt = old.get("salt") or secrets.token_hex(16)  # kept across rebuilds so that hashes (and rater assignments) stay valid
    h = lambda *parts: hashlib.sha1("|".join((salt, *parts)).encode()).hexdigest()[:16]
    pairs, jobs = [], {}
    for key, size, reg_dir, oth_dir, method, cfg, desc in COMPARISONS:
        reg = index(data / "outputs" / reg_dir, size, REG, 4.0)
        oth = index(data / "outputs" / oth_dir, size, method, cfg)
        pids = sorted(set(reg) & set(oth))
        if a.limit:
            pids = pids[:a.limit]
        for p in pids:
            assert reg[p]["prompt"] == oth[p]["prompt"] and reg[p]["seed"] == oth[p]["seed"], (key, p)
            sides = {}
            for role, r in (("reg", reg[p]), ("other", oth[p])):
                hh = h(key, p, role)  # per pair and role: the REG image has a different name in every comparison
                jobs[hh] = (r["image"], str(img / f"{hh}_fit.jpg"), str(img / f"{hh}_full.jpg"))
                sides[role] = {"hash": hh, "source": r["image"], "id": r["id"]}
            pairs.append({"pair": h("pair", key, p), "comparison": key, "size": size, "prompt_id": p, "prompt": reg[p]["prompt"],
                          "seed": reg[p]["seed"], **sides})
        print(f"{key:>10}: {len(pids)} pairs  ({desc})")
    with ProcessPoolExecutor(a.workers) as ex:
        sizes = [s for _, s in ex.map(convert, jobs.values(), chunksize=4)]
    mf.write_text(json.dumps({"salt": salt, "comparisons": {c[0]: {"size": c[1], "other": c[4], "other_cfg": c[5], "desc": c[6]} for c in COMPARISONS},
                              "pairs": pairs}, indent=1))
    print(f"{len(pairs)} pairs, {len(jobs)} images, full-resolution JPEGs {sum(sizes) / 2**30:.1f} GB (mean {sum(sizes) / len(sizes) / 2**20:.1f} MB) -> {out}")


if __name__ == "__main__":
    main()
