"""Generate-then-upscale baseline: take finished native-resolution (1K) images and upscale them to the target size,
writing run_grid-style records so run_pref / run_eval / speckle / reg_paired score them like any other cell.

The source images are the kill-or-go 1K rows (plain CFG at the native weight, the 1K default truncation); only
prompts that have a record in the reference 4K/5K output are used, so every row pairs with the REG and plain rows
by (prompt_id, seed). Two upscalers: Lanczos (PIL) and Real-ESRGAN x4plus (weights in $REG_DATA/weights/realesrgan;
a 4096 output needs 4x from 1024, a 5120 output is 4x then a Lanczos 1.25x). Run with the separate upscaler venv:

  /data1/moghadamp/reg_upscale_venv/bin/python scripts/upscale_baseline.py --src kill_or_go --ref reg_pyramid_grid \
      --out upscale_4k --target 4096 --methods lanczos realesrgan

Records: <out>/<id>.json with tags {experiment, size, cfg, method up_<name>, prompt_id, prompt_set, src_id, trunc},
status ok, seconds_per_image (upscale time) and src_seconds (the 1K generation time). Resume-safe: an existing
record is skipped. Does not import the reg package (the upscaler venv has its own torch).
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
DATA = Path(os.environ.get("REG_DATA", "/data1/moghadamp/reg_data"))


def load_ok(d: Path) -> list[dict]:
    rows = []
    for f in glob.glob(str(d / "*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if r.get("status") == "ok" and "tags" in r:
            rows.append(r)
    return rows


def lanczos(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.LANCZOS)


class RealESRGAN:
    def __init__(self, device: str):
        import torch
        import torchvision.transforms.functional as F
        sys.modules.setdefault("torchvision.transforms.functional_tensor", F)  # basicsr imports a module removed in torchvision 0.17
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.up = RealESRGANer(scale=4, model_path=str(DATA / "weights/realesrgan/RealESRGAN_x4plus.pth"), model=model,
                               tile=512, tile_pad=16, pre_pad=0, half=device.startswith("cuda"), device=torch.device(device))

    def __call__(self, img: Image.Image, size: int) -> Image.Image:
        arr = np.array(img.convert("RGB"))[:, :, ::-1]  # BGR, as the package expects
        out, _ = self.up.enhance(arr, outscale=4)
        res = Image.fromarray(out[:, :, ::-1])
        return res if res.size == (size, size) else lanczos(res, size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="kill_or_go", help="output dir (under $REG_DATA/outputs) with the native-resolution rows")
    ap.add_argument("--ref", default="reg_pyramid_grid", help="output dir whose --target rows define the prompt set")
    ap.add_argument("--out", default="upscale_4k")
    ap.add_argument("--target", type=int, default=4096)
    ap.add_argument("--src-size", type=int, default=1024)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--trunc", type=float, default=0.9)
    ap.add_argument("--methods", nargs="+", default=["lanczos", "realesrgan"], choices=["lanczos", "realesrgan"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out = DATA / "outputs" / a.out
    out.mkdir(parents=True, exist_ok=True)
    ref = {(r["tags"]["prompt_id"], int(r["seed"])) for r in load_ok(DATA / "outputs" / a.ref) if r["tags"].get("size") == a.target}
    src = [r for r in load_ok(DATA / "outputs" / a.src)
           if r["tags"].get("size") == a.src_size and float(r["tags"].get("cfg", -1)) == a.cfg
           and float(r["tags"].get("trunc", a.trunc)) == a.trunc and (r["tags"]["prompt_id"], int(r["seed"])) in ref]
    src.sort(key=lambda r: r["id"])
    print(f"{len(ref)} reference prompts at {a.target}, {len(src)} matching {a.src_size} rows (cfg {a.cfg}, trunc {a.trunc})", flush=True)

    ups = {}
    for m in a.methods:
        ups[m] = lanczos if m == "lanczos" else RealESRGAN(a.device)
    n = 0
    for m, up in ups.items():
        for r in src:
            rid = f"up_{m}_{a.target}_w{int(a.cfg)}_{r['tags']['prompt_id']}"
            rec_path, img_path = out / f"{rid}.json", out / f"{rid}.png"
            if rec_path.exists():
                continue
            t0 = time.time()
            img = up(Image.open(r["image"]).convert("RGB"), a.target)
            img.save(img_path)
            dt = time.time() - t0
            rec = {k: v for k, v in r.items() if k not in ("image", "id", "tags", "seconds", "peak_vram_gb", "cache_hits", "cache_misses", "host", "gpu", "git", "idx", "seconds_per_image")}
            rec.update(id=rid, height=a.target, width=a.target, image=str(img_path), status="ok", seconds_per_image=dt,
                       src_id=r["id"], src_seconds=r.get("seconds"),
                       tags={"experiment": a.out, "size": a.target, "cfg": a.cfg, "method": f"up_{m}",
                             "prompt_id": r["tags"]["prompt_id"], "prompt_set": r["tags"].get("prompt_set"),
                             "trunc": a.trunc, "src_id": r["id"]})
            rec_path.write_text(json.dumps(rec, indent=2))
            n += 1
            print(f"{rid} ok {dt:.1f}s", flush=True)
    print(f"done: {n} new images in {out}", flush=True)


if __name__ == "__main__":
    main()
