#!/usr/bin/env python
"""Generate images from one prompt at one resolution; write PNGs + a JSON sidecar with timing.

The sidecar (seconds, peak VRAM, git sha, host) is what scripts/benchmark.py aggregates.

  source env.sh
  python scripts/sample.py --prompt "a red fox in snow" --size 1024
  python scripts/sample.py --prompt "..." --height 2048 --width 2048 --steps 30 --guidance 4 --seed 0 \
      --out $REG_DATA/outputs/smoke
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="lumina", choices=["lumina"])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--size", type=int, default=None, help="square shortcut for --height/--width")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--cfg-trunc", type=float, default=1.0, help="Lumina cfg_trunc_ratio")
    ap.add_argument("--no-cfg-norm", action="store_true", help="disable Lumina cfg_normalization")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1, help="images (seeds seed..seed+n-1), one batch")
    ap.add_argument("--out", default=os.path.join(os.environ.get("REG_DATA", "."), "outputs", "smoke"))
    ap.add_argument("--tag", default=None, help="file stem; default <model>_<HxW>_s<seed>")
    args = ap.parse_args()
    if args.size:
        args.height = args.width = args.size

    import torch

    t_load = time.perf_counter()
    if args.model == "lumina":
        from reg.models.lumina import Lumina

        model = Lumina()
    load_s = time.perf_counter() - t_load

    images, info = model.generate(
        [args.prompt] * args.n,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        cfg_trunc_ratio=args.cfg_trunc,
        cfg_normalization=not args.no_cfg_norm,
        negative_prompt=args.negative,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = args.tag or f"{info['model']}_{args.height}x{args.width}_s{args.seed}"
    paths = []
    for i, im in enumerate(images):
        p = out / (f"{stem}.png" if args.n == 1 else f"{stem}_{i}.png")
        im.save(p)
        paths.append(str(p))

    record = {
        **info,
        "prompt": args.prompt,
        "negative": args.negative,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "guidance": args.guidance,
        "cfg_trunc_ratio": args.cfg_trunc,
        "cfg_normalization": not args.no_cfg_norm,
        "seed": args.seed,
        "n": args.n,
        "seconds_per_image": info["seconds"] / args.n,
        "load_seconds": load_s,
        "images": paths,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(),
        "git": git_sha(),
        "torch": torch.__version__,
    }
    (out / f"{stem}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in ("seconds_per_image", "peak_vram_gb", "load_seconds", "images")}))


if __name__ == "__main__":
    main()
