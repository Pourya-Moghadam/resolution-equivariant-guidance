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
    ap.add_argument("--model", default="lumina", choices=["lumina", "flux"])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--size", type=int, default=None, help="square shortcut for --height/--width")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=None, help="default: lumina 30, flux 28")
    ap.add_argument("--guidance", type=float, default=None, help="default: lumina 4.0 (CFG), flux 3.5 (embedded)")
    ap.add_argument("--cfg-trunc", type=float, default=1.0, help="Lumina cfg_trunc_ratio")
    ap.add_argument("--no-cfg-norm", action="store_true", help="disable Lumina cfg_normalization")
    ap.add_argument("--shift", type=float, default=None, help="Lumina: override the fixed scheduler shift (default 6)")
    ap.add_argument("--true-cfg", type=float, default=1.0, help="FLUX: real CFG scale with --negative (>1 enables)")
    ap.add_argument("--shift-mu", type=float, default=None, help="FLUX: override the dynamic-shift mu")
    ap.add_argument("--quant", default=None, choices=[None, "fp8"], help="FLUX: float8 weight-only transformer (needed at 4K on 32 GB)")
    ap.add_argument("--extrap", default="none", choices=["none", "yarn", "ntk"], help="RoPE extrapolation (DyPE unless --static-extrap)")
    ap.add_argument("--static-extrap", action="store_true", help="plain YaRN/NTK without DyPE's kappa(t) dynamics")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1, help="images (seeds seed..seed+n-1), one batch")
    ap.add_argument("--out", default=os.path.join(os.environ.get("REG_DATA", "."), "outputs", "smoke"))
    ap.add_argument("--tag", default=None, help="file stem; default <model>_<HxW>_s<seed>")
    args = ap.parse_args()
    if args.size:
        args.height = args.width = args.size
    defaults = {"lumina": (30, 4.0), "flux": (28, 3.5)}[args.model]
    args.steps = args.steps if args.steps is not None else defaults[0]
    args.guidance = args.guidance if args.guidance is not None else defaults[1]

    import torch

    t_load = time.perf_counter()
    common = dict(height=args.height, width=args.width, steps=args.steps, guidance_scale=args.guidance,
                  seed=args.seed, negative_prompt=args.negative)
    if args.model == "lumina":
        from reg.models.lumina import Lumina

        model = Lumina()
        extra = dict(cfg_trunc_ratio=args.cfg_trunc, cfg_normalization=not args.no_cfg_norm, shift=args.shift)
        variant_tag = f"_sh{args.shift:g}" if args.shift is not None else ""
    else:
        from reg.models.flux import Flux

        model = Flux(quant=args.quant)
        extra = dict(true_cfg_scale=args.true_cfg, shift_mu=args.shift_mu)
        variant_tag = (f"_{args.quant}" if args.quant else "") + (f"_cfg{args.true_cfg:g}" if args.true_cfg > 1 else "") + (
            f"_mu{args.shift_mu:g}" if args.shift_mu is not None else "")
    if args.extrap != "none":
        from reg.extrap.dype import apply_dype

        if args.model == "flux":  # load order matters for memory: encoders first, then the DiT
            model.encode([args.prompt] * args.n, args.negative if args.true_cfg > 1 else None)
            model.free_text_encoders()
        apply_dype(model.pipe.transformer, method=args.extrap, dype=not args.static_extrap)
        variant_tag += f"_{'' if args.static_extrap else 'dy_'}{args.extrap}"
    load_s = time.perf_counter() - t_load  # for FLUX the real loading happens inside generate()

    images, info = model.generate([args.prompt] * args.n, **common, **extra)
    info["extrap"] = f"{'' if args.static_extrap else 'dy_'}{args.extrap}" if args.extrap != "none" else "none"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = args.tag or f"{info['model']}_{args.height}x{args.width}{variant_tag}_s{args.seed}"
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
