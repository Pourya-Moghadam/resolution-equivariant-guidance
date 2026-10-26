#!/usr/bin/env python
"""Phase 0 step 4: DyPE on FLUX.1-dev with our implementation (reg/extrap/dype.py).

Same prompt and seed once per method (default `base` = vanilla RoPE, `yarn` with DyPE), using
the stock diffusers pipeline through reg.models.flux.Flux with the DyPE schedule convention
(mu clamped to 1.15, see 07-dype-reference.md) for every method, so the methods differ only
in positional encoding. Writes PNG + JSON sidecars to $REG_DATA/outputs/dype_repro/.

  source env.sh
  python scripts/repro_dype_flux.py --size 2048
  python scripts/repro_dype_flux.py --size 4096 --methods yarn
  python scripts/repro_dype_flux.py --size 2048 --methods yarn --no-dype      # static YaRN
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

DYPE_MU = 1.15


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="A red fox standing in fresh snow at dawn, soft golden light, photorealistic")
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5, help="embedded guidance; FLUX.1-dev default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="base,yarn", help="comma list of base|ntk|yarn")
    ap.add_argument("--no-dype", action="store_true", help="static ntk/yarn instead of DyPE's kappa(t) dynamics")
    ap.add_argument("--shift-mu", type=float, default=DYPE_MU, help="schedule mu for all methods")
    ap.add_argument("--quant", default=None, choices=[None, "fp8"], help="float8 weight-only transformer (4K on 32 GB)")
    ap.add_argument("--out", default=os.path.join(os.environ.get("REG_DATA", "."), "outputs", "dype_repro"))
    args = ap.parse_args()

    import torch

    from reg.extrap.dype import apply_dype, remove_dype
    from reg.models.flux import Flux

    model = Flux(quant=args.quant)
    model.encode([args.prompt])  # fill the cache while only the text encoders are resident
    model.free_text_encoders()
    pipe = model.pipe

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for method in args.methods.split(","):
        use_dype = method != "base" and not args.no_dype
        tag = f"{'dy_' if use_dype else ''}{method}"
        if method == "base":
            remove_dype(pipe.transformer)
        else:
            apply_dype(pipe.transformer, method=method, dype=use_dype)

        images, info = model.generate(
            [args.prompt], height=args.size, width=args.size, steps=args.steps,
            guidance_scale=args.guidance, seed=args.seed, shift_mu=args.shift_mu,
        )
        stem = f"flux1-dev{'_' + args.quant if args.quant else ''}_{args.size}x{args.size}_{tag}_s{args.seed}"
        images[0].save(out / f"{stem}.png")
        rec = {
            **info, "method": tag, "dype": use_dype, "prompt": args.prompt, "height": args.size, "width": args.size,
            "steps": args.steps, "guidance": args.guidance, "seed": args.seed,
            "seconds_per_image": info["seconds"], "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(),
            "images": [str(out / f"{stem}.png")],
        }
        (out / f"{stem}.json").write_text(json.dumps(rec, indent=2))
        print(json.dumps({k: rec[k] for k in ("method", "seconds_per_image", "peak_vram_gb", "shift_mu", "images")}), flush=True)
    remove_dype(pipe.transformer)


if __name__ == "__main__":
    main()
