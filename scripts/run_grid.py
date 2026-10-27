#!/usr/bin/env python
"""Run this server's share of a job file; one PNG + JSON per row; resume-safe.

  source env.sh
  python scripts/run_grid.py --jobs jobs/smoke.jsonl --out $REG_DATA/outputs/smoke_grid
  # server id / count default to REG_SERVER_ID / REG_NUM_SERVERS from env.sh

Rows are grouped by (model, quant) so each model loads once; DyPE is switched per row.
Across the cluster (from reg1):
  bash scripts/fanout.sh 'cd ~/reg && tmux new -d -s job "python scripts/run_grid.py --jobs jobs/smoke.jsonl \
      --out $REG_DATA/outputs/smoke_grid 2>&1 | tee -a $REG_DATA/outputs/smoke_grid/log_$(hostname -s).txt"'
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path

from reg.jobs import my_rows, pending, read_jobs


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unknown"


NEEDED_GB = {"lumina": 14, "flux": 26, "qwen": 22}  # bf16 Lumina / bf16 FLUX / fp8 Qwen at 1K, from measured peaks + headroom


def wait_for_gpu(needed_gb: float, timeout_s: float, poll_s: float = 60.0) -> float:
    """Block until `needed_gb` of GPU memory is free (the servers are shared with other users).

    Returns the free GB when the wait ended; raises TimeoutError if it never came free.
    """
    import torch

    t0 = time.perf_counter()
    while True:
        free_gb = torch.cuda.mem_get_info()[0] / 2**30
        if free_gb >= needed_gb:
            return free_gb
        if time.perf_counter() - t0 > timeout_s:
            raise TimeoutError(f"only {free_gb:.1f} GB free after {timeout_s:.0f}s, need {needed_gb} GB (another user on the GPU?)")
        print(f"[{socket.gethostname()}] waiting for GPU: {free_gb:.1f} GB free, need {needed_gb} GB", flush=True)
        time.sleep(poll_s)


def load_model(model: str, quant: str | None):
    if model == "lumina":
        from reg.models.lumina import Lumina

        return Lumina()
    if model == "flux":
        from reg.models.flux import Flux

        return Flux(quant=quant)
    from reg.models.qwen import Qwen

    return Qwen(quant=quant or "fp8")


def generate_kwargs(row: dict) -> dict:
    kw = dict(height=row["height"], width=row["width"], steps=row["steps"], guidance_scale=row["guidance"],
              seed=row["seed"], negative_prompt=row["negative"])
    extra = {"lumina": ("cfg_trunc_ratio", "cfg_normalization", "shift"),
             "flux": ("true_cfg_scale", "shift_mu"),
             "qwen": ("shift_mu",)}[row["model"]]
    for k in extra:
        if k in row:
            kw[k] = row[k]
    if row["model"] == "lumina":  # guidance zoo: CFG interval, FDG, PAG/SEG/TPG from the row (reg/guidance)
        from reg.guidance import build_guidance

        kw["guidance"] = build_guidance(row)
    return kw


def needs_negative(row: dict) -> bool:
    return (row["model"] == "flux" and row.get("true_cfg_scale", 1.0) > 1) or (row["model"] == "qwen" and row["guidance"] > 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-id", type=int, default=int(os.environ.get("REG_SERVER_ID", 0)))
    ap.add_argument("--num-servers", type=int, default=int(os.environ.get("REG_NUM_SERVERS", 1)))
    ap.add_argument("--dry-run", action="store_true", help="list this server's pending rows and exit")
    ap.add_argument("--gpu-wait", type=float, default=6 * 3600, help="seconds to wait for a shared GPU to free up before loading a model (0 = don't wait)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_jobs(args.jobs)
    mine = my_rows(rows, args.server_id, args.num_servers)
    todo = pending(mine, out)
    host = socket.gethostname()
    print(f"[{host}] server {args.server_id}/{args.num_servers}: {len(rows)} rows total, {len(mine)} mine, {len(todo)} pending", flush=True)
    if args.dry_run:
        for r in todo:
            print(r["idx"], r["id"], r["model"], f"{r['height']}x{r['width']}", r["extrap"])
        return

    import torch

    from reg.extrap.dype import apply_dype, remove_dype

    sha = git_sha()
    # group by (model, quant) so each model is loaded once; keep file order within a group
    groups: dict[tuple, list[dict]] = {}
    for r in todo:
        groups.setdefault((r["model"], r.get("quant")), []).append(r)

    n_ok = n_fail = 0
    for (model_name, quant), group in groups.items():
        needed = NEEDED_GB[model_name] + (6 if max(r["height"] * r["width"] for r in group) > 2048**2 else 0)
        if args.gpu_wait > 0:
            free = wait_for_gpu(needed, args.gpu_wait)
            print(f"[{host}] loading {model_name} ({quant or 'bf16'}): {free:.1f} GB free, need ~{needed} GB", flush=True)
        model = load_model(model_name, quant)
        if model_name != "lumina":  # two-phase models: fill the cache with encoders resident, then free them
            for r in group:
                model.encode([r["prompt"]], (r["negative"] or None) if needs_negative(r) else None)
            model.free_text_encoders()
        pipe = model.pipe
        for r in group:
            if r["extrap"] == "none":
                remove_dype(pipe.transformer)
            else:
                apply_dype(pipe.transformer, method=r["extrap"], dype=not r["static_extrap"], use_mscale=r.get("extrap_mscale", True))
            t0 = time.perf_counter()
            try:
                images, info = model.generate([r["prompt"]], **generate_kwargs(r))
                png = out / f"{r['id']}.png"
                images[0].save(png)
                rec = {**r, **info, "image": str(png), "seconds_per_image": info["seconds"], "wall_seconds": time.perf_counter() - t0,
                       "host": host, "gpu": torch.cuda.get_device_name(), "git": sha, "status": "ok"}
                n_ok += 1
            except Exception as e:  # record the failure so the row isn't retried blindly; delete the JSON to retry
                rec = {**r, "host": host, "git": sha, "status": "error", "error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()[-2000:]}
                n_fail += 1
                torch.cuda.empty_cache()
            (out / f"{r['id']}.json").write_text(json.dumps(rec, indent=2))
            print(f"[{host}] {r['idx']:>6} {r['id']} {rec['status']} {rec.get('seconds_per_image', 0):.1f}s", flush=True)
        remove_dype(pipe.transformer)
        del model, pipe
        torch.cuda.empty_cache()
    print(f"[{host}] done: {n_ok} ok, {n_fail} failed", flush=True)


if __name__ == "__main__":
    main()
