"""Job files and their division across servers.

A job file is JSONL: one row per image. Each server runs the rows where
``idx % num_servers == server_id`` (idx = 0-based line number, so the split is a pure function
of the file and needs no coordination). A row is done when ``<out>/<id>.json`` exists, so a
crashed or restarted run just skips finished rows.

Row keys (all optional except ``prompt``):
  id            unique stem for the output files; default ``r{idx:06d}``
  model         lumina | flux | qwen (default lumina)
  prompt, negative, height, width, steps, guidance, seed
  extrap        none | yarn | ntk;  static_extrap: bool (plain YaRN/NTK, no DyPE dynamics);
                extrap_mscale: bool (default True; False drops YaRN's attention temperature, diagnostic)
  Lumina: cfg_trunc_ratio, cfg_normalization, shift, and the guidance-zoo keys of
          reg.guidance.build_guidance: cfg_start, fdg {w_low, w_high, down}, pag/seg/tpg {scale, layers, ...}
  FLUX:   true_cfg_scale, shift_mu, quant (fp8)
  tags          free-form dict copied into the result JSON
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL_DEFAULTS = {
    "lumina": {"steps": 30, "guidance": 4.0},  # guidance = CFG scale
    "flux": {"steps": 28, "guidance": 3.5},  # guidance = embedded (distilled) guidance; true_cfg_scale separate
    "qwen": {"steps": 50, "guidance": 4.0},  # guidance = true CFG scale
}
COMMON_DEFAULTS = {"negative": "", "height": 1024, "width": 1024, "seed": 0, "extrap": "none", "static_extrap": False}


def read_jobs(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            if "prompt" not in row:
                raise ValueError(f"{path}:{idx + 1}: row has no prompt")
            row.setdefault("model", "lumina")
            if row["model"] not in MODEL_DEFAULTS:
                raise ValueError(f"{path}:{idx + 1}: unknown model {row['model']!r}")
            for k, v in {**COMMON_DEFAULTS, **MODEL_DEFAULTS[row["model"]]}.items():
                row.setdefault(k, v)
            row.setdefault("id", f"r{len(rows):06d}")
            row["idx"] = len(rows)
            rows.append(row)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"{path}: duplicate ids {dupes[:5]}")
    return rows


def my_rows(rows: list[dict], server_id: int, num_servers: int) -> list[dict]:
    if not 0 <= server_id < num_servers:
        raise ValueError(f"server_id {server_id} not in [0, {num_servers})")
    return [r for r in rows if r["idx"] % num_servers == server_id]


def is_done(row: dict, out: str | Path) -> bool:
    return (Path(out) / f"{row['id']}.json").exists()


def pending(rows: list[dict], out: str | Path) -> list[dict]:
    return [r for r in rows if not is_done(r, out)]


def write_jobs(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "idx"}) + "\n")
