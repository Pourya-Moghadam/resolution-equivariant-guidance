#!/usr/bin/env python
"""Write jobs/kill_or_go.jsonl — the Phase 1 grid (PLAN.md Phase 1 step 3, decided 2026-10-26, CFG axis
extended 2026-10-27 after sweeps B/C in 08).

Lumina-Image 2.0 + dynamic NTK above 1K, 30 steps, one fixed seed per prompt (its index in the pool).
Prompts: prompts/kill_or_go_200.jsonl = 100 Aesthetic-4K captions (with reference image) + 100 own
single-subject prompts (with 'subject' for the repetition count). Higher resolutions take the first
N of each half so every cell has both prompt types.

  python scripts/make_kill_or_go.py            # prints the row count and GPU-h estimate
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# per resolution: CFG scales, interval ends (cfg_trunc_ratio), prompts per half, measured s/image (06)
GRID = {
    1024: dict(cfg=[2, 3, 4, 5, 6, 8, 10, 12], trunc=[0.5, 0.7, 0.9, 1.0], per_half=100, sec=16),
    2048: dict(cfg=[2, 3, 4, 5, 6, 8, 10, 12], trunc=[0.5, 0.7, 0.9, 1.0], per_half=50, sec=94),
    3072: dict(cfg=[2, 3, 4, 5, 6, 8, 10, 12, 16], trunc=[0.7, 1.0], per_half=50, sec=282),
    4096: dict(cfg=[3, 4, 5, 6, 8, 10, 12, 16], trunc=[1.0], per_half=25, sec=692),
}


def main() -> None:
    pool = [json.loads(l) for l in open(ROOT / "prompts" / "kill_or_go_200.jsonl")]
    a4k = [p for p in pool if "ref_file" in p]
    single = [p for p in pool if "subject" in p]
    assert len(a4k) == 100 and len(single) == 100, (len(a4k), len(single))
    seed_of = {p["id"]: i for i, p in enumerate(pool)}

    rows, hours = [], 0.0
    for size, g in GRID.items():
        prompts = a4k[: g["per_half"]] + single[: g["per_half"]]
        for pi, p in enumerate(prompts):
            for cfg in g["cfg"]:
                for tr in g["trunc"]:
                    row = {
                        "id": f"kog_{size}_c{cfg:g}_t{tr:g}_{p['id']}",
                        "model": "lumina",
                        "prompt": p["prompt"],
                        "height": size,
                        "width": size,
                        "seed": seed_of[p["id"]],
                        "guidance": float(cfg),
                        "cfg_trunc_ratio": tr,
                        "extrap": "none" if size == 1024 else "ntk",
                        "tags": {"experiment": "kill_or_go", "size": size, "cfg": float(cfg), "trunc": tr,
                                 "prompt_id": p["id"], "prompt_set": "a4k" if "ref_file" in p else "single"},
                    }
                    for k in ("subject", "expected_count", "ref_file"):
                        if k in p:
                            row[k] = p[k]
                    rows.append(row)
                    hours += g["sec"] / 3600
    # order: prompt-major within each resolution, resolutions interleaved by prompt index, so a partial
    # run already covers every (size, cfg, trunc) cell for the first prompts
    rows.sort(key=lambda r: (r["tags"]["prompt_id"], r["height"], r["guidance"], r["cfg_trunc_ratio"]))
    out = ROOT / "jobs" / "kill_or_go.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    per_size = {s: sum(r["height"] == s for r in rows) for s in GRID}
    print(f"{len(rows)} rows -> {out}; per size {per_size}; ~{hours:.0f} GPU-h ≈ {hours / 4 / 24:.1f} days on 4 GPUs")


if __name__ == "__main__":
    main()
