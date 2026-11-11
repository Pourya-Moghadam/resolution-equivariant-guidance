#!/usr/bin/env python
"""README headline figure: HPSv2 against output resolution on Lumina-Image 2.0, for plain CFG at the native weight,
plain CFG at the best weight of the sweep at each size, and REG at the native weight.

  python scripts/readme_figure.py        # writes assets/quality_vs_resolution_{light,dark}.svg

Standard library only (no matplotlib), so it runs anywhere. Reads the committed tables:
  results/kill_or_go_pref.csv   1K plain CFG (trunc 1.0, n = 200)
  results/reg_weight_curve.csv  2K-5K plain CFG and the REG rule cells (n = 100 at 2K/3K, 50 at 4K/5K)
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIZES = [1024, 2048, 3072, 4096, 5120]
NATIVE_W = 4.0
RULE = {3072: "anchor_base_s12", 4096: "anchor_base_pyr_s12", 5120: "anchor_base_pyr_s12"}  # the REG rule's cell per size

THEMES = {
    "light": dict(surface="#fcfcfb", border="rgba(11,11,11,0.10)", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", series=("#2a78d6", "#eb6834", "#1baf7a")),
    "dark": dict(surface="#1a1a19", border="rgba(255,255,255,0.10)", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", series=("#3987e5", "#d95926", "#199e70")),
}
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def load() -> list[tuple[str, int, list[tuple[int, float]]]]:
    plain: dict[int, dict[float, float]] = {s: {} for s in SIZES}
    reg: dict[int, float] = {}
    with open(ROOT / "results" / "kill_or_go_pref.csv", newline="") as f:
        for r in csv.DictReader(f):
            if int(r["size"]) == 1024 and float(r["trunc"]) == 1.0:
                plain[1024][float(r["cfg"])] = float(r["hps"])
    with open(ROOT / "results" / "reg_weight_curve.csv", newline="") as f:
        for r in csv.DictReader(f):
            size, w = int(r["size"]), float(r["cfg"])
            if r["method"] == "cfg":
                plain[size][w] = float(r["mean"])
            elif RULE.get(size) == r["method"] and w == NATIVE_W:
                reg[size] = float(r["mean"])
    return [  # label, colour slot, points
        ("REG at the native weight", 0, [(s, reg[s]) for s in SIZES if s in reg]),
        ("Plain CFG at the native weight", 1, [(s, plain[s][NATIVE_W]) for s in SIZES]),
        ("Plain CFG, weight re-tuned per size", 2, [(s, max(plain[s].values())) for s in SIZES]),
    ]


def render(theme: dict, series) -> str:
    W, H = 800, 440
    left, right, top, bottom = 52, 64, 112, 62
    pw, ph = W - left - right, H - top - bottom
    ymax = 35.0

    def x(size: int) -> float:
        return left + pw * (SIZES.index(size) + 0.5) / len(SIZES)

    def y(v: float) -> float:
        return top + ph * (1 - v / ymax)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}" role="img" '
         f'aria-label="HPSv2 against output resolution: plain CFG at the native weight falls from 32 at 1K to 9 at 5K; REG stays at 30 to 31.">',
         f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="8" fill="{theme["surface"]}" stroke="{theme["border"]}"/>',
         f'<text x="24" y="36" font-size="16" font-weight="600" fill="{theme["ink"]}">Standard guidance collapses beyond the training resolution; REG holds</text>',
         f'<text x="24" y="57" font-size="12" fill="{theme["ink2"]}">HPSv2 preference score (higher is better) · Lumina-Image 2.0, trained at 1K · native guidance weight 4</text>']

    lx = 24.0  # legend: one row, line key + marker, names in text ink
    for label, slot, _ in series:
        c = theme["series"][slot]
        o.append(f'<line x1="{lx}" y1="84" x2="{lx + 22}" y2="84" stroke="{c}" stroke-width="2" stroke-linecap="round"/>')
        o.append(f'<circle cx="{lx + 11}" cy="84" r="4.5" fill="{c}" stroke="{theme["surface"]}" stroke-width="2"/>')
        o.append(f'<text x="{lx + 30}" y="88" font-size="12" fill="{theme["ink2"]}">{label}</text>')
        lx += 30 + 5.7 * len(label) + 28

    for t in (0, 10, 20, 30):
        colour = theme["axis"] if t == 0 else theme["grid"]
        o.append(f'<line x1="{left}" y1="{y(t):.1f}" x2="{W - right}" y2="{y(t):.1f}" stroke="{colour}" stroke-width="1"/>')
        o.append(f'<text x="{left - 10}" y="{y(t) + 4:.1f}" font-size="12" text-anchor="end" fill="{theme["muted"]}">{t}</text>')
    for s in SIZES:
        o.append(f'<text x="{x(s):.1f}" y="{top + ph + 22}" font-size="12" text-anchor="middle" fill="{theme["muted"]}">{s // 1024}K</text>')
    o.append(f'<text x="{left + pw / 2}" y="{H - 16}" font-size="12" text-anchor="middle" fill="{theme["ink2"]}">'
             f'Output resolution (1K = 1024 × 1024 px, the training resolution)</text>')

    for label, slot, pts in sorted(series, key=lambda s: -s[1]):  # REG drawn last, on top
        c = theme["series"][slot]
        path = " ".join(f'{"M" if i == 0 else "L"}{x(s):.1f},{y(v):.1f}' for i, (s, v) in enumerate(pts))
        o.append(f'<path d="{path}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for s, v in pts:
            o.append(f'<circle cx="{x(s):.1f}" cy="{y(v):.1f}" r="4.5" fill="{c}" stroke="{theme["surface"]}" stroke-width="2"/>')
        s, v = pts[-1]  # value at the line end, in text ink
        o.append(f'<text x="{x(s) + 12:.1f}" y="{y(v) + 4:.1f}" font-size="13" font-weight="600" fill="{theme["ink"]}" '
                 f'stroke="{theme["surface"]}" stroke-width="4" paint-order="stroke">{v:.1f}</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def main() -> None:
    series = load()
    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    for name, theme in THEMES.items():
        path = out / f"quality_vs_resolution_{name}.svg"
        path.write_text(render(theme, series), encoding="utf-8", newline="\n")
        print(path.relative_to(ROOT))
    for label, _, pts in series:
        print(f"{label}: " + ", ".join(f"{s // 1024}K {v:.1f}" for s, v in pts))


if __name__ == "__main__":
    main()
