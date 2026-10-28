"""Guidance zoo (CFG, FDG, PAG, SEG, TPG) built on a common GuidanceSignal base.

``build_guidance(row)`` turns a job row into a ``Guidance``. Row keys (all optional):

  guidance            CFG scale w (1 = off); start/end of the CFG interval: cfg_start (default 0),
                      cfg_trunc_ratio (default 1; Lumina's name for the interval end)
  fdg                 {"w_low": .., "w_high": .., "down": 2} — replaces the CFG term by FDG's band-split CFG
  anchor              {"scale": w, "ratio": r, "base": false} — REG (C): CFG with the sub-native band of Δ
                      (and of the base prediction if base) from native-resolution passes (reg.guidance.anchor);
                      ratio defaults to height/1024
  pag / seg / tpg     {"scale": 3.0, "layers": [..], "start": 0, "end": 1, "sigma": 10 (seg), "frac": 1.0 (tpg)}
  cfg_normalization   Lumina's norm rescale after guidance (default True)
  diag                {"downs": [2, 4, 8, 16]} — record per-step band norms and spectra (reg.diag)

Several terms may be combined, e.g. CFG 4 + PAG 3 (the PAG/SEG papers' joint form).
"""

from __future__ import annotations

from .anchor import AnchorCFGTerm
from .bands import LaplacianSplit
from .base import CFGTerm, Guidance, PertTerm, Step, Term
from .perturb import PAG, SEG, TPG

__all__ = ["Guidance", "Step", "Term", "CFGTerm", "PertTerm", "PAG", "SEG", "TPG", "LaplacianSplit", "AnchorCFGTerm", "build_guidance"]


def build_guidance(row: dict) -> Guidance:
    terms: list[Term] = []
    start, end = float(row.get("cfg_start", 0.0)), float(row.get("cfg_trunc_ratio", 1.0))
    if "anchor" in row:  # REG (C): native-anchored CFG; ratio defaults to height / native 1024
        a = row["anchor"]
        ratio = a.get("ratio", int(row.get("height", 1024)) / 1024)
        terms.append(AnchorCFGTerm(float(a["scale"]), ratio, start, end, base=bool(a.get("base", False))))
    elif "fdg" in row:
        f = row["fdg"]
        split = LaplacianSplit(down=f.get("down", 2.0), sigma=f.get("sigma"))
        terms.append(CFGTerm(f["w_high"], start, end, split=split, band_scales=(f["w_low"], f["w_high"])))
    elif float(row.get("guidance", 1.0)) != 1.0:
        terms.append(CFGTerm(float(row["guidance"]), start, end))
    for key, cls in (("pag", PAG), ("seg", SEG), ("tpg", TPG)):
        if key in row:
            p = dict(row[key])
            terms.append(cls(p.pop("scale"), **p))
    g = Guidance(terms, normalization=bool(row.get("cfg_normalization", True)))
    if "diag" in row:  # per-step guidance diagnostics (reg.diag); run_grid saves <id>_diag.npz
        from reg.diag import Recorder

        g.recorder = Recorder(**(row["diag"] or {}))
    return g
