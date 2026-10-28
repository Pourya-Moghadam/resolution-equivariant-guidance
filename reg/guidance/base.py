"""Guidance as a sum of signals on top of the conditional prediction.

Every guidance method we study has the form

    v = v_base + Σ_terms Σ_bands  g_term,band(step) · ψ_band(Δ_term)

with Δ = v_cond − v_uncond (CFG, FDG, and the native-anchored CFG of ``reg.guidance.anchor``) or
Δ = v_cond − v_perturbed (PAG, SEG, TPG), ψ an
optional band split (``reg.guidance.bands``) and g a scalar gain that may depend on the step
(scale, interval; later REG's per-band schedule). ``v_base`` is v_uncond while a CFG term is
active (so plain CFG is exactly v_u + w·Δ, bit-identical to the Lumina pipeline) and v_cond
otherwise.

A ``Guidance`` tells the sampler which forward passes a step needs (``passes``), how to
perturb the network for the ``pert`` pass (``perturbation``) and how to combine the predictions
(``combine``). The sampler lives in the model wrappers (``reg.models.lumina.Lumina.generate``).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class Step:
    i: int  # 0-based step index
    n: int  # number of steps
    sigma: float  # flow-matching noise level in (0, 1]; 1 = pure noise

    @property
    def frac(self) -> float:
        """Fraction of the trajectory completed after this step, as Lumina's cfg_trunc_ratio counts it."""
        return (self.i + 1) / self.n


class Term:
    """One guidance signal. Subclasses set ``kind`` ('cfg' or 'pert') and may override ``gains``."""

    kind: str = "cfg"
    name: str = "term"

    def __init__(self, scale: float, start: float = 0.0, end: float = 1.0, split=None, band_scales: tuple[float, ...] | None = None):
        """``scale``: guidance scale w (CFG convention: w = 1 is off) or perturbation scale s (s = 0 is off).
        Active while ``start < step.frac <= end`` (``end`` is Lumina's cfg_trunc_ratio).
        ``split``: a band split applied to Δ; ``band_scales`` gives one scale per band (FDG's w_low, w_high)."""
        self.scale = float(scale)
        self.start, self.end = float(start), float(end)
        self.split = split
        if split is not None:
            if band_scales is None or len(band_scales) != len(split.names):
                raise ValueError(f"need one scale per band {split.names}")
            self.band_scales = tuple(float(s) for s in band_scales)
        else:
            self.band_scales = (self.scale,)

    def active(self, step: Step) -> bool:
        return self.start < step.frac <= self.end and any(self._gain(s) != 0 for s in self.band_scales)

    def _gain(self, scale: float) -> float:
        return scale  # for 'cfg' terms the sampler supplies v_u as base, so the gain is w itself

    def gains(self, step: Step) -> tuple[float, ...]:
        """Per-band gains at this step; REG's schedules override this."""
        return tuple(self._gain(s) for s in self.band_scales)

    def delta(self, preds: dict[str, torch.Tensor]) -> torch.Tensor:
        return preds["cond"] - preds["uncond" if self.kind == "cfg" else "pert"]

    def contribution(self, preds: dict[str, torch.Tensor], step: Step) -> torch.Tensor:
        d = self.delta(preds)
        g = self.gains(step)
        if self.split is None:
            return g[0] * d
        bands = self.split.split(d)
        return sum(gi * b for gi, b in zip(g, bands))

    def perturbation(self, transformer, ctx: dict):
        """Context manager that perturbs ``transformer`` for the 'pert' pass. ``ctx`` carries
        text_len, grid (latent patches h, w) and a torch.Generator for reproducible perturbations."""
        return contextlib.nullcontext()

    def describe(self) -> dict:
        d = {"name": self.name, "scale": self.scale, "start": self.start, "end": self.end}
        if self.split is not None:
            d["split"] = repr(self.split)
            d["band_scales"] = self.band_scales
        return d


class CFGTerm(Term):
    kind, name = "cfg", "cfg"

    def active(self, step: Step) -> bool:
        return self.start < step.frac <= self.end and any(s != 1.0 for s in self.band_scales)


class PertTerm(Term):
    kind, name = "pert", "pert"


@dataclass
class Guidance:
    terms: list[Term] = field(default_factory=list)
    normalization: bool = True  # Lumina's cfg_normalization: rescale the guided v to |v_cond| (over the last axis)
    recorder: object = None  # optional reg.diag.Recorder, called with (preds, step, v) at every step

    def active_terms(self, step: Step) -> list[Term]:
        return [t for t in self.terms if t.active(step)]

    def passes(self, step: Step) -> list[str]:
        p = ["cond"]
        kinds = {t.kind for t in self.active_terms(step)}
        if kinds & {"cfg", "anchor"}:
            p.append("uncond")
        if "pert" in kinds:
            p.append("pert")
        if "anchor" in kinds:  # native-resolution passes (reg.guidance.anchor); the sampler fills anchor_cond/anchor_uncond
            p.append("anchor")
        return p

    def anchor_terms(self, step: Step) -> list[Term]:
        return [t for t in self.active_terms(step) if t.kind == "anchor"]

    def pert_terms(self, step: Step) -> list[Term]:
        return [t for t in self.active_terms(step) if t.kind == "pert"]

    def combine(self, preds: dict[str, torch.Tensor], step: Step) -> torch.Tensor:
        v = self._combine(preds, step)
        if self.recorder is not None:
            self.recorder(preds, step, v)
        return v

    def _combine(self, preds: dict[str, torch.Tensor], step: Step) -> torch.Tensor:
        active = self.active_terms(step)
        if not active:
            return preds["cond"]
        cfg_active = any(t.kind in ("cfg", "anchor") for t in active)
        v = preds["uncond"] if cfg_active else preds["cond"]
        for t in active:
            v = v + t.contribution(preds, step)
        if self.normalization:
            cond_norm = torch.norm(preds["cond"], dim=-1, keepdim=True)
            v = v * (cond_norm / torch.norm(v, dim=-1, keepdim=True))
        return v

    def describe(self) -> dict:
        return {"terms": [t.describe() for t in self.terms], "normalization": self.normalization}
