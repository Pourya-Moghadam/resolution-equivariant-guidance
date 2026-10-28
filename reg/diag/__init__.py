"""Per-step diagnostics of the guidance signal (PLAN Phase 2 step 1).

A ``Recorder`` attached to a ``Guidance`` (``guidance.recorder``) is called from ``combine`` at every
step with the raw predictions and the guided output. It records, per step:

  sigma, frac                    noise level and trajectory fraction
  rms_cond / rms_uncond /        root-mean-square of v_cond, v_uncond, Δ = v_cond − v_uncond and the
  rms_delta / rms_out            guided v (after Lumina's normalization), over all latent entries
  low_rms_{delta,cond,uncond}    the same restricted to the low band of a LaplacianSplit at each cutoff
  [n_steps, n_downs]             in ``downs`` (decimation factors on the latent grid; 2 is FDG's native
                                 cutoff, 2·r is the same cutoff in cycles per image at ratio r)
  cos_low [n_steps, n_downs]     cosine between low(Δ) and low(v_cond)
  spec_{delta,cond} [n_steps, n_bins]
                                 radial power spectrum summed over channels, binned in octaves of
                                 cycles per image (bin 0 = DC, bin k = [2^(k-1), 2^k) cycles/image), so
                                 spectra from different resolutions line up by image content

Row key ``"diag": {"downs": [2, 4, 8, 16]}`` turns it on; run_grid saves ``<out>/<id>_diag.npz``.
The key quantity for REG is ``low_rms_delta / low_rms_cond`` at matched sigma across resolutions: if
the coarse-band guidance signal shrinks relative to the base prediction under extrapolation, the
native-matched weight is w_native × (ratio_native / ratio_r), with nothing to tune.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from reg.guidance.bands import LaplacianSplit

N_BINS = 12  # DC + octaves up to 2^11 = 2048 cycles/image (latent side 4096/8 = 512 → max 362)


def _rms(x: torch.Tensor) -> float:
    return x.float().pow(2).mean().sqrt().item()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


class Recorder:
    def __init__(self, downs=(2, 4, 8, 16)):
        self.downs = [float(d) for d in downs]
        self.splits = [LaplacianSplit(down=d) for d in self.downs]
        self.rows: list[dict] = []
        self._bins: dict[tuple, torch.Tensor] = {}
        self.grid: tuple[int, int] | None = None

    def _bin_index(self, h: int, w: int, device) -> torch.Tensor:
        key = (h, w, str(device))
        if key not in self._bins:
            ky = torch.fft.fftfreq(h, device=device) * h  # cycles per image along y
            kx = torch.fft.rfftfreq(w, device=device) * w
            r = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
            idx = torch.where(r < 1, torch.zeros_like(r), torch.floor(torch.log2(r.clamp(min=1))) + 1)
            self._bins[key] = idx.long().clamp(max=N_BINS - 1).flatten()
        return self._bins[key]

    def spectrum(self, x: torch.Tensor) -> np.ndarray:
        h, w = x.shape[-2:]
        power = torch.fft.rfft2(x.float()).abs().pow(2).sum(dim=(0, 1)).flatten()
        spec = torch.bincount(self._bin_index(h, w, x.device), weights=power, minlength=N_BINS)
        return spec.cpu().numpy()

    @torch.no_grad()
    def __call__(self, preds: dict[str, torch.Tensor], step, out: torch.Tensor) -> None:
        cond = preds["cond"]
        self.grid = tuple(cond.shape[-2:])
        row = {"sigma": step.sigma, "frac": step.frac, "rms_cond": _rms(cond), "rms_out": _rms(out)}
        nan = float("nan")
        if "uncond" in preds:
            uncond = preds["uncond"]
            delta = cond - uncond
            row.update(rms_uncond=_rms(uncond), rms_delta=_rms(delta))
            lows = [(s.lowpass(delta), s.lowpass(cond), s.lowpass(uncond)) for s in self.splits]
            row["low_rms_delta"] = [_rms(d) for d, _, _ in lows]
            row["low_rms_cond"] = [_rms(c) for _, c, _ in lows]
            row["low_rms_uncond"] = [_rms(u) for _, _, u in lows]
            row["cos_low"] = [_cos(d, c) for d, c, _ in lows]
            row["spec_delta"] = self.spectrum(delta)
        else:
            row.update(rms_uncond=nan, rms_delta=nan, low_rms_delta=[nan] * len(self.downs),
                       low_rms_cond=[nan] * len(self.downs), low_rms_uncond=[nan] * len(self.downs),
                       cos_low=[nan] * len(self.downs), spec_delta=np.full(N_BINS, nan))
        row["spec_cond"] = self.spectrum(cond)
        self.rows.append(row)

    def arrays(self) -> dict[str, np.ndarray]:
        keys = ["sigma", "frac", "rms_cond", "rms_uncond", "rms_delta", "rms_out", "low_rms_delta", "low_rms_cond",
                "low_rms_uncond", "cos_low", "spec_delta", "spec_cond"]
        out = {k: np.array([r[k] for r in self.rows], dtype=np.float64) for k in keys}
        out["downs"] = np.array(self.downs)
        out["grid"] = np.array(self.grid if self.grid else (0, 0))
        out["bin_edges_cycles"] = np.array([0] + [2.0**k for k in range(N_BINS)])
        return out

    def save(self, path) -> None:
        np.savez(path, **self.arrays())


def summarize(npz_path) -> dict:
    """Trajectory means of the quantities a status line needs."""
    d = np.load(npz_path)
    ratio = d["low_rms_delta"] / d["low_rms_cond"]
    return {"grid": tuple(int(x) for x in d["grid"]), "downs": d["downs"].tolist(),
            "mean_ratio_global": float(np.nanmean(d["rms_delta"] / d["rms_cond"])),
            "mean_ratio_low": np.nanmean(ratio, axis=0).tolist(),
            "mean_cos_low": np.nanmean(d["cos_low"], axis=0).tolist()}
