"""Patch-FID and patch-KID on Inception-v3 pool features (torchmetrics / torch-fidelity).

Both metrics compare a *set* of generated patches with a *set* of reference patches (see
``reg.eval.patches``). Reference features depend only on the reference set and patch
sampling, so they are computed once and cached to ``$REG_DATA/results/ref_feats/``.

KID (unbiased MMD with a polynomial kernel) is reported alongside FID because FID is biased
for a few thousand samples and KID is not.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from PIL import Image

from .patches import iter_patches


def _to_uint8(img: Image.Image, side: int = 299) -> torch.Tensor:
    im = img.convert("RGB").resize((side, side), Image.BICUBIC)
    return torch.from_numpy(__import__("numpy").asarray(im)).permute(2, 0, 1).contiguous()


class InceptionFeatures:
    """2048-d pool3 features of Inception-v3 (the FID network), on 299x299 uint8 inputs."""

    def __init__(self, device: str = "cuda"):
        from torchmetrics.image.fid import FrechetInceptionDistance

        self.device = torch.device(device)
        # reuse torchmetrics' wrapped torch-fidelity network; we only need its feature extractor
        self._fid = FrechetInceptionDistance(feature=2048, normalize=False).to(self.device)
        self.net = self._fid.inception.eval()

    @torch.inference_mode()
    def __call__(self, images: list[Image.Image], batch: int = 32) -> torch.Tensor:
        feats = []
        for i in range(0, len(images), batch):
            x = torch.stack([_to_uint8(im) for im in images[i : i + batch]]).to(self.device)
            feats.append(self.net(x).float().cpu())
        return torch.cat(feats)

    def patch_features(self, image: Image.Image, size: int = 1024, n: int = 16, seed: int = 0) -> torch.Tensor:
        return self(list(iter_patches(image, size, n, seed)))


def fid_from_features(gen: torch.Tensor, ref: torch.Tensor) -> float:
    from torchmetrics.image.fid import _compute_fid

    mu1, mu2 = gen.double().mean(0), ref.double().mean(0)
    s1 = torch.cov(gen.double().T)
    s2 = torch.cov(ref.double().T)
    return float(_compute_fid(mu1, s1, mu2, s2))


def kid_from_features(gen: torch.Tensor, ref: torch.Tensor, subsets: int = 50, subset_size: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Unbiased KID (mean, std over subsets) with the standard cubic polynomial kernel."""
    from torchmetrics.image.kid import poly_mmd

    g = torch.Generator().manual_seed(seed)
    n = min(subset_size, len(gen), len(ref))
    vals = []
    for _ in range(subsets):
        gi = torch.randperm(len(gen), generator=g)[:n]
        ri = torch.randperm(len(ref), generator=g)[:n]
        vals.append(poly_mmd(gen[gi].double(), ref[ri].double(), degree=3, gamma=None, coef=1.0))
    v = torch.tensor(vals)
    return float(v.mean()), float(v.std())


def reference_features(extractor: InceptionFeatures, paths: list[str | Path], size: int, n: int, seed: int = 0,
                       cache_dir: str | Path | None = None) -> torch.Tensor:
    """Features for a reference image set, cached by (paths, size, n, seed)."""
    cache_dir = Path(cache_dir or os.path.join(os.environ.get("REG_DATA", "."), "results", "ref_feats"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(("\n".join(map(str, sorted(map(str, paths)))) + f"|{size}|{n}|{seed}").encode()).hexdigest()[:16]
    f = cache_dir / f"inception2048_{len(paths)}imgs_{size}px_{n}crops_{key}.pt"
    if f.exists():
        return torch.load(f)
    feats = torch.cat([extractor.patch_features(Image.open(p), size, n, seed) for p in paths])
    torch.save(feats, f)
    return feats
