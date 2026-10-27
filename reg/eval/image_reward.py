"""ImageReward (Xu et al. 2023): learned human-preference score for text-image pairs.

The ``image-reward`` package imports OpenAI's ``clip`` at package import for its aesthetic
scorer, which we do not use and cannot install here (external git package); a stub module
satisfies the import. Weights (ImageReward-v1.0, ~2 GB incl. BLIP) are cached under
``$REG_DATA/weights/imagereward``.
"""

from __future__ import annotations

import os
import sys
import types

import torch
from PIL import Image


class ImageRewardScorer:
    def __init__(self, device: str = "cuda"):
        os.environ.setdefault("TORCH_HOME", os.path.join(os.environ.get("REG_DATA", "."), "weights", "torch"))
        sys.modules.setdefault("clip", types.ModuleType("clip"))
        import ImageReward as RM

        self.device = torch.device(device)
        root = os.path.join(os.environ.get("REG_DATA", "."), "weights", "imagereward")
        self.model = RM.load("ImageReward-v1.0", device=str(self.device), download_root=root)

    @torch.inference_mode()
    def score(self, image: Image.Image, prompt: str) -> float:
        return float(self.model.score(prompt, image.convert("RGB")))
