"""CLIP score: cosine similarity between image and prompt embeddings (OpenCLIP ViT-L/14, laion2B).

Reported as 100 * cos, the convention of torchmetrics' CLIPScore and most T2I papers. Images
above 1K are downscaled to the encoder's 224 px input by the preprocessing transform anyway;
for high-resolution work we additionally report the mean over patch crops (``score_patches``)
so that local text-image agreement at 4K is measured, not just the global thumbnail.
"""

from __future__ import annotations

import os

import torch
from PIL import Image

MODEL = ("ViT-L-14", "laion2b_s32b_b82k")


class ClipScorer:
    def __init__(self, device: str = "cuda"):
        import open_clip

        self.device = torch.device(device)
        cache = os.path.join(os.environ.get("REG_DATA", "."), "weights", "open_clip")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL[0], pretrained=MODEL[1], cache_dir=cache)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL[0])

    @torch.inference_mode()
    def text_features(self, prompts: list[str]) -> torch.Tensor:
        tok = self.tokenizer(prompts).to(self.device)
        f = self.model.encode_text(tok).float()
        return f / f.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def image_features(self, images: list[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        f = self.model.encode_image(x).float()
        return f / f.norm(dim=-1, keepdim=True)

    def score(self, image: Image.Image, prompt: str) -> float:
        return float(100 * (self.image_features([image]) @ self.text_features([prompt]).T).item())

    def score_patches(self, patches: list[Image.Image], prompt: str) -> float:
        t = self.text_features([prompt])
        return float(100 * (self.image_features(patches) @ t.T).mean().item())
