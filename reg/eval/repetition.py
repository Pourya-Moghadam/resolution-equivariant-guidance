"""Repetition count: how many instances of the prompt's subject appear in the image.

Open-vocabulary detection with GroundingDINO (transformers). For a single-subject prompt the
expected count is 1; counts above 1 measure the duplication that resolution extrapolation
causes (see 06/07). Images are evaluated at a bounded working resolution (the detector's
own transform resizes to ~800 px anyway), so 4K inputs are downscaled first.
"""

from __future__ import annotations

import torch
from PIL import Image

MODEL_ID = "IDEA-Research/grounding-dino-base"


class RepetitionCounter:
    def __init__(self, device: str = "cuda", box_threshold: float = 0.35, text_threshold: float = 0.25, max_side: int = 1600):
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID).to(self.device).eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_side = max_side

    @torch.inference_mode()
    def count(self, img: Image.Image, subject: str) -> dict:
        """Return {"count", "scores", "boxes"} for the phrase ``subject`` (boxes in the resized frame)."""
        w, h = img.size
        if max(w, h) > self.max_side:
            s = self.max_side / max(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
        text = subject.strip().lower().rstrip(".") + "."  # GroundingDINO wants lower-case phrases ending in '.'
        inputs = self.processor(images=img, text=text, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[img.size[::-1]],
        )[0]
        scores = [float(s) for s in res["scores"]]
        return {"count": len(scores), "scores": scores, "boxes": [[float(v) for v in b] for b in res["boxes"]]}
