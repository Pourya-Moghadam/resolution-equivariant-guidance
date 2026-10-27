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


def _shim_transformers() -> None:
    """image-reward 1.5 vendors a BLIP written against transformers 4.x; restore the three
    helpers it imports from ``transformers.modeling_utils`` that 5.x moved or removed."""
    import transformers.modeling_utils as mu
    import transformers.pytorch_utils as pu

    for name in ("apply_chunking_to_forward", "prune_linear_layer"):
        if not hasattr(mu, name):
            setattr(mu, name, getattr(pu, name))
    if not hasattr(mu, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            # transformers 4.x implementation; only used when pruning attention heads (never at inference)
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        mu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    from transformers import BertTokenizer

    if not hasattr(BertTokenizer, "additional_special_tokens_ids"):  # removed from tokenizers in 5.x; BLIP reads it
        BertTokenizer.additional_special_tokens_ids = property(lambda self: self.convert_tokens_to_ids(self.additional_special_tokens))


class ImageRewardScorer:
    def __init__(self, device: str = "cuda"):
        os.environ.setdefault("TORCH_HOME", os.path.join(os.environ.get("REG_DATA", "."), "weights", "torch"))
        sys.modules.setdefault("clip", types.ModuleType("clip"))
        _shim_transformers()
        import ImageReward as RM

        self.device = torch.device(device)
        root = os.path.join(os.environ.get("REG_DATA", "."), "weights", "imagereward")
        self.model = RM.load("ImageReward-v1.0", device=str(self.device), download_root=root)

    @torch.inference_mode()
    def score(self, image: Image.Image, prompt: str) -> float:
        return float(self.model.score(prompt, image.convert("RGB")))
