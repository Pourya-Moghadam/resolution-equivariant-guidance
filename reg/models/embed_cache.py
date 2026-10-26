"""On-disk cache of text-encoder outputs, shared by all model loaders.

Each entry is one safetensors file under ``$REG_DATA/text_embeds/<model_name>/``, named
by a hash of the *exact* text handed to the encoder plus any encoder settings. The
cache stores whatever dict of tensors the encoder returns (embeddings, attention mask,
pooled vector, ...), so the same class serves Lumina, FLUX and Qwen-Image.

Workers and the hub keep separate caches; the hub's is rsynced to the workers along
with the weights (SERVER_SETUP.md step 5), and a worker simply recomputes a miss.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

EncodeFn = Callable[[str], dict[str, torch.Tensor]]


def default_root() -> Path:
    data = os.environ.get("REG_DATA")
    if not data:
        raise RuntimeError("REG_DATA is not set; `source env.sh` first")
    return Path(data) / "text_embeds"


class TextEmbedCache:
    def __init__(self, model_name: str, root: str | Path | None = None, **encoder_settings):
        """``encoder_settings`` (e.g. max_sequence_length) become part of every key."""
        self.dir = Path(root) if root is not None else default_root()
        self.dir = self.dir / model_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.settings = dict(sorted(encoder_settings.items()))
        self.hits = 0
        self.misses = 0

    def key(self, text: str) -> str:
        payload = json.dumps({"model": self.model_name, "text": text, **self.settings}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def path(self, text: str) -> Path:
        return self.dir / f"{self.key(text)}.safetensors"

    def get(
        self,
        text: str,
        encode: EncodeFn,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return the cached tensors for ``text``, computing and saving them on a miss.

        ``encode`` is only called on a miss and must return a dict of tensors for this
        single text (no batch dimension handling is done here).
        """
        p = self.path(text)
        if p.exists():
            self.hits += 1
            out = load_file(str(p))
        else:
            self.misses += 1
            out = {k: v.detach().to("cpu").contiguous() for k, v in encode(text).items()}
            tmp = p.with_suffix(".tmp")
            save_file(out, str(tmp), metadata={"text": text, **{k: str(v) for k, v in self.settings.items()}})
            os.replace(tmp, p)  # atomic: a crash mid-write never leaves a readable half file
        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}
        return out

    def get_batch(self, texts: list[str], encode: EncodeFn, device=None) -> dict[str, torch.Tensor]:
        """Stack per-text entries along a new batch dimension (all entries must share shapes)."""
        items = [self.get(t, encode, device) for t in texts]
        return {k: torch.stack([it[k] for it in items]) for k in items[0]}
