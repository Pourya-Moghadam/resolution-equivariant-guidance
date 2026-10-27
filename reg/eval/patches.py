"""Patch sampling for patch-FID/KID at high resolution.

Standard FID on a few hundred 4K images is meaningless; instead every image contributes
``n`` crops of ``size`` pixels (default 16 crops of 1024²), drawn on a fixed grid plus
jittered positions from a seeded RNG, so generated and reference sets are sampled the same
way. The crops are what the Inception feature extractor sees (it resizes to 299 internally).
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from PIL import Image


def grid_positions(width: int, height: int, size: int, n: int, seed: int = 0) -> list[tuple[int, int]]:
    """Top-left corners of ``n`` ``size``² crops: a regular grid first, jittered extras after.

    Images smaller than ``size`` on a side yield one crop of the whole image (caller resizes).
    """
    if width <= size and height <= size:
        return [(0, 0)]
    cols = max(1, width // size)
    rows = max(1, height // size)
    # spread the grid so the first and last crops touch the borders
    xs = [round(i * (width - size) / max(1, cols - 1)) for i in range(cols)] if cols > 1 else [(width - size) // 2]
    ys = [round(j * (height - size) / max(1, rows - 1)) for j in range(rows)] if rows > 1 else [(height - size) // 2]
    grid = [(x, y) for y in ys for x in xs]
    rng = random.Random(seed)
    rng.shuffle(grid)
    out = grid[:n]
    while len(out) < n:  # more crops wanted than grid cells: add jittered random positions
        out.append((rng.randint(0, max(0, width - size)), rng.randint(0, max(0, height - size))))
    return out


def iter_patches(img: Image.Image, size: int = 1024, n: int = 16, seed: int = 0) -> Iterator[Image.Image]:
    w, h = img.size
    for x, y in grid_positions(w, h, size, n, seed):
        crop = img.crop((x, y, min(x + size, w), min(y + size, h)))
        if crop.size != (size, size):  # small image or border crop: letterbox-free resize to the patch size
            crop = crop.resize((size, size), Image.BICUBIC)
        yield crop
