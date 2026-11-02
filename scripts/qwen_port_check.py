#!/usr/bin/env python
"""Qwen-Image port check (needs a free 32 GB card; ~10 min):

  1. plain CFG 4 at 1024², 8 steps: our loop (reg.models.qwen.Qwen._sample) vs the stock pipeline, latent output
     -> reports max |diff| (expected 0: same kernels, same order of operations)
  2. REG anchor at 2048² (ratio 2, base anchor, w4, super 12), 4 steps -> runs, finite, time / peak memory
  3. DyPE NTK at 2048², 2 steps -> runs; at 1024² (scale 64/83 < 1) the DyPE RoPE equals the stock RoPE

  python scripts/qwen_port_check.py [--out DIR]
"""
import argparse
import time

import torch

from reg.extrap.dype import apply_dype, remove_dype
from reg.guidance import build_guidance
from reg.models.qwen import Qwen

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=None)
ap.add_argument("--steps", type=int, default=8)
a = ap.parse_args()
PROMPT = "A lighthouse on a rocky shore at dusk, long exposure, dramatic sky"
m = Qwen()
m.encode([PROMPT], " ")
m.free_text_encoders()

# 3b. RoPE equality below the native extent (no model call needed)
tr = m.pipe.transformer
stock = tr.pos_embed
vid0, txt0 = stock([[(1, 32, 32)]], device=m.device, max_txt_seq_len=20)
pe = apply_dype(tr, method="ntk", dype=True)
vid1, txt1 = tr.pos_embed([[(1, 32, 32)]], device=m.device, max_txt_seq_len=20)
print(f"[rope] 1024 (32x32 tokens... grid 64 patches): dype==stock: {torch.equal(vid0, vid1) and torch.equal(txt0, txt1)}  "
      f"max|d| {(vid0 - vid1).abs().max().item():.2e}")
vid2, _ = tr.pos_embed([[(1, 128, 128)]], device=m.device, max_txt_seq_len=20)
vid2s, _ = stock([[(1, 128, 128)]], device=m.device, max_txt_seq_len=20)
print(f"[rope] 2048 grid: dype != stock: {not torch.equal(vid2, vid2s)}; shapes {tuple(vid2.shape)} {tuple(vid2s.shape)}")
remove_dype(tr)

# 1. bit identity, plain CFG
t0 = time.perf_counter()
ref, _ = m.generate([PROMPT], 1024, 1024, steps=a.steps, guidance_scale=4.0, seed=0, output_type="latent", use_pipeline=True)
t1 = time.perf_counter()
ours, info = m.generate([PROMPT], 1024, 1024, steps=a.steps, guidance_scale=4.0, seed=0, output_type="latent")
t2 = time.perf_counter()
d = (ref - ours).abs().max().item()
print(f"[cfg 1K] pipeline {t1 - t0:.0f}s, ours {t2 - t1:.0f}s; max |diff| {d:.3e}; equal: {torch.equal(ref, ours)}; peak {info['peak_vram_gb']:.1f} GB")

# 2. anchor at 2K
row = {"guidance": 4.0, "height": 2048, "anchor": {"scale": 4.0, "ratio": 2, "base": True, "super_scale": 12.0}}
g = build_guidance(row)
imgs, info = m.generate([PROMPT], 2048, 2048, steps=4, seed=0, guidance=g)
print(f"[reg 2K] {info['seconds']:.0f}s, peak {info['peak_vram_gb']:.1f} GB, guidance {info['guidance']}")
if a.out:
    imgs[0].save(f"{a.out}/qwen_reg_2k.png")

# 3. DyPE at 2K
apply_dype(tr, method="ntk", dype=True)
imgs, info = m.generate([PROMPT], 2048, 2048, steps=2, seed=0, guidance=build_guidance(row))
print(f"[reg+ntk 2K] {info['seconds']:.0f}s, peak {info['peak_vram_gb']:.1f} GB")
remove_dype(tr)
if a.out:
    imgs[0].save(f"{a.out}/qwen_reg_ntk_2k.png")
print("done")
