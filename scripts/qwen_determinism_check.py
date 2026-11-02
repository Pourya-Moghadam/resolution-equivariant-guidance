#!/usr/bin/env python
"""Is Qwen-Image sampling reproducible run to run (fp8 torchao weights, SDPA)? Runs the stock pipeline twice and our loop twice
at 1024², 4 steps, same seed, and prints the max |diff| of the four pairs. If pipeline-vs-pipeline is nonzero, bit identity with
the pipeline is not a meaningful test and the port check compares against that noise floor instead."""
import torch

from reg.models.qwen import Qwen

PROMPT = "A lighthouse on a rocky shore at dusk, long exposure, dramatic sky"
m = Qwen()
m.encode([PROMPT], " ")
m.free_text_encoders()
k = dict(height=1024, width=1024, steps=4, guidance_scale=4.0, seed=0, output_type="latent")
p1, _ = m.generate([PROMPT], use_pipeline=True, **k)
p2, _ = m.generate([PROMPT], use_pipeline=True, **k)
o1, _ = m.generate([PROMPT], **k)
o2, _ = m.generate([PROMPT], **k)
d = lambda a, b: (a - b).abs().max().item()  # noqa: E731
print(f"pipeline vs pipeline: {d(p1, p2):.3e}   ours vs ours: {d(o1, o2):.3e}   pipeline vs ours: {d(p1, o1):.3e} / {d(p2, o2):.3e}   |latent| max {p1.abs().max().item():.2f}")
