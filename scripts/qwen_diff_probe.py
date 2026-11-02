#!/usr/bin/env python
"""Find where Qwen._sample departs from the stock pipeline: wrap transformer.forward, record the first calls' inputs and
outputs in both modes (1024², 2 steps, same seed) and print per-call max |diff| of every tensor argument and of the output."""
import torch

from reg.models.qwen import Qwen

PROMPT = "A lighthouse on a rocky shore at dusk, long exposure, dramatic sky"
m = Qwen()
m.encode([PROMPT], " ")
m.free_text_encoders()
tr = m.pipe.transformer
orig = tr.forward
log = {"pipe": [], "ours": []}
mode = ["pipe"]


def probe(*args, **kw):
    out = orig(*args, **kw)
    rec = {k: (v.detach().float().clone() if torch.is_tensor(v) else v) for k, v in kw.items()}
    rec["_out"] = out[0].detach().float().clone()
    rec["_args"] = len(args)
    log[mode[0]].append(rec)
    return out


tr.forward = probe
k = dict(height=1024, width=1024, steps=2, guidance_scale=4.0, seed=0, output_type="latent")
p, _ = m.generate([PROMPT], use_pipeline=True, **k)
mode[0] = "ours"
o, _ = m.generate([PROMPT], **k)
tr.forward = orig
print(f"final latents max|diff| {(p - o).abs().max().item():.3e}; calls pipe {len(log['pipe'])} ours {len(log['ours'])}")
for i, (a, b) in enumerate(zip(log["pipe"], log["ours"])):
    print(f"-- call {i}: positional args pipe {a['_args']} ours {b['_args']}")
    for key in sorted(set(a) | set(b)):
        if key.startswith("_a"):
            continue
        va, vb = a.get(key, "MISSING"), b.get(key, "MISSING")
        if torch.is_tensor(va) and torch.is_tensor(vb):
            same = va.shape == vb.shape
            d = (va - vb).abs().max().item() if same else float("nan")
            flag = "" if same and d == 0 else "   <-- DIFF"
            print(f"   {key:28s} {str(tuple(va.shape)):>20s} {str(tuple(vb.shape)):>20s} max|d| {d:.3e}{flag}")
        else:
            flag = "" if va == vb else "   <-- DIFF"
            print(f"   {key:28s} {str(va)[:40]:>20s} {str(vb)[:40]:>20s}{flag}")
