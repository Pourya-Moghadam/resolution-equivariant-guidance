"""CPU tests for the guidance zoo: algebra of the terms, band split, perturbation mechanics."""

import math

import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_lumina2 import Lumina2AttnProcessor2_0

from reg.guidance import PAG, SEG, TPG, CFGTerm, Guidance, LaplacianSplit, Step, build_guidance
from reg.guidance.perturb import PerturbedAttnProcessor

torch.manual_seed(0)


def preds(shape=(1, 4, 8, 8)):
    return {k: torch.randn(shape) for k in ("cond", "uncond", "pert")}


def test_cfg_matches_pipeline_formula():
    p = preds()
    g = Guidance([CFGTerm(4.0)], normalization=True)
    v = g.combine(p, Step(0, 30, 1.0))
    ref = p["uncond"] + 4.0 * (p["cond"] - p["uncond"])
    ref = ref * (torch.norm(p["cond"], dim=-1, keepdim=True) / torch.norm(ref, dim=-1, keepdim=True))
    assert torch.equal(v, ref)
    assert g.passes(Step(0, 30, 1.0)) == ["cond", "uncond"]


def test_cfg_interval_and_truncation():
    p = preds()
    g = Guidance([CFGTerm(4.0, start=0.2, end=0.7)], normalization=False)
    assert g.passes(Step(0, 10, 1.0)) == ["cond"]  # frac 0.1 <= start: off (Kynkäänniemi-style late start)
    assert torch.equal(g.combine(p, Step(0, 10, 1.0)), p["cond"])
    assert not torch.equal(g.combine(p, Step(2, 10, 0.7)), p["cond"])  # frac 0.3: active
    assert g.passes(Step(2, 10, 0.7)) == ["cond", "uncond"]
    assert g.passes(Step(7, 10, 0.3)) == ["cond"]  # frac 0.8 > end: truncated, as cfg_trunc_ratio=0.7
    assert torch.equal(g.combine(p, Step(7, 10, 0.3)), p["cond"])
    # guidance_scale 1 is "no CFG"
    assert Guidance([CFGTerm(1.0)]).passes(Step(0, 10, 1.0)) == ["cond"]


def test_laplacian_split_reconstructs_and_separates():
    x = torch.randn(2, 4, 32, 32)
    s = LaplacianSplit(down=2)
    low, high = s.split(x)
    assert torch.allclose(low + high, x, atol=1e-6)
    # a smooth input is almost entirely low band; white noise mostly high band
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 32), torch.linspace(0, 1, 32), indexing="ij")
    smooth = torch.sin(2 * math.pi * xx)[None, None].expand(1, 4, 32, 32).contiguous()
    lo, hi = s.split(smooth)
    assert hi.norm() < 0.1 * lo.norm()
    assert high.norm() > low.norm()
    # larger `down` keeps less in the low band
    assert LaplacianSplit(down=8).split(x)[0].norm() < low.norm()


def test_fdg_equal_scales_is_cfg_and_unequal_is_not():
    p = preds((1, 4, 16, 16))
    cfg = Guidance([CFGTerm(4.0)], normalization=False).combine(p, Step(0, 1, 1.0))
    fdg = build_guidance({"fdg": {"w_low": 4.0, "w_high": 4.0}, "cfg_normalization": False}).combine(p, Step(0, 1, 1.0))
    assert torch.allclose(cfg, fdg, atol=1e-5)
    fdg2 = build_guidance({"fdg": {"w_low": 2.0, "w_high": 6.0}, "cfg_normalization": False}).combine(p, Step(0, 1, 1.0))
    assert not torch.allclose(cfg, fdg2, atol=1e-3)
    # low band of the FDG update uses w_low, high band w_high
    split = LaplacianSplit(2.0)
    d = p["cond"] - p["uncond"]
    lo, hi = split.split(d)
    assert torch.allclose(fdg2, p["uncond"] + 2.0 * lo + 6.0 * hi, atol=1e-5)


def test_pert_term_combination_with_cfg():
    p = preds()
    g = build_guidance({"guidance": 4.0, "pag": {"scale": 3.0}, "cfg_normalization": False})
    st = Step(0, 10, 1.0)
    assert g.passes(st) == ["cond", "uncond", "pert"]
    ref = p["uncond"] + 4.0 * (p["cond"] - p["uncond"]) + 3.0 * (p["cond"] - p["pert"])
    assert torch.allclose(g.combine(p, st), ref)
    g2 = build_guidance({"guidance": 1.0, "seg": {"scale": 3.0, "sigma": 10}, "cfg_normalization": False})
    assert g2.passes(st) == ["cond", "pert"]
    assert torch.allclose(g2.combine(p, st), p["cond"] + 3.0 * (p["cond"] - p["pert"]))
    assert isinstance(g2.terms[0], SEG) and g2.terms[0].sigma == 10


def _attn():
    a = Attention(query_dim=16, cross_attention_dim=None, dim_head=8, qk_norm="rms_norm", heads=2, kv_heads=1, eps=1e-5,
                  bias=False, out_bias=False, processor=Lumina2AttnProcessor2_0())
    return a.eval()


def test_pag_identity_processor_is_value_passthrough():
    a = _attn()
    x = torch.randn(1, 3 + 16, 16)  # 3 text tokens + 4x4 image grid
    a.set_processor(PerturbedAttnProcessor("identity", text_len=3, grid=(4, 4)))
    out = a(hidden_states=x, encoder_hidden_states=x)
    v = a.to_v(x)  # kv_heads=1 → repeated over the 2 heads
    ref = a.to_out[0](torch.cat([v, v], dim=-1))
    assert torch.allclose(out, ref, atol=1e-6)


def test_seg_processor_with_zero_sigma_equals_original():
    a = _attn()
    x = torch.randn(1, 3 + 16, 16)
    ref = a(hidden_states=x, encoder_hidden_states=x)
    a.set_processor(PerturbedAttnProcessor("blur", sigma=0.0, text_len=3, grid=(4, 4)))
    out = a(hidden_states=x, encoder_hidden_states=x)
    assert torch.allclose(out, ref, atol=1e-5)
    a.set_processor(PerturbedAttnProcessor("blur", sigma=math.inf, text_len=3, grid=(4, 4)))
    out_inf = a(hidden_states=x, encoder_hidden_states=x)
    assert out_inf.shape == ref.shape and not torch.allclose(out_inf, ref, atol=1e-3)
    # with the mean query every image token attends identically → identical image outputs
    assert torch.allclose(out_inf[0, 3:].std(0), torch.zeros(16), atol=1e-5)


class _Tok(torch.nn.Module):  # per-token layer: shuffling then un-shuffling must be a no-op
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, hidden_states, attention_mask=None, rotary=None, temb=None):
        return self.lin(hidden_states)


class _Mix(_Tok):  # mixes tokens along the sequence: shuffling changes the result
    def forward(self, hidden_states, attention_mask=None, rotary=None, temb=None):
        return torch.cumsum(self.lin(hidden_states), dim=1)


class _Tr(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layers = torch.nn.ModuleList([layer() for _ in range(4)])

    def forward(self, x):
        for l in self.layers:
            x = l(x, None, None, None)
        return x


def test_tpg_hooks_unshuffle_exactly():
    x = torch.randn(1, 3 + 16, 8)
    ctx = {"text_len": 3, "grid": (4, 4), "generator": torch.Generator("cpu").manual_seed(0)}
    tr = _Tr(_Tok)
    ref = tr(x)
    with TPG(3.0, layers=[1, 2]).perturbation(tr, ctx):
        out = tr(x)
    assert torch.allclose(out, ref, atol=1e-6)
    assert torch.allclose(tr(x), ref)  # hooks removed
    assert len(tr.layers[1]._forward_pre_hooks) == 0 and len(tr.layers[1]._forward_hooks) == 0
    tr2 = _Tr(_Mix)
    ref2 = tr2(x)
    with TPG(3.0, layers=[1, 2]).perturbation(tr2, ctx):
        out2 = tr2(x)
    assert not torch.allclose(out2, ref2, atol=1e-3)
    assert torch.allclose(out2[:, :3], ref2[:, :3])  # text tokens untouched (cumsum prefix identical)


def test_attn_pert_context_swaps_and_restores_processors():
    class _Blk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _attn()

    class _Tr(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([_Blk() for _ in range(8)])

    tr = _Tr()
    orig = [b.attn.processor for b in tr.layers]
    with PAG(3.0).perturbation(tr, {"text_len": 3, "grid": (4, 4)}):
        swapped = [isinstance(b.attn.processor, PerturbedAttnProcessor) for b in tr.layers]
    assert swapped == [False, False, False, True, True, False, False, False]  # middle quarter of 8 → [3, 5)
    assert [b.attn.processor for b in tr.layers] == orig


def test_build_guidance_describe_roundtrip():
    g = build_guidance({"guidance": 6.0, "cfg_trunc_ratio": 0.7, "tpg": {"scale": 2.0, "layers": [10, 11], "frac": 0.5}})
    d = g.describe()
    assert d["terms"][0] == {"name": "cfg", "scale": 6.0, "start": 0.0, "end": 0.7}
    assert d["terms"][1]["name"] == "tpg" and d["terms"][1]["frac"] == 0.5 and d["terms"][1]["layers"] == [10, 11]
    assert build_guidance({}).passes(Step(0, 1, 1.0)) == ["cond"]
