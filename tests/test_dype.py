"""python -m unittest tests/test_dype.py   (CPU only, no weights)"""

import math
import unittest

import torch
from diffusers.models.transformers.transformer_flux import FluxPosEmbed

from reg.extrap.dype import DyPEFluxPosEmbed, apply_dype, remove_dype, yarn_freqs

THETA, AXES = 10000, [16, 56, 56]  # FLUX.1 configuration


def flux_ids(patches: int, txt_len: int = 8) -> torch.Tensor:
    img = torch.zeros(patches * patches, 3)
    img[:, 1] = torch.arange(patches).repeat_interleave(patches)
    img[:, 2] = torch.arange(patches).repeat(patches)
    return torch.cat([torch.zeros(txt_len, 3), img])


class TestDyPEPosEmbed(unittest.TestCase):
    def test_native_resolution_matches_stock_exactly(self):
        ids = flux_ids(64)  # 1024 px
        stock_cos, stock_sin = FluxPosEmbed(THETA, AXES)(ids)
        for method in ("base", "ntk", "yarn"):
            pe = DyPEFluxPosEmbed(THETA, AXES, method=method, dype=True)
            pe.set_timestep(0.7)
            cos, sin = pe(ids)
            torch.testing.assert_close(cos, stock_cos, rtol=0, atol=0)
            torch.testing.assert_close(sin, stock_sin, rtol=0, atol=0)

    def test_base_method_never_scales(self):
        ids = flux_ids(128)  # 2048 px
        stock = FluxPosEmbed(THETA, AXES)(ids)
        ours = DyPEFluxPosEmbed(THETA, AXES, method="base")(ids)
        torch.testing.assert_close(ours[0], stock[0], rtol=0, atol=0)

    def test_yarn_scaled_axes_differ_and_text_axis_untouched(self):
        ids = flux_ids(128)
        stock_cos, _ = FluxPosEmbed(THETA, AXES)(ids)
        pe = DyPEFluxPosEmbed(THETA, AXES, method="yarn", dype=False)
        cos, _ = pe(ids)
        self.assertEqual(cos.shape, stock_cos.shape)
        torch.testing.assert_close(cos[:, :16], stock_cos[:, :16], rtol=0, atol=0)  # axis 0 (text/time)
        self.assertFalse(torch.allclose(cos[:, 16:], stock_cos[:, 16:]))

    def test_yarn_frequency_bands(self):
        # scale 2, kappa 1. RoPE dim index 0 is the highest frequency (1 rad/patch), the last index the
        # lowest. YaRN keeps high-frequency dims native and linearly interpolates low-frequency dims.
        dim, scale = 56, 2.0
        freqs, mscale = yarn_freqs(dim, THETA, scale, 1.0, 64, torch.float64, "cpu")
        base = 1.0 / (THETA ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        self.assertAlmostEqual(mscale, 0.1 * math.log(2) + 1.0)
        self.assertAlmostEqual((freqs[0] / base[0]).item(), 1.0, places=9)  # highest-freq dim: native
        self.assertAlmostEqual((freqs[-1] / base[-1]).item(), 1.0 / scale, places=9)  # lowest-freq dim: linear
        ratios = freqs / base
        self.assertTrue(bool((ratios[1:] <= ratios[:-1] + 1e-12).all()), "ratio should fall monotonically from 1 to 1/scale")

    def test_dype_relaxes_to_native_as_t_goes_to_zero(self):
        ids = flux_ids(256)  # 4096 px
        stock_cos, _ = FluxPosEmbed(THETA, AXES)(ids)
        pe = DyPEFluxPosEmbed(THETA, AXES, method="yarn", dype=True)
        dists = []
        for t in (1.0, 0.5, 0.1, 0.036):
            pe.set_timestep(t)
            cos, _ = pe(ids)
            dists.append((cos[:, 16:] / pe_mscale(pe, 256) - stock_cos[:, 16:]).abs().mean().item())
        self.assertTrue(all(a > b for a, b in zip(dists, dists[1:])), dists)
        self.assertTrue(torch.isfinite(cos).all())

    def test_apply_and_remove_on_a_stub_transformer(self):
        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pos_embed = FluxPosEmbed(THETA, AXES)
                self.seen = []

            def forward(self, x, timestep=None):
                self.seen.append(timestep)
                return self.pos_embed(x)[0].sum()

        m = Stub()
        orig_pe = m.pos_embed
        pe = apply_dype(m, method="yarn")
        m(flux_ids(128), timestep=torch.tensor([0.25, 0.25]))
        self.assertEqual(pe.t, 0.25)
        self.assertEqual(len(m.seen), 1)
        remove_dype(m)
        self.assertIs(m.pos_embed, orig_pe)
        self.assertFalse(hasattr(m, "_dype_orig"))


def pe_mscale(pe: DyPEFluxPosEmbed, patches: int) -> float:
    return 0.1 * math.log(patches / pe.base_patches) + 1.0


if __name__ == "__main__":
    unittest.main()
