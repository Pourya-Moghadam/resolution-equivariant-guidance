"""python -m unittest tests/test_dype_lumina.py   (CPU only, no weights)"""

import unittest

import torch
from diffusers.models.transformers.transformer_lumina2 import Lumina2RotaryPosEmbed

from reg.extrap.dype import DyPELuminaRope, apply_dype, remove_dype

AXES_DIM, AXES_LENS, THETA = (32, 32, 32), (300, 512, 512), 10000  # Lumina-Image 2.0 config


def lumina_ids(patches: int, cap_len: int = 8) -> torch.Tensor:
    """Position ids exactly as Lumina2RotaryPosEmbed.forward builds them (batch of 1)."""
    s = cap_len + patches * patches
    ids = torch.zeros(1, s, 3, dtype=torch.int32)
    ids[0, :cap_len, 0] = torch.arange(cap_len)
    ids[0, cap_len:, 0] = cap_len
    ids[0, cap_len:, 1] = torch.arange(patches).repeat_interleave(patches)
    ids[0, cap_len:, 2] = torch.arange(patches).repeat(patches)
    return ids


class TestDyPELumina(unittest.TestCase):
    def setUp(self):
        self.rope = Lumina2RotaryPosEmbed(THETA, list(AXES_DIM), list(AXES_LENS), patch_size=2)

    def test_native_matches_stock_exactly(self):
        ids = lumina_ids(64)  # 1024 px
        stock = self.rope._get_freqs_cis(ids)
        for method in ("base", "ntk", "yarn"):
            ours = DyPELuminaRope(self.rope, method=method)(ids)
            self.assertEqual(ours.dtype, stock.dtype)
            torch.testing.assert_close(ours, stock, rtol=0, atol=0)

    def test_scaled_axes_change_but_caption_axis_and_magnitude_behave(self):
        ids = lumina_ids(128)  # 2048 px
        stock = self.rope._get_freqs_cis(ids)
        ours = DyPELuminaRope(self.rope, method="yarn", dype=False)(ids)
        torch.testing.assert_close(ours[..., :16], stock[..., :16], rtol=0, atol=0)  # axis 0 untouched
        self.assertFalse(torch.allclose(ours[..., 16:], stock[..., 16:]))
        mscale = 0.1 * torch.log(torch.tensor(2.0)) + 1.0
        torch.testing.assert_close(ours[..., 16:].abs().float(), torch.full_like(ours[..., 16:].abs().float(), float(mscale)))
        self.assertTrue(torch.isfinite(ours.real).all() and torch.isfinite(ours.imag).all())

    def test_hook_inverts_lumina_timestep(self):
        class Stub(torch.nn.Module):
            def __init__(s, rope):
                super().__init__()
                s.rope_embedder = rope

            def forward(s, hidden_states, timestep, encoder_hidden_states=None, encoder_attention_mask=None):
                return s.rope_embedder._get_freqs_cis(hidden_states).abs().sum()

        m = Stub(self.rope)
        pe = apply_dype(m, method="yarn")
        ids = lumina_ids(128)
        m(ids, torch.tensor([0.25, 0.25]))  # Lumina convention: 0.25 = early (noisy) step
        self.assertAlmostEqual(pe.t, 0.75)
        m(ids, timestep=torch.tensor([0.9]))
        self.assertAlmostEqual(pe.t, 0.1)
        remove_dype(m)
        self.assertIs(m.rope_embedder._get_freqs_cis.__func__ if hasattr(m.rope_embedder._get_freqs_cis, "__func__") else None,
                      Lumina2RotaryPosEmbed._get_freqs_cis)
        self.assertFalse(hasattr(m, "_dype_orig"))


if __name__ == "__main__":
    unittest.main()
