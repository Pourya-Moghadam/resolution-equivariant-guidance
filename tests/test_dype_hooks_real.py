"""python -m unittest tests/test_dype_hooks_real.py   (CPU; tiny real transformer classes, no weights)

Checks that apply_dype() really receives the per-step timestep inside the real diffusers
transformer forward passes (Lumina2 keyword call with 1 - t/1000, FLUX with t), and that the
dynamic variant produces different outputs from the static one at an extrapolated extent.
"""

import unittest

import torch
from diffusers import FluxTransformer2DModel, Lumina2Transformer2DModel

from reg.extrap.dype import apply_dype, remove_dype

torch.manual_seed(0)


class TestRealHooks(unittest.TestCase):
    def test_lumina_hook_sees_inverted_timestep_and_dynamics_matter(self):
        m = Lumina2Transformer2DModel(
            sample_size=16, patch_size=2, in_channels=4, hidden_size=64, num_layers=1, num_refiner_layers=1,
            num_attention_heads=2, num_kv_heads=2, multiple_of=32, cap_feat_dim=16, axes_dim_rope=(16, 8, 8),
            axes_lens=(300, 512, 512),
        ).eval()
        # 2K-equivalent extent: 128 x 128 patches (> base 64) but tiny channels; keep it small: 72 x 72 patches
        x = torch.randn(1, 4, 144, 144)
        cap = torch.randn(1, 5, 16)
        mask = torch.ones(1, 5, dtype=torch.long)
        with torch.no_grad():
            pe = apply_dype(m, method="yarn", dype=True)
            out_t_early = m(x, torch.tensor([0.2]), cap, mask, return_dict=False)[0]  # Lumina: 0.2 => noise level 0.8
            self.assertAlmostEqual(pe.t, 0.8)
            out_t_late = m(x, torch.tensor([0.95]), cap, mask, return_dict=False)[0]  # noise level 0.05
            self.assertAlmostEqual(pe.t, 0.05)
            self.assertFalse(torch.allclose(out_t_early, out_t_late))
            pe_s = apply_dype(m, method="yarn", dype=False)
            out_static = m(x, torch.tensor([0.95]), cap, mask, return_dict=False)[0]
            self.assertEqual(pe_s.kappa, 1.0)
            # at the same timestep, dynamic (kappa=0.0025) and static (kappa=1) must differ
            self.assertFalse(torch.allclose(out_t_late, out_static, atol=1e-5))
            remove_dype(m)
            out_plain = m(x, torch.tensor([0.95]), cap, mask, return_dict=False)[0]
            self.assertFalse(torch.allclose(out_plain, out_static, atol=1e-5))

    def test_flux_hook_sees_timestep(self):
        m = FluxTransformer2DModel(
            patch_size=1, in_channels=8, num_layers=1, num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
            joint_attention_dim=16, pooled_projection_dim=8, guidance_embeds=False, axes_dims_rope=(4, 6, 6),
        ).eval()
        hp = wp = 72  # > 64 base patches
        img_ids = torch.zeros(hp * wp, 3)
        img_ids[:, 1] = torch.arange(hp).repeat_interleave(wp)
        img_ids[:, 2] = torch.arange(wp).repeat(hp)
        txt_ids = torch.zeros(3, 3)
        x = torch.randn(1, hp * wp, 8)
        enc = torch.randn(1, 3, 16)
        pooled = torch.randn(1, 8)
        with torch.no_grad():
            pe = apply_dype(m, method="yarn", dype=True)
            a = m(hidden_states=x, encoder_hidden_states=enc, pooled_projections=pooled, timestep=torch.tensor([0.9]),
                  img_ids=img_ids, txt_ids=txt_ids, return_dict=False)[0]
            self.assertAlmostEqual(pe.t, 0.9)
            b = m(hidden_states=x, encoder_hidden_states=enc, pooled_projections=pooled, timestep=torch.tensor([0.05]),
                  img_ids=img_ids, txt_ids=txt_ids, return_dict=False)[0]
            self.assertAlmostEqual(pe.t, 0.05)
            self.assertFalse(torch.allclose(a, b))
            remove_dype(m)


if __name__ == "__main__":
    unittest.main()
