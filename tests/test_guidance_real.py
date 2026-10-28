"""GPU tests with the real Lumina weights (skipped without CUDA):

  python -m pytest -q tests/test_guidance_real.py

1. Our sampling loop with plain CFG is bit-identical to Lumina2Pipeline (with and without truncation).
2. Every guidance method runs end to end and returns finite latents.
"""

import math
import unittest

import torch

from reg.guidance import build_guidance

PROMPT = "a red fox sitting in a snowy forest clearing at dawn, photorealistic"


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU and the Lumina weights")
class TestLuminaGuidanceReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from reg.models.lumina import Lumina

        cls.m = Lumina()

    def _latents(self, **kw):
        kw.setdefault("height", 1024)
        kw.setdefault("width", 1024)
        kw.setdefault("steps", 8)
        out, info = self.m.generate([PROMPT], output_type="latent", **kw)
        return out, info

    def test_own_loop_matches_pipeline_bitwise(self):
        for trunc in (1.0, 0.6):
            with self.subTest(cfg_trunc_ratio=trunc):
                ref, _ = self._latents(guidance_scale=4.0, cfg_trunc_ratio=trunc, use_pipeline=True)
                ours, info = self._latents(guidance_scale=4.0, cfg_trunc_ratio=trunc)
                self.assertTrue(torch.equal(ref, ours), f"max abs diff {(ref - ours).abs().max().item()}")
                self.assertEqual(info["guidance"]["terms"][0]["end"], trunc)

    def test_methods_run(self):
        rows = {
            "fdg": {"fdg": {"w_low": 2.0, "w_high": 6.0}},
            "pag": {"guidance": 4.0, "pag": {"scale": 3.0}},
            "seg": {"guidance": 4.0, "seg": {"scale": 3.0, "sigma": 10.0}},
            "seg_inf": {"guidance": 1.0, "seg": {"scale": 3.0, "sigma": math.inf}},
            "tpg": {"guidance": 4.0, "tpg": {"scale": 3.0, "frac": 0.5}},
            "interval": {"guidance": 6.0, "cfg_start": 0.25, "cfg_trunc_ratio": 0.75},
        }
        base, _ = self._latents(guidance_scale=4.0, steps=4, height=512, width=512)
        for name, row in rows.items():
            with self.subTest(method=name):
                lat, info = self._latents(guidance=build_guidance(row), steps=4, height=512, width=512)
                self.assertTrue(torch.isfinite(lat).all())
                self.assertFalse(torch.equal(lat, base), "guidance had no effect")
                self.assertEqual(lat.shape, base.shape)

    def test_tpg_is_reproducible(self):
        g = build_guidance({"guidance": 4.0, "tpg": {"scale": 3.0}})
        a, _ = self._latents(guidance=g, steps=3, height=512, width=512)
        b, _ = self._latents(guidance=g, steps=3, height=512, width=512)
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU and the Lumina weights")
class TestAnchorReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from reg.models.lumina import Lumina

        cls.m = Lumina()

    def test_anchor_ratio_one_matches_pipeline_bitwise(self):
        ref, _ = self.m.generate([PROMPT], height=1024, width=1024, steps=6, guidance_scale=4.0, use_pipeline=True, output_type="latent")
        g = build_guidance({"anchor": {"scale": 4.0, "ratio": 1}})
        ours, info = self.m.generate([PROMPT], height=1024, width=1024, steps=6, guidance=g, output_type="latent")
        self.assertTrue(torch.equal(ref, ours))
        self.assertEqual(info["guidance"]["terms"][0]["ratio"], 1)

    def test_anchor_ratio_two_runs(self):
        from reg.extrap.dype import apply_dype, remove_dype

        apply_dype(self.m.pipe.transformer, method="ntk", dype=True)
        try:
            g = build_guidance({"anchor": {"scale": 4.0}, "height": 2048, "width": 2048})
            lat, info = self.m.generate([PROMPT], height=2048, width=2048, steps=4, guidance=g, output_type="latent")
            base, _ = self.m.generate([PROMPT], height=2048, width=2048, steps=4, guidance_scale=4.0, output_type="latent")
        finally:
            remove_dype(self.m.pipe.transformer)
        self.assertTrue(torch.isfinite(lat).all())
        self.assertFalse(torch.equal(lat, base))
        self.assertEqual(info["guidance"]["terms"][0]["ratio"], 2)
