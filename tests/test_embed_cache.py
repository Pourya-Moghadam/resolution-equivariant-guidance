"""python -m unittest tests/test_embed_cache.py   (no GPU, no model weights needed)"""

import tempfile
import unittest
from pathlib import Path

import torch

from reg.models.embed_cache import TextEmbedCache


class TestTextEmbedCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def encode(self, text):
        self.calls.append(text)
        return {"embeds": torch.full((4, 3), float(len(text)), dtype=torch.bfloat16), "mask": torch.ones(4, dtype=torch.int64)}

    def test_miss_then_hit_and_persistence(self):
        c = TextEmbedCache("m", root=self.tmp.name, max_sequence_length=4)
        a = c.get("hello", self.encode)
        b = c.get("hello", self.encode)
        self.assertEqual(self.calls, ["hello"])  # encoder ran once
        self.assertEqual((c.hits, c.misses), (1, 1))
        torch.testing.assert_close(a["embeds"], b["embeds"])
        self.assertEqual(a["embeds"].dtype, torch.bfloat16)
        # a fresh instance (new process) reads the same file
        c2 = TextEmbedCache("m", root=self.tmp.name, max_sequence_length=4)
        c2.get("hello", self.encode)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(list(Path(self.tmp.name).rglob("*.safetensors"))), 1)

    def test_key_depends_on_text_model_and_settings(self):
        c = TextEmbedCache("m", root=self.tmp.name, max_sequence_length=4)
        keys = {
            c.key("hello"),
            c.key("hello "),
            TextEmbedCache("m2", root=self.tmp.name, max_sequence_length=4).key("hello"),
            TextEmbedCache("m", root=self.tmp.name, max_sequence_length=8).key("hello"),
        }
        self.assertEqual(len(keys), 4)

    def test_batch_stacks(self):
        c = TextEmbedCache("m", root=self.tmp.name)
        out = c.get_batch(["a", "bb"], self.encode)
        self.assertEqual(out["embeds"].shape, (2, 4, 3))
        self.assertEqual(out["embeds"][1, 0, 0].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
