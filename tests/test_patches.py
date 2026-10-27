"""python -m unittest tests/test_patches.py"""

import unittest

from PIL import Image

from reg.eval.patches import grid_positions, iter_patches


class TestPatches(unittest.TestCase):
    def test_4k_gives_16_distinct_full_size_crops(self):
        pos = grid_positions(4096, 4096, 1024, 16, seed=0)
        self.assertEqual(len(pos), 16)
        self.assertEqual(len(set(pos)), 16)
        self.assertTrue(all(0 <= x <= 3072 and 0 <= y <= 3072 for x, y in pos))
        self.assertIn((0, 0), pos)
        self.assertIn((3072, 3072), pos)  # grid touches the borders
        crops = list(iter_patches(Image.new("RGB", (4096, 4096)), 1024, 16))
        self.assertTrue(all(c.size == (1024, 1024) for c in crops))

    def test_2k_needs_jittered_extras(self):
        pos = grid_positions(2048, 2048, 1024, 16, seed=1)
        self.assertEqual(len(pos), 16)
        self.assertEqual(len(set(pos[:4])), 4)  # the 4 grid cells come first

    def test_deterministic_in_seed(self):
        self.assertEqual(grid_positions(3072, 3072, 1024, 16, seed=3), grid_positions(3072, 3072, 1024, 16, seed=3))
        self.assertNotEqual(grid_positions(2048, 2048, 1024, 16, seed=3), grid_positions(2048, 2048, 1024, 16, seed=4))

    def test_small_image_is_one_resized_crop(self):
        crops = list(iter_patches(Image.new("RGB", (640, 480)), 1024, 16))
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0].size, (1024, 1024))


if __name__ == "__main__":
    unittest.main()
