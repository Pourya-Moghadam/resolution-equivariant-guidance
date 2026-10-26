"""python -m unittest tests/test_jobs.py"""

import json
import tempfile
import unittest
from pathlib import Path

from reg.jobs import is_done, my_rows, pending, read_jobs, write_jobs


class TestJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "j.jsonl"
        rows = [{"prompt": f"p{i}", "model": "flux" if i % 3 == 0 else "lumina", "seed": i} for i in range(23)]
        write_jobs(self.path, rows)

    def tearDown(self):
        self.tmp.cleanup()

    def test_defaults_and_ids(self):
        rows = read_jobs(self.path)
        self.assertEqual(len(rows), 23)
        self.assertEqual(rows[0]["id"], "r000000")
        self.assertEqual((rows[0]["steps"], rows[0]["guidance"]), (28, 3.5))  # flux defaults
        self.assertEqual((rows[1]["steps"], rows[1]["guidance"]), (30, 4.0))  # lumina defaults
        self.assertEqual(rows[5]["extrap"], "none")

    def test_split_is_a_partition(self):
        rows = read_jobs(self.path)
        for n in (1, 4, 5):
            shares = [my_rows(rows, k, n) for k in range(n)]
            ids = [r["id"] for s in shares for r in s]
            self.assertEqual(sorted(ids), sorted(r["id"] for r in rows))  # every row exactly once
            self.assertEqual(len(ids), len(set(ids)))
        with self.assertRaises(ValueError):
            my_rows(rows, 5, 5)

    def test_resume_skips_done_rows(self):
        rows = read_jobs(self.path)
        out = Path(self.tmp.name) / "out"
        out.mkdir()
        (out / "r000003.json").write_text(json.dumps({"status": "ok"}))
        self.assertTrue(is_done(rows[3], out))
        self.assertEqual(len(pending(rows, out)), 22)

    def test_duplicate_ids_and_missing_prompt_rejected(self):
        p = Path(self.tmp.name) / "bad.jsonl"
        p.write_text('{"id":"a","prompt":"x"}\n{"id":"a","prompt":"y"}\n')
        with self.assertRaises(ValueError):
            read_jobs(p)
        p.write_text('{"id":"a"}\n')
        with self.assertRaises(ValueError):
            read_jobs(p)

    def test_comments_and_blank_lines_ignored(self):
        p = Path(self.tmp.name) / "c.jsonl"
        p.write_text('# header\n\n{"prompt":"x"}\n')
        self.assertEqual(len(read_jobs(p)), 1)


if __name__ == "__main__":
    unittest.main()
