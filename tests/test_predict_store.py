"""Host stores, store-size estimate and the RAM / disk choice (predict-script design 4.1, 6).

TF-free: runs in stage 1 of tests/run_tests_wsl.sh (no tensorflow import).
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from inference import store as S
from models.config import ModelConfig


class _StoreCases(object):
    """Round trip, dtype, free, nbytes for one store class (mixed into TestCase subclasses)."""

    def make(self):
        raise NotImplementedError

    def test_round_trip_and_free(self):
        rng = np.random.default_rng(0)
        with self.make() as st:
            for dtype in (np.float32, np.float16):
                name = "a_" + np.dtype(dtype).name
                full = rng.normal(size=(5, 3, 4, 2, 2)).astype(dtype)
                st.alloc(name, full.shape, dtype)
                for i0, i1 in ((0, 2), (2, 4), (4, 5)):
                    st.put(name, i0, full[i0:i1])
                got = st.get(name, 1, 4)
                self.assertEqual(got.dtype, np.dtype(dtype))
                np.testing.assert_array_equal(got, full[1:4])
                np.testing.assert_array_equal(st.get(name, 0, 5), full)
            expect = 5 * 3 * 4 * 2 * 2 * (4 + 2)
            self.assertEqual(st.nbytes(), expect)
            self.assertEqual(st.peak_nbytes(), expect)
            st.free("a_float32")
            self.assertNotIn("a_float32", st)
            self.assertEqual(st.nbytes(), 5 * 3 * 4 * 2 * 2 * 2)
            self.assertEqual(st.peak_nbytes(), expect)
            with self.assertRaises(ValueError):
                st.alloc("a_float16", (1,), np.float32)   # already allocated
            with self.assertRaises(ValueError):
                st.alloc("bad/name", (1,), np.float32)
            with self.assertRaises(IndexError):
                st.put("a_float16", 4, np.zeros((2, 3, 4, 2, 2), np.float16))
        self.assertEqual(st.nbytes(), 0)


class TestRamStore(_StoreCases, unittest.TestCase):
    def make(self):
        return S.RamStore()


class TestDiskStore(_StoreCases, unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def make(self):
        return S.DiskStore(os.path.join(self.tmp.name, "stream"))

    def test_files_and_cleanup(self):
        parent = os.path.join(self.tmp.name, "stream")
        st = S.DiskStore(parent)
        st.alloc("x", (2, 3), np.float16)
        path = os.path.join(st.directory, "x.npy")
        self.assertTrue(os.path.isfile(path))
        st.free("x")
        self.assertFalse(os.path.exists(path))
        st.close()
        self.assertFalse(os.path.exists(st.directory))
        self.assertFalse(os.path.exists(parent))   # created by the store and empty

    def test_existing_dir_kept(self):
        parent = self.tmp.name   # like $TMPDIR: exists, must survive
        with S.DiskStore(parent) as st:
            st.alloc("x", (2, 3), np.float32)
        self.assertFalse(os.path.exists(st.directory))
        self.assertTrue(os.path.isdir(parent))

    def test_cleanup_on_error(self):
        parent = os.path.join(self.tmp.name, "stream")
        with self.assertRaises(RuntimeError):
            with S.DiskStore(parent) as st:
                st.alloc("x", (4, 8), np.float32)
                st.put("x", 0, np.ones((2, 8), np.float32))
                raise RuntimeError("boom")
        self.assertFalse(os.path.exists(st.directory))
        self.assertFalse(os.path.exists(parent))


class TestGeometry(unittest.TestCase):
    def test_level1_default(self):
        g = S.level_geometry(1, ModelConfig())
        self.assertEqual(g["stem"], ((160, 224, 192), 32))
        self.assertEqual(g["l2"], ((80, 112, 96), 64))
        self.assertEqual(g["l3"], ((40, 56, 48), 128))
        self.assertEqual(g["l6"], ((5, 7, 6), 384))
        self.assertEqual(g["d2"], ((80, 112, 96), 32))
        self.assertEqual(g["d1"], ((160, 224, 192), 16))

    def test_bottleneck_every_level(self):
        for level in (4, 2, 1):
            self.assertEqual(S.level_geometry(level, ModelConfig())["l6"][0], (5, 7, 6))


class TestEstimate(unittest.TestCase):
    def test_level1_deform_n101(self):
        cfg = ModelConfig()
        gb = S.estimate_store_gb(1, cfg, 101, "deform", "float16")
        print("\n[estimate] level 1 default widths N 101 deform fp16: %.1f GB "
              "(float32 %.1f, affine+deform %.1f, affine %.1f)"
              % (gb, S.estimate_store_gb(1, cfg, 101, "deform", "float32"),
                 S.estimate_store_gb(1, cfg, 101, "affine+deform", "float16"),
                 S.estimate_store_gb(1, cfg, 101, "affine", "float16")), file=sys.stderr)
        self.assertLessEqual(abs(gb - 40.0), 0.25 * 40.0)

    def test_monotone_and_ordering(self):
        cfg = ModelConfig()
        vals = [S.estimate_store_gb(1, cfg, n, "deform") for n in (2, 10, 50, 101)]
        self.assertEqual(vals, sorted(vals))
        self.assertLess(vals[0], vals[1])
        self.assertLess(S.estimate_store_gb(1, cfg, 101, "deform", "float16"),
                        S.estimate_store_gb(1, cfg, 101, "deform", "float32"))
        self.assertLess(S.estimate_store_gb(1, cfg, 101, "deform"),
                        S.estimate_store_gb(1, cfg, 101, "affine+deform"))
        self.assertLess(S.estimate_store_gb(1, cfg, 101, "rigid"),
                        S.estimate_store_gb(1, cfg, 101, "deform"))
        self.assertLess(S.estimate_store_gb(4, cfg, 101, "deform"),
                        S.estimate_store_gb(2, cfg, 101, "deform"))

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            S.estimate_store_gb(1, ModelConfig(), 4, "rigid+deform")
        with self.assertRaises(ValueError):
            S.estimate_store_gb(1, ModelConfig(), 4, "deform", "bfloat16")


class TestChooseStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_auto_slurm(self):
        d = os.path.join(self.tmp.name, "_stream_tmp")
        with mock.patch.dict(os.environ, {"SLURM_MEM_PER_NODE": "96000"}):
            budget, source = S.memory_budget_gb()
            self.assertEqual(source, "SLURM_MEM_PER_NODE")
            self.assertAlmostEqual(budget, 96000 * 2 ** 20 / 1e9)
            ram = S.choose_store("auto", 40.0, d)
            self.assertIsInstance(ram, S.RamStore)
            self.assertEqual(ram.choice["kind"], "ram")
            with S.choose_store("auto", 0.71 * budget, d) as disk:
                self.assertIsInstance(disk, S.DiskStore)
                self.assertEqual(disk.choice["kind"], "disk")
                self.assertTrue(disk.directory.startswith(d))
        self.assertFalse(os.path.exists(d))

    def test_explicit_kinds(self):
        self.assertIsInstance(S.choose_store("ram", 1e6), S.RamStore)
        with S.choose_store("disk", 0.0, self.tmp.name) as st:
            self.assertIsInstance(st, S.DiskStore)
        with self.assertRaises(ValueError):
            S.choose_store("disk", 0.0, None)
        with self.assertRaises(ValueError):
            S.choose_store("gpu", 0.0, self.tmp.name)

    def test_meminfo_and_fallback(self):
        path = os.path.join(self.tmp.name, "meminfo")
        with open(path, "w") as fh:
            fh.write("MemTotal:       131072000 kB\nMemAvailable:    65536000 kB\n")
        budget, source = S.memory_budget_gb(environ={}, meminfo=path)
        self.assertEqual(source, "MemAvailable")
        self.assertAlmostEqual(budget, 65536000 * 1024 / 1e9)
        self.assertEqual(S.memory_budget_gb(environ={}, meminfo=path + "_missing"), (None, None))
        with mock.patch.object(S, "memory_budget_gb", return_value=(None, None)):
            with self.assertLogs(S.logger, level="WARNING"):
                st = S.choose_store("auto", 1e6, self.tmp.name)
        self.assertIsInstance(st, S.RamStore)


if __name__ == "__main__":
    unittest.main()
