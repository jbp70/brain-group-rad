"""Host stores, store-size estimate and the RAM / disk choice (predict-script design 4.1, 6).

TF-free: runs in stage 1 of tests/run_tests_wsl.sh (no tensorflow import).
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from data_loading import params as P
from inference import store as S
from models.config import ModelConfig

# estimate_store_gb(level, ModelConfig(), 101, branches, dtype) of commit 19b8a60 (before the TTA
# term; printed by running that commit's inference/store.py), TTA design T11.
HEAD_ESTIMATES_N101 = {
    (4, "rigid", "float16"): 9.07855872, (4, "rigid", "float32"): 13.2486144,
    (4, "affine", "float16"): 9.07855872, (4, "affine", "float32"): 13.2486144,
    (4, "deform", "float16"): 10.85952, (4, "deform", "float32"): 16.68022272,
    (4, "affine+deform", "float16"): 13.63955712, (4, "affine+deform", "float32"): 19.46025984,
    (2, "rigid", "float16"): 19.1127552, (2, "rigid", "float32"): 33.0129408,
    (2, "affine", "float16"): 19.1127552, (2, "affine", "float32"): 33.0129408,
    (2, "deform", "float16"): 24.58595328, (2, "deform", "float32"): 42.65619456,
    (2, "affine+deform", "float16"): 27.3659904, (2, "affine+deform", "float32"): 45.43623168,
    (1, "rigid", "float16"): 33.0129408, (1, "rigid", "float32"): 35.44547328,
    (1, "affine", "float16"): 33.0129408, (1, "affine", "float32"): 35.44547328,
    (1, "deform", "float16"): 44.7412224, (1, "deform", "float32"): 54.21072384,
    (1, "affine+deform", "float16"): 47.52125952, (1, "affine+deform", "float32"): 56.99076096,
}


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

    def test_unchanged_without_tta(self):
        """TTA design 6 / T11: the default (no tta_levels) equals the values of the code before
        the TTA term (commit 19b8a60, N 101, default widths), for every level / branch / dtype."""
        cfg = ModelConfig()
        for (lv, b, dt), want in HEAD_ESTIMATES_N101.items():
            got = S.estimate_store_gb(lv, cfg, 101, b, dt)
            self.assertEqual(got, want, (lv, b, dt))
            self.assertEqual(S.estimate_store_gb(lv, cfg, 101, b, dt, tta_levels=()), want, (lv, b, dt))

    def test_tta_term(self):
        """T11: +28 V bytes per image at a TTA deform level (s1 12 V + s2 4 V + c0 12 V); rigid
        unaffected; affine+deform keeps one src_affine term."""
        cfg = ModelConfig()
        n = 101
        for level in (4, 2, 1):
            run_levels = [lv for lv in (4, 2, 1) if lv >= level]
            term = 28 * int(np.prod(P.LEVEL_SHAPE[level])) * n / 1e9
            for dt in ("float16", "float32"):
                for b in ("deform", "affine+deform", "rigid"):
                    base = S.estimate_store_gb(level, cfg, n, b, dt)
                    self.assertEqual(S.estimate_store_gb(level, cfg, n, b, dt, tta_levels=()), base)
                    tta = S.estimate_store_gb(level, cfg, n, b, dt, tta_levels=(level,))
                    all_tta = S.estimate_store_gb(level, cfg, n, b, dt, tta_levels=tuple(run_levels))
                    if b == "rigid":
                        self.assertEqual(tta, base, (level, dt))
                        self.assertEqual(all_tta, base, (level, dt))
                    elif level == 1 and dt == "float16":
                        # the frozen float32 level-2 term is the max without TTA here, so the
                        # max over levels grows by less than the level-1 term
                        self.assertGreater(tta - base, 0.0)
                        self.assertLess(tta - base, term)
                        self.assertEqual(all_tta, tta, (level, b, dt))
                    else:
                        # the run's own (finest) level is the max term: exactly + 28 V n
                        self.assertAlmostEqual(tta - base, term, places=9, msg=(level, b, dt))
                        self.assertEqual(all_tta, tta, (level, b, dt))
                d_plain = (S.estimate_store_gb(level, cfg, n, "affine+deform", dt)
                           - S.estimate_store_gb(level, cfg, n, "deform", dt))
                d_tta = (S.estimate_store_gb(level, cfg, n, "affine+deform", dt, tta_levels=(level,))
                         - S.estimate_store_gb(level, cfg, n, "deform", dt, tta_levels=(level,)))
                self.assertAlmostEqual(d_tta, d_plain, places=9, msg=(level, dt))
        # level-1 checkpoint, N 101, default widths, level 1 the max term (float32 trainable
        # stage): +28 x 6881280 x 101 B = 19.46 GB (design 6)
        grow = (S.estimate_store_gb(1, cfg, 101, "deform", "float32", tta_levels=(4, 2, 1))
                - S.estimate_store_gb(1, cfg, 101, "deform", "float32"))
        self.assertAlmostEqual(grow, 28 * 6881280 * 101 / 1e9, places=9)
        self.assertAlmostEqual(grow, 19.46, places=2)
        g16 = (S.estimate_store_gb(1, cfg, 101, "deform", tta_levels=(4, 2, 1))
               - S.estimate_store_gb(1, cfg, 101, "deform"))
        print("\n[estimate] level 1 N 101 deform with TTA at all levels: float32 %.2f GB (+%.2f), "
              "float16 %.2f GB (+%.2f)"
              % (S.estimate_store_gb(1, cfg, 101, "deform", "float32", tta_levels=(4, 2, 1)), grow,
                 S.estimate_store_gb(1, cfg, 101, "deform", tta_levels=(4, 2, 1)), g16), file=sys.stderr)

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
