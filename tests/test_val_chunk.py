"""--val-image-chunk (level-1 validation memory, tf-intensity-aug addendum 2): the chunked deform
validation metrics equal the unchunked ones (identity baselines and LNCC included; plain and lean
sampler), and on a GPU (fixed TF memory limit from tests/tf_setup.py) the validation peak drops and
a train -> validate -> train sequence runs. Peaks are printed for the report."""
import sys
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory setup)
from tests.runtime_helpers import free_memory, narrow_config, synthetic_group, tiny_config
import numpy as np
import tensorflow as tf

from models.group_net import GroupRegLevelModel
from runtime import fixed_data as fd
from runtime import optimizers as O
from runtime.run import Trainer, TrainConfig
from runtime.schedulers import ScheduleConfig

N = 4


def make(level, chunk, pair_chunk=16, lean=False, dgroup=None, S=None, n=N, ncc=1.0, tiny=False):
    cfg = TrainConfig(level=level, group_size=n, steps=10, val_every=0, log_every=1, task_ratio={"deform": 1.0},
                      schedule=ScheduleConfig("constant", base_lr=1e-3, total_steps=10, warmup_steps=0), seed=0,
                      lambda_ncc_tmpl=ncc, lambda_ncc_pair=ncc, val_image_chunk=chunk, val_pair_chunk=pair_chunk,
                      lean_sampler=lean)
    mc = (tiny_config(seed=1, deform_decoder=True) if tiny else
          narrow_config(dropout=0.0, seed=1, deform_decoder=True))
    model = GroupRegLevelModel(level, mc,
                               lean_sampler=lean).build_model(n)
    vs = fd.DeformValidationSet.from_group(dgroup["images"], dgroup["labels"], level)
    return Trainer(model, O.make_optimizer("adam", lr=1e-3), {"deform": fd.FixedGroupSource(dgroup["batch"])},
                   S, cfg, val_sets={"deform": vs})


class TestChunkedEqualsUnchunked(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dgroup = fd.synthetic_deform_group(4, N, seed=0)
        cls.S = synthetic_group(4, N, "rigid", seed=0)["S"]

    def tearDown(self):
        free_memory()

    def test_metrics_equal(self):
        for lean in (False, True):
            ref = None
            for chunk, pc in ((0, 16), (1, 1), (3, 2)):
                tr = make(4, chunk, pc, lean, self.dgroup, self.S)
                tr.train_step("deform", self.dgroup["batch"])
                tr.train_step("deform", self.dgroup["batch"])      # non-zero fields
                m = tr.validate()
                if ref is None:
                    ref = m
                    continue
                keys = sorted(k for k in ref if not k.endswith("time_s"))
                self.assertEqual(keys, sorted(k for k in m if not k.endswith("time_s")))
                for k in keys:
                    self.assertAlmostEqual(float(ref[k]), float(m[k]), delta=1e-4 * max(1.0, abs(float(ref[k]))),
                                           msg="%s (chunk %d, lean %s)" % (k, chunk, lean))
                free_memory()


@unittest.skipUnless(tf.config.list_physical_devices("GPU"), "needs a GPU")
@unittest.skipIf(tf_setup.DEFER_MEMORY_TESTS, tf_setup.DEFER_MEMORY_REASON)
class TestValidationMemory(unittest.TestCase):
    """Level 2 (80x112x96), TINY widths, deform N 2, LNCC 1/1, lean sampler: the train step peaks at
    ~4.6 GB of the 6 GB limit (N 4 does not fit): validation peak unchunked vs chunk 1,
    then a train step after each validation (fixed TF limit of tests/tf_setup.py)."""

    @classmethod
    def setUpClass(cls):
        cls.dgroup = fd.synthetic_deform_group(2, 2, seed=0)
        cls.S = synthetic_group(4, N, "rigid", seed=0)["S"]

    def run_seq(self, tr, chunk, pair_chunk):
        """Validation peak (second pass) of `tr` with the given chunks, then one train step."""
        tr.cfg.val_image_chunk, tr.cfg.val_pair_chunk = chunk, pair_chunk   # read at call time
        tr.validate()                                         # trace + first pass
        base = tf.config.experimental.get_memory_info("GPU:0")["current"]
        tf.config.experimental.reset_memory_stats("GPU:0")
        tr.validate()
        peak = tf.config.experimental.get_memory_info("GPU:0")["peak"]
        res = tr.train_step("deform", self.dgroup["batch"])   # train after validate
        self.assertTrue(bool(res["finite"]))
        return peak / 2.0 ** 20, base / 2.0 ** 20

    def test_peaks(self):
        # One trainer (a second one can exhaust the pinned host pool under WSL): train -> validate
        # unchunked -> train -> validate chunk 1 / pair chunk 1 -> train -> validate unchunked -> train.
        tr = make(2, 0, 16, True, self.dgroup, self.S, n=2, tiny=True)
        tr.train_step("deform", self.dgroup["batch"])
        tf.config.experimental.reset_memory_stats("GPU:0")
        tr.train_step("deform", self.dgroup["batch"])
        t0 = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2.0 ** 20
        v0, b0 = self.run_seq(tr, 0, 16)
        v1, b1 = self.run_seq(tr, 1, 1)
        v2, _ = self.run_seq(tr, 0, 16)
        sys.stderr.write("\n[val chunk, level 2 TINY N 2] train peak %.0f MiB; validation peak unchunked %.0f MiB "
                         "(resident %.0f; again after the chunked pass %.0f), chunk 1 / pair chunk 1 %.0f MiB "
                         "(resident %.0f)\n" % (t0, v0, b0, v2, v1, b1))
        self.assertLess(v1, v0)


if __name__ == "__main__":
    unittest.main()
