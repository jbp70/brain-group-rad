"""Tests of `inference.loader` (predict-script FEAT-003; design 3.2, 3.3, AC 5).

Precision mapping (an AMP checkpoint with deform_decoder_dtype 'float16' loads under float32 with
'policy' recorded, under 'checkpoint' on the GPU with float16 decoder stages, and with the CPU
rule as 'float32 (cpu)'), old configs without the knob keys, fingerprint mismatches at levels 2
and 1, missing config.json, a deform branch on a model without decoder, and the unchunked
group-size bound (default widths, level 1: deform 4, linear 9) from the config alone.
"""
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory limit)
from tests.predict_helpers import save_cascade_run, save_chain_run, save_level4_run
from tests.runtime_helpers import free_memory, tiny_config
from tests.test_cascade import STAGE_KW, randomize
from tests.test_predict_streaming import TINY_L1
import numpy as np
import tensorflow as tf

from inference import loader as LD
from models import cascade as CS
from models.config import ModelConfig
from models.group_net import DECODER_STAGES, GroupRegLevelModel

DEFORM_KW = dict(deform_decoder=True, seed=1, integration_steps=3)


def _assert_weights_equal(tc, got, ref, label):
    tc.assertEqual(len(got.weights), len(ref.weights), label)
    for a, b in zip(got.weights, ref.weights):
        tc.assertEqual(tuple(a.shape), tuple(b.shape), "%s %s" % (label, a.name))
        np.testing.assert_array_equal(a.numpy(), b.numpy(), "%s %s" % (label, a.name))


def _edit_config(prefix, fn):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(prefix))), "config.json")
    with open(path) as f:
        cfg = json.load(f)
    fn(cfg)
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


class LoaderCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_predict_loader_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        tf.keras.mixed_precision.set_global_policy("float32")
        free_memory()


class TestLoaderLevel4(LoaderCase):
    def test_float16_decoder_precision_mapping(self):
        cfg = tiny_config(**DEFORM_KW)
        prefix, src = save_level4_run(self.tmp, "amp", cfg, seed=3, amp=True,
                                      extra_args={"deform_decoder_dtype": "float16"})
        lm = LD.load_model(os.path.dirname(prefix), "deform", precision="float32")
        self.assertEqual(lm.prefix, os.path.abspath(prefix))
        self.assertEqual((lm.level, lm.step, lm.frozen), (4, 7, None))
        self.assertEqual(lm.precision["deform_decoder_dtype"],
                         {"checkpoint": "float16", "used": "policy (float32 build)"})
        self.assertEqual(lm.precision["trainable"], "float32")
        self.assertEqual(lm.knobs["lean_sampler"], {"checkpoint": True, "used": True})
        self.assertTrue(lm.model.lean_sampler)
        for name in DECODER_STAGES:
            self.assertEqual(getattr(lm.model.deform_decoder, name).compute_dtype, "float32", name)
        _assert_weights_equal(self, lm.model, src, "float32 build")
        self.assertEqual(lm.training_group_sizes, {"group_size": 4, "deform_group_size": 2})
        self.assertEqual(tf.keras.mixed_precision.global_policy().name, "float32")
        # CPU rule on an AMP checkpoint: float32 with a WARNING
        with self.assertLogs(LD.logger, level=logging.WARNING):
            lm_cpu = LD.load_model(prefix, "deform", precision="checkpoint", device="cpu")
        self.assertEqual(lm_cpu.precision["trainable"], "float32 (cpu)")
        self.assertEqual(lm_cpu.precision["deform_decoder_dtype"]["used"], "policy (float32 build)")
        self.assertEqual(lm_cpu.model.deform_decoder.d3.compute_dtype, "float32")
        # 'checkpoint' on the GPU: mixed_float16 with float16 decoder stages
        lm16 = LD.load_model(prefix, "deform", precision="checkpoint", device="gpu")
        self.assertEqual(lm16.precision["trainable"], "mixed_float16")
        self.assertEqual(lm16.precision["deform_decoder_dtype"],
                         {"checkpoint": "float16", "used": "float16"})
        for name in DECODER_STAGES:
            self.assertEqual(getattr(lm16.model.deform_decoder, name).compute_dtype, "float16", name)
        _assert_weights_equal(self, lm16.model, src, "mixed_float16 build")
        self.assertEqual(tf.keras.mixed_precision.global_policy().name, "float32")

    def test_old_config_defaults(self):
        cfg = tiny_config(**DEFORM_KW)
        prefix, src = save_level4_run(self.tmp, "old", cfg, seed=4)

        def drop(c):
            for k in ("amp", "split_group_conv", "deform_decoder_dtype", "fp16_act_clip",
                      "fp16_act_clip_scope", "bn_epsilon_override", "lowres_method",
                      "lean_sampler", "group_size", "deform_group_size"):
                del c["args"][k]
        _edit_config(prefix, drop)
        lm = LD.load_model(prefix, "affine+deform", precision="checkpoint")
        k = lm.knobs
        self.assertEqual((k["amp"], k["split_group_conv"], k["deform_decoder_dtype"], k["fp16_act_clip"],
                          k["fp16_act_clip_scope"], k["bn_epsilon_override"], k["lowres_method"]),
                         (False, False, "policy", 0.0, "deform_decoder", 0.0, None))
        self.assertEqual(k["lean_sampler"], {"checkpoint": False, "used": True})
        self.assertFalse(lm.model.split_group_conv)
        self.assertEqual(lm.precision["trainable"], "float32")
        _assert_weights_equal(self, lm.model, src, "old config")

    def test_missing_config_bad_path_no_decoder(self):
        cfg = tiny_config()                                   # no deform decoder
        prefix, _ = save_level4_run(self.tmp, "lin", cfg, seed=5)
        self.assertEqual(LD.load_model(prefix, "rigid").level, 4)
        for br in ("deform", "affine+deform"):
            with self.assertRaisesRegex(LD.LoaderError, "deform decoder"):
                LD.load_model(prefix, br)
        with self.assertRaisesRegex(LD.LoaderError, "branches must be one of"):
            LD.load_model(prefix, "rigid+deform")
        with self.assertRaises(LD.LoaderError):
            LD.load_model(os.path.join(self.tmp, "nope", "ckpt", "last"), "rigid")
        os.remove(os.path.join(self.tmp, "lin", "config.json"))
        with self.assertRaisesRegex(LD.LoaderError, "no config.json next to the checkpoint"):
            LD.load_model(prefix, "rigid")
        self.assertTrue(issubclass(LD.LoaderError, ValueError))


class TestLoaderCascade(LoaderCase):
    def test_level2_restore_and_fingerprint(self):
        cfg = tiny_config(**DEFORM_KW)
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        randomize(m4, 3, 0.02)
        frozen = CS.FrozenStage.from_model(m4)
        m2 = GroupRegLevelModel(2, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
        m2.copy_weights_from(m4)
        randomize(m2, 4, 0.01)
        prefix = save_cascade_run(self.tmp, "l2", m2, frozen)
        lm = LD.load_model(prefix, "affine+deform", precision="float32")
        self.assertEqual((lm.level, lm.step, lm.levels), (2, 5, (4, 2)))
        self.assertIsInstance(lm.frozen, CS.FrozenStage)
        _assert_weights_equal(self, lm.model, m2, "level 2")
        _assert_weights_equal(self, lm.frozen.model, m4, "frozen level 4")

        def alter(c):
            c["cascade"]["fingerprint"]["abs_sum"] *= 1.001
        _edit_config(prefix, alter)
        with self.assertRaisesRegex(LD.LoaderError, "level-4"):
            LD.load_model(prefix, "deform")

        def drop(c):
            del c["cascade"]
        _edit_config(prefix, drop)
        with self.assertRaisesRegex(LD.LoaderError, "cascade.fingerprint"):
            LD.load_model(prefix, "deform")

    def test_level1_chain_restore_and_fingerprint_cpu(self):
        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("tf_gpu env requires a visible GPU (the level-1 case runs on /CPU:0)")
        t0 = time.time()
        with tf.device("/CPU:0"):
            cfg = tiny_config(**dict(STAGE_KW, **TINY_L1))
            m4 = GroupRegLevelModel(4, cfg).build_model(2)
            randomize(m4, 3, 0.02)
            m2 = GroupRegLevelModel(2, cfg).build_model(2)
            m2.copy_weights_from(m4)
            randomize(m2, 4, 0.01)
            chain = CS.FrozenChain.from_models(m4, m2)
            m1 = GroupRegLevelModel(1, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
            m1.copy_weights_from(chain.model2)
            randomize(m1, 5, 0.01)
            prefix = save_chain_run(self.tmp, "l1", m1, chain)
            lm = LD.load_model(prefix, "deform", precision="float32")
            self.assertEqual((lm.level, lm.levels), (1, (4, 2, 1)))
            self.assertIsInstance(lm.frozen, CS.FrozenChain)
            _assert_weights_equal(self, lm.model, m1, "level 1")
            _assert_weights_equal(self, lm.frozen.stage4.model, m4, "frozen level 4")
            _assert_weights_equal(self, lm.frozen.model2, m2, "frozen level 2")

            def alter(c):
                c["cascade"]["stages"][1]["fingerprint"]["sq_sum"] *= 1.001
            _edit_config(prefix, alter)
            with self.assertRaisesRegex(LD.LoaderError, "level-2"):
                LD.load_model(prefix, "deform")
        print("\n[loader] level-1 chain restore on CPU %.0f s" % (time.time() - t0), file=sys.stderr)


class TestUnchunkedBound(unittest.TestCase):
    """Design 3.2 step 4b from the config alone (no model build)."""

    def test_default_widths_level1(self):
        cfg = ModelConfig(deform_decoder=True)
        self.assertEqual(LD.unchunked_n_max(1, cfg, ("deform",), True), 4)
        self.assertEqual(LD.unchunked_n_max(1, cfg, ("rigid",), True), 9)
        self.assertEqual(LD.unchunked_n_max(1, cfg, ("affine",), True), 9)
        self.assertEqual(LD.unchunked_n_max(1, cfg, "affine+deform", True), 4)
        self.assertEqual(LD.unchunked_n_max(1, cfg, "rigid", True), 9)
        self.assertEqual(LD.unchunked_n_max(1, cfg, ("deform",), False), 2)   # group concat 2C
        self.assertEqual(LD.check_unchunked(4, 1, cfg, "deform", True), 4)
        self.assertEqual(LD.check_unchunked(9, 1, cfg, "affine", True), 9)
        with self.assertRaises(LD.LoaderError) as cm:
            LD.check_unchunked(5, 1, cfg, "deform", True)
        msg = str(cm.exception)
        print("\n[loader] n_max message: %s" % msg, file=sys.stderr)
        for part in ("unchunked mode at level 1 supports at most 4 subjects", "d1 skip concat",
                     "440401920 elements per image", "--mode streamed or fewer --n-subjects"):
            self.assertIn(part, msg)
        with self.assertRaisesRegex(LD.LoaderError, "at most 9 subjects"):
            LD.check_unchunked(10, 1, cfg, "rigid", True)
        # levels 2 / 4 (frozen stages unsplit): bounded by the trainable stage for default widths
        self.assertEqual(LD.unchunked_n_max(2, cfg, "deform", True), 19)
        self.assertEqual(LD.unchunked_n_max(4, cfg, "deform", True), 89)


if __name__ == "__main__":
    unittest.main()
