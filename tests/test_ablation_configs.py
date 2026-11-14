"""Paper-ablation model variants end to end (ablation-prereqs, `.agents/tasks/ablation-prereqs/`).

Group-block placement (FEAT-001, plan items 1-5; mapping in linear-heads.md): ``--group-config``
'none' (no group blocks) and 'E' (stem / encoder only) through
* TestGroupConfigCascade: a TINY level-2 cascade Trainer from a level-4 source of the same config
  (frozen stage via `FrozenStage.from_model`), one rigid, affine and deform step each (finite,
  trainable weights move, frozen ones do not); mixed configs across levels are rejected
  (`check_cascade_config`, `FrozenChain`; plan D3).
* TestGroupConfigMain: main.py overfit-synthetic at level 4 with --group-config none, then
  --cascade-from at level 2 (config.json model_config.group_config, ModelConfig round trip) and a
  level-2 run with another --group-config refused (SystemExit naming group_config).
* TestGroupConfigPredict: predict.py `run` on level-2 cascades of config E and none, --branches
  affine+deform: default (internal check), --levels 2 (skip of level 4), --tta flip8 + --recenter 1;
  and `check-streaming` (float32, deform) passes.

Instance norm (FEAT-002, plan item 6, decisions D4 / D5): ``--norm instance`` with the final recipe's
knobs (AMP, gradient checkpointing 'full', lean sampler, split group conv, config B) through
* TestInstanceNormTraining: level-4 three-task Trainer steps under mixed_float16 (no BN layers,
  nothing to freeze, no BN restore in `checkpointing.recompute`); a level-2 cascade from an IN
  level-4 source (float32 FrozenStage); a level-1 chain (`FrozenChain.from_models`) with the
  float32_fine deform decoder at TINY_L1 widths.
* TestInstanceNormParser: --bn-recal-batches 0 with --norm instance parses at level 1, > 0 exits.
* TestInstanceNormStreaming: StreamedLevel == model(x, training=False) for IN, levels 4 and 2,
  float32 and mixed_float16, chunks 1 / 2 / N (the bounds of tests/test_predict_streaming.py).
* TestInstanceNormPredict: predict.py run on an IN level-2 cascade, affine+deform, flip8 + recenter 1.
From-scratch level 1 (FEAT-002, plan item 8, decision D6):
* TestScratchLevel1Main: main.py overfit-synthetic at --level 1 without --cascade-from with the
  r_level1_scratch.sh knobs (delayed deform, curriculum, delayed G, float32_fine, AMP, ...), a resume
  with the same run name, and the same path with --norm instance --group-config none.
Measured values are printed to stderr.
"""
from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory limit)
from tests import test_predict_streaming as TS
from tests.predict_helpers import save_cascade_run, write_subjects
from tests.runtime_helpers import TINY_ARGS, free_memory, synthetic_group, tiny_config
from tests.test_cascade import (N2, N_CLASSES, ND2, STAGE_CKPT, STAGE_KW, cascade_config, chain_config,
                                randomize)
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np
import tensorflow as tf

import main as MAIN
from inference import cli as CLI
from inference import predict as PR
from models import cascade as CS
from models import checkpointing as CK
from models import layers as L
from models.config import ModelConfig
from models.group_net import GroupRegLevelModel
from models.precision import policy_scope
from runtime import args as A
from runtime import checkpoint as C
from runtime import fixed_data as fd
from runtime import optimizers as O
from runtime.run import Trainer, TrainConfig
from runtime.schedulers import ScheduleConfig

COMMON = ["--log-backend", "jsonl", "--quiet", "true", "--seed", "0"]
RATIO3 = {"rigid": 1.0, "affine": 1.0, "deform": 2.0}
IN_KW = dict(norm="instance", group_config="B")          # the IN ablation: config B, instance norm


def _say(msg):
    print("\n[ablation] " + msg, file=sys.stderr)


def _vals(variables):
    return [v.numpy().copy() for v in variables]


def _same(before, variables):
    return all(np.array_equal(a, v.numpy()) for a, v in zip(before, variables))


class TestGroupConfigCascade(unittest.TestCase):
    """Level-2 cascade training steps for configs none / E; mixed configs rejected."""

    @classmethod
    def setUpClass(cls):
        groups = {t: synthetic_group(2, N2, t, seed=i) for i, t in enumerate(("rigid", "affine"))}
        cls.S = groups["rigid"]["S"]
        cls.batches = {"rigid": groups["rigid"]["batch"], "affine": groups["affine"]["batch"],
                       "deform": fd.synthetic_deform_group(2, ND2, seed=0)["batch"]}

    def tearDown(self):
        free_memory()

    def _trainer(self, group_config):
        cfg = tiny_config(group_config=group_config, **STAGE_KW)
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        randomize(m4, 3, 0.02)
        m2 = GroupRegLevelModel(2, cfg, gradient_checkpointing=STAGE_CKPT).build_model(2)
        m2.copy_weights_from(m4)
        frozen = CS.FrozenStage.from_model(m4)
        sources = {t: fd.FixedGroupSource(b) for t, b in self.batches.items()}
        return Trainer(m2, O.make_optimizer("adam", lr=1e-3), sources, self.S, cascade_config(),
                       cascade=frozen)

    def test_cascade_steps(self):
        for g in ("none", "E"):
            tr = self._trainer(g)
            self.assertEqual(tr.cascade.model.config.group_config, g)
            fz_before = _vals(tr.cascade.model.weights)
            tv_before = _vals(tr.model.trainable_variables)
            losses = {}
            for t in ("rigid", "affine", "deform"):
                res = tr.train_step(t, self.batches[t])
                self.assertTrue(res["finite"], (g, t))
                self.assertTrue(np.isfinite(res["loss"]), (g, t))
                losses[t] = float(res["loss"])
            self.assertTrue(_same(fz_before, tr.cascade.model.weights), g)
            self.assertFalse(_same(tv_before, tr.model.trainable_variables), g)
            self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})
            _say("level-2 cascade config %s: losses %s" % (g, {k: "%.4g" % v for k, v in losses.items()}))
            del tr
            free_memory()

    def test_mixed_configs_rejected(self):
        e, b, n = (tiny_config(group_config=g, **STAGE_KW) for g in ("E", "B", "none"))
        with self.assertRaises(ValueError) as cm:
            C.check_cascade_config(e, b)
        self.assertIn("group_config", str(cm.exception))
        C.check_cascade_config(e, e.replace(seed=7))                  # seed may differ
        m4 = GroupRegLevelModel(4, n).build_model(1)
        m2 = GroupRegLevelModel(2, b).build_model(1)
        with self.assertRaises(ValueError) as cm:
            CS.FrozenChain.from_models(m4, m2)
        self.assertIn("group_config", str(cm.exception))


class TestGroupConfigMain(unittest.TestCase):
    """main.py: --group-config none at level 4 -> --cascade-from at level 2."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_abl_main_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        free_memory()

    def run_dir(self, name):
        return os.path.join(self.tmp, "results", name)

    def test_none_level4_to_level2(self):
        res = os.path.join(self.tmp, "results")
        common = ["--overfit-one-group", "--overfit-synthetic", "--group-size", "4", "--task-ratio", "rigid:1",
                  "--deform-decoder", "true", "--log-every", "1", "--val-every", "0", "--val-png", "false",
                  "--results", res, "--overfit-max-steps", "1"] + TINY_ARGS + COMMON
        none = common + ["--group-config", "none"]
        self.assertIn(MAIN.main(none + ["--level", "4", "--run-name", "src4"]), (0, 1))
        free_memory()
        src = os.path.join(self.run_dir("src4"), "ckpt", "last")
        self.assertIn(MAIN.main(none + ["--level", "2", "--cascade-from", src, "--run-name", "c2"]), (0, 1))
        free_memory()
        for name in ("src4", "c2"):
            with open(os.path.join(self.run_dir(name), "config.json")) as f:
                rec = json.load(f)
            self.assertEqual(rec["model_config"]["group_config"], "none", name)
            self.assertEqual(rec["args"]["group_config"], "none", name)
            cfg = ModelConfig.from_dict(rec["model_config"])
            self.assertEqual((cfg.group_encoder(), cfg.group_decoders()), (False, False))
        with open(os.path.join(self.run_dir("c2"), "config.json")) as f:
            rec = json.load(f)
        self.assertEqual((rec["init_weights"]["via"], rec["init_weights"]["mode"]), ("cascade", "full"))
        with open(os.path.join(self.run_dir("c2"), "overfit_summary.json")) as f:
            s = json.load(f)
        self.assertTrue(np.isfinite(s["final_val"]["val_rigid/center_rms_mm"]))
        # Another group config at level 2 than the source's: refused, naming the field.
        with self.assertRaises(SystemExit) as cm:
            MAIN.main(common + ["--group-config", "B", "--level", "2", "--cascade-from", src,
                                "--run-name", "bad"])
        self.assertIn("group_config", str(cm.exception.code))


class TestGroupConfigPredict(unittest.TestCase):
    """predict.py on TINY level-2 cascades of configs E and none (N 3, 1 mm synthetic subjects)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgr_abl_pred_")
        cls.proc = os.path.join(cls.tmp, "proc")
        write_subjects(cls.proc, 3, textured=True, seed=2)
        cls.ckpt = {}
        for g in ("E", "none"):
            cfg = tiny_config(group_config=g, **STAGE_KW)
            m4 = GroupRegLevelModel(4, cfg).build_model(2)
            randomize(m4, 3, 0.02)
            frozen = CS.FrozenStage.from_model(m4)
            m2 = GroupRegLevelModel(2, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
            m2.copy_weights_from(m4)
            randomize(m2, 4, 0.01)
            cls.ckpt[g] = save_cascade_run(cls.tmp, "l2_" + g, m2, frozen)
            del m4, m2, frozen
            free_memory()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        free_memory()

    def _run(self, g, name, *extra):
        out = os.path.join(self.tmp, "out_%s_%s" % (g, name))
        args = CLI.parse_args(["run", "--checkpoint", self.ckpt[g], "--processed-dir", self.proc, "--out", out,
                               "--branches", "affine+deform"] + list(extra))
        code = PR.run_predict(args, configure_device=False)
        free_memory()
        with open(os.path.join(out, "predict.json")) as f:
            return code, json.load(f)

    def test_run_variants(self):
        for g in ("E", "none"):
            with self.subTest(config=g):
                code, rec = self._run(g, "default", "--check-internal", "1")
                self.assertEqual(code, 0, rec.get("message"))
                self.assertEqual(rec["nonfinite"]["total"], 0)
                self.assertTrue(rec["internal_check"]["passed"])
                code, rec = self._run(g, "lv2", "--levels", "2", "--allow-untrained-skip",
                                      "--check-internal", "0")
                self.assertEqual(code, 0, rec.get("message"))
                self.assertEqual(rec["nonfinite"]["total"], 0)
                code, rec = self._run(g, "tta_rc", "--tta", "flip8", "--recenter", "1",
                                      "--check-internal", "0")
                self.assertEqual(code, 0, rec.get("message"))
                self.assertEqual((rec["tta"], rec["recenter"]["status"]), ("flip8", "ok"))
                self.assertEqual(rec["nonfinite"]["total"], 0)
                h = [x["mean_mm"] for x in rec["recenter"]["centrality_by_round"]]
                _say("config %s affine+deform flip8 + recenter 1: centrality %s" % (g, h))

    def test_check_streaming(self):
        for g in ("E", "none"):
            with self.subTest(config=g):
                out = os.path.join(self.tmp, "check_%s.json" % g)
                code = PR.run_check_streaming(CLI.parse_args(
                    ["check-streaming", "--checkpoint", self.ckpt[g], "--processed-dir", self.proc,
                     "--out", out, "--n-subjects", "3"]), configure_device=False)
                free_memory()
                with open(out) as f:
                    rec = json.load(f)
                self.assertEqual(code, 0, rec.get("message"))
                self.assertEqual(rec["status"], "pass")
                self.assertEqual(rec["precision"]["bound_set"], "float32")
                _say("check-streaming config %s L2 N 3 deform: %s" % (
                    g, {k: "%.2e" % v for k, v in rec["differences"]["deform"].items()}))


# ============================================================ instance norm (FEAT-002)
def _step_until_finite(test, tr, task, batch, tries=24):
    """Train steps of `task` until one is applied. Under AMP the LossScaleOptimizer skips steps
    whose scaled gradients overflow and halves the scale (the mm^2 linear losses at the initial
    scale 2^15, as in tests/test_cascade.py TestCascadeAMP). The loss itself must stay finite."""
    for _ in range(tries):
        res = tr.train_step(task, batch)
        test.assertTrue(np.isfinite(res["loss"]), task)
        if res["finite"]:
            return res
    test.fail("%s: no finite step in %d tries" % (task, tries))


def _assert_no_bn(test, model):
    """No BatchNorm anywhere: nothing to recalibrate or freeze, no snapshot / restore ops in
    `checkpointing.recompute` (`_restores_needed` False even in training)."""
    test.assertEqual(L.batch_norm_layers(model), [])
    test.assertEqual(Trainer._bn_layers([model]), [])
    test.assertEqual(model.set_bn_stats_frozen(True), 0)
    test.assertEqual(model.set_bn_stats_frozen(False), 0)
    test.assertFalse(CK._restores_needed(L.batch_norm_layers(model), True))
    test.assertIsInstance(model.encoder.l2.norm, L.InstanceNormWithAffine)
    if model.deform_decoder is not None:
        test.assertIsInstance(model.deform_decoder.d1.norm, L.InstanceNormWithAffine)


class TestInstanceNormTraining(unittest.TestCase):
    """--norm instance training steps with the final recipe's knobs (plan item 6 a / b)."""

    def tearDown(self):
        tf.keras.mixed_precision.set_global_policy("float32")
        free_memory()

    def test_level4_three_task_amp(self):
        """Level 4, config B, mixed_float16, gradient checkpointing 'full', lean sampler, split
        group conv: rigid, affine, deform steps finite; the IN gamma / beta train."""
        n = 2
        groups = {t: synthetic_group(4, n, t, seed=i) for i, t in enumerate(("rigid", "affine"))}
        batches = {"rigid": groups["rigid"]["batch"], "affine": groups["affine"]["batch"],
                   "deform": fd.synthetic_deform_group(4, n, seed=0)["batch"]}
        cfg = tiny_config(seed=1, deform_decoder=True, integration_steps=3, **IN_KW)
        tcfg = TrainConfig(level=4, group_size=n, deform_group_size=n, steps=100, val_every=0, log_every=0,
                           task_ratio=dict(RATIO3), seed=0, amp=True, lean_sampler=True, n_classes=N_CLASSES,
                           schedule=ScheduleConfig("constant", base_lr=1e-3, total_steps=100, warmup_steps=0))
        with policy_scope("mixed_float16"):
            model = GroupRegLevelModel(4, cfg, gradient_checkpointing="full", lean_sampler=True,
                                       split_group_conv=True).build_model(n)
            self.assertEqual(model.encoder.l2.compute_dtype, "float16")
            _assert_no_bn(self, model)
            head = model.deform_decoder.svf_head        # non-zero SVF head: deform gradients reach the net
            head.kernel.assign(np.random.default_rng(0).normal(0, 1e-3, head.kernel.shape).astype(np.float32))
            tr = Trainer(model, O.make_optimizer("adam", lr=1e-3, amp=True),
                         {t: fd.FixedGroupSource(b) for t, b in batches.items()}, groups["rigid"]["S"], tcfg)
            norm_vars = model.encoder.l3.norm.trainable_variables + model.deform_decoder.d2.norm.trainable_variables
            self.assertTrue(norm_vars)
            before = _vals(norm_vars)
            losses = {t: float(_step_until_finite(self, tr, t, batches[t])["loss"]) for t in RATIO3}
        self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})
        self.assertFalse(_same(before, norm_vars))
        _say("level-4 IN mixed_float16 three-task: losses %s" % {k: "%.4g" % v for k, v in losses.items()})

    def test_level2_cascade_amp(self):
        """Level-2 cascade from an IN level-4 source: the FrozenStage is float32 with IN (no BN),
        the trainable level-2 stage mixed_float16; one applied step per task, frozen unchanged."""
        groups = {t: synthetic_group(2, N2, t, seed=i) for i, t in enumerate(("rigid", "affine"))}
        batches = {"rigid": groups["rigid"]["batch"], "affine": groups["affine"]["batch"],
                   "deform": fd.synthetic_deform_group(2, ND2, seed=0)["batch"]}
        cfg = tiny_config(**dict(STAGE_KW, **IN_KW))
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        randomize(m4, 3, 0.02)
        frozen = CS.FrozenStage.from_model(m4)
        self.assertEqual(frozen.model.compute_dtype, "float32")
        _assert_no_bn(self, frozen.model)
        with policy_scope("mixed_float16"):
            m2 = GroupRegLevelModel(2, cfg, gradient_checkpointing="full", lean_sampler=True,
                                    split_group_conv=True).build_model(2)
            m2.copy_weights_from(m4)
            self.assertEqual(m2.encoder.l2.compute_dtype, "float16")
            _assert_no_bn(self, m2)
            tr = Trainer(m2, O.make_optimizer("adam", lr=1e-3, amp=True),
                         {t: fd.FixedGroupSource(b) for t, b in batches.items()}, groups["rigid"]["S"],
                         cascade_config(amp=True, lean_sampler=True), cascade=frozen)
            fz_before = _vals(frozen.model.weights)
            tv_before = _vals(m2.trainable_variables)
            losses = {t: float(_step_until_finite(self, tr, t, batches[t])["loss"]) for t in RATIO3}
        self.assertTrue(_same(fz_before, frozen.model.weights))
        self.assertFalse(_same(tv_before, m2.trainable_variables))
        self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})
        _say("level-2 IN cascade mixed_float16: losses %s" % {k: "%.4g" % v for k, v in losses.items()})

    def test_level1_chain_float32_fine(self):
        """Level-1 chain (IN level-4 + level-2 frozen, float32) and a trainable level-1 stage with
        the float32_fine deform decoder under mixed_float16 (TINY_L1 widths, N 2)."""
        cfg = tiny_config(**dict(STAGE_KW, **dict(TS.TINY_L1, **IN_KW)))
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        randomize(m4, 3, 0.02)
        m2 = GroupRegLevelModel(2, cfg).build_model(2)
        m2.copy_weights_from(m4)
        randomize(m2, 4, 0.01)
        chain = CS.FrozenChain.from_models(m4, m2)
        for m in chain.models:
            self.assertEqual(m.compute_dtype, "float32")
            _assert_no_bn(self, m)
        groups = {t: synthetic_group(1, 2, t, seed=i) for i, t in enumerate(("rigid", "affine"))}
        batches = {"rigid": groups["rigid"]["batch"], "affine": groups["affine"]["batch"],
                   "deform": fd.synthetic_deform_group(1, 2, seed=0)["batch"]}
        with policy_scope("mixed_float16"):
            m1 = GroupRegLevelModel(1, cfg, gradient_checkpointing="full", lean_sampler=True,
                                    split_group_conv=True, deform_decoder_dtype="float32_fine").build_model(1)
            m1.copy_weights_from(chain.model2)
            self.assertEqual(m1.deform_decoder.d1.compute_dtype, "float32")
            self.assertEqual(m1.deform_decoder.d3.compute_dtype, "float16")
            _assert_no_bn(self, m1)
            tr = Trainer(m1, O.make_optimizer("adam", lr=1e-3, amp=True),
                         {t: fd.FixedGroupSource(b) for t, b in batches.items()}, groups["rigid"]["S"],
                         chain_config(task_ratio=dict(RATIO3), lean_losses=True), cascade=chain)
            fz_before = _vals([v for m in chain.models for v in m.weights])
            losses = {}
            t0 = time.time()
            for t in ("rigid", "affine"):
                losses[t] = float(_step_until_finite(self, tr, t, batches[t])["loss"])
            # The level-1 deform train step does not fit the local 6000 MB cap even at TINY_L1
            # (OOM, as the BN chain in tests/test_cascade.py, section comment of the 3-level
            # chain): traced only. Its IN layers are the same code as the executed levels 4 / 2.
            tr._steps["deform"].get_concrete_function()
        self.assertTrue(_same(fz_before, [v for m in chain.models for v in m.weights]))
        self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})
        _say("level-1 IN chain float32_fine mixed_float16: losses %s (%.0f s)"
             % ({k: "%.4g" % v for k, v in losses.items()}, time.time() - t0))


class TestInstanceNormParser(unittest.TestCase):
    """Plan item 6 c: BN recalibration is rejected for IN; 0 batches (r_cascade NORM=instance) parse."""

    def test_recal_flags(self):
        casc = ["--data", "x", "--level", "1", "--cascade-from", "x", "--norm", "instance",
                "--task-ratio", "rigid:1,affine:1,deform:2"]
        a = A.parse_args(casc + ["--bn-recal-batches", "0"])
        self.assertEqual((a.norm, a.bn_recal_batches, a.bn_recal_freeze), ("instance", 0, True))
        self.assertEqual(A.model_config_from_args(a).norm, "instance")
        A.parse_args(casc + ["--bn-recal-batches", "0", "--bn-recal-freeze", "true"])   # no-op without batches
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            A.parse_args(casc + ["--bn-recal-batches", "64"])
        self.assertIn("--bn-recal-batches > 0 needs --norm batch, got --norm instance", err.getvalue())


class TestInstanceNormStreaming(unittest.TestCase):
    """Plan item 6 d / D5: IN is per image, so the streamed group-mean engine stays exact (the
    float32 and fp16 bounds of tests/test_predict_streaming.py)."""

    _check_model = TS.TestStreamedLevel._check_model

    def tearDown(self):
        free_memory()

    def test_levels_4_2(self):
        for level in (4, 2):
            for policy, fp16 in (("float32", False), ("mixed_float16", True)):
                m = TS._model(level, policy=policy, norm="instance")
                self.assertEqual(L.batch_norm_layers(m), [])
                self._check_model("L%d IN %s" % (level, policy), m, TS._images(level), (1, 2, TS.N),
                                  ("deform", "rigid", "affine"), fp16=fp16)
                del m
                free_memory()

    def test_config_b_level2(self):
        for policy, fp16 in (("float32", False), ("mixed_float16", True)):
            m = TS._model(2, policy=policy, **IN_KW)
            self._check_model("L2 IN config B %s" % policy, m, TS._images(2), (2,), ("deform", "rigid"),
                              fp16=fp16)
            del m
            free_memory()


class TestInstanceNormPredict(unittest.TestCase):
    """Plan item 6 e: predict.py run on a TINY IN level-2 cascade (config B, N 3, 1 mm subjects)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgr_abl_in_pred_")
        cls.proc = os.path.join(cls.tmp, "proc")
        write_subjects(cls.proc, 3, textured=True, seed=2)
        cfg = tiny_config(**dict(STAGE_KW, **IN_KW))
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        randomize(m4, 3, 0.02)
        frozen = CS.FrozenStage.from_model(m4)
        m2 = GroupRegLevelModel(2, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
        m2.copy_weights_from(m4)
        randomize(m2, 4, 0.01)
        cls.ckpt = save_cascade_run(cls.tmp, "l2_in", m2, frozen)
        del m4, m2, frozen
        free_memory()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        free_memory()

    def test_run_tta_recenter(self):
        out = os.path.join(self.tmp, "out_tta_rc")
        args = CLI.parse_args(["run", "--checkpoint", self.ckpt, "--processed-dir", self.proc, "--out", out,
                               "--branches", "affine+deform", "--tta", "flip8", "--recenter", "1",
                               "--check-internal", "0"])
        code = PR.run_predict(args, configure_device=False)
        free_memory()
        with open(os.path.join(out, "predict.json")) as f:
            rec = json.load(f)
        self.assertEqual(code, 0, rec.get("message"))
        self.assertEqual((rec["tta"], rec["recenter"]["status"]), ("flip8", "ok"))
        self.assertEqual(rec["nonfinite"]["total"], 0)
        h = [x["mean_mm"] for x in rec["recenter"]["centrality_by_round"]]
        _say("IN level-2 cascade affine+deform flip8 + recenter 1: centrality %s" % h)

    def test_check_streaming(self):
        out = os.path.join(self.tmp, "check_in.json")
        code = PR.run_check_streaming(CLI.parse_args(
            ["check-streaming", "--checkpoint", self.ckpt, "--processed-dir", self.proc, "--out", out,
             "--n-subjects", "3"]), configure_device=False)
        free_memory()
        with open(out) as f:
            rec = json.load(f)
        self.assertEqual(code, 0, rec.get("message"))
        self.assertEqual(rec["status"], "pass")
        _say("check-streaming IN L2 N 3 deform: %s" % {k: "%.2e" % v for k, v in rec["differences"]["deform"].items()})


# ============================================================ from-scratch level 1 (FEAT-002)
# main.py --level 1 without --cascade-from with the r_level1_scratch.sh knobs (plan D6) at TINY_L1
# widths, N 2. Local limits (GTX 1080, 6000 MB cap): the level-1 deform train step and the level-1
# deform validation OOM even at TINY_L1 (measured: the warped one-hot of the chunked validation, the
# train step as in tests/test_cascade.py), so the main.py runs train the linear tasks with a
# delayed affine start (the same --task-start-step code path as deform:3000) with the deform
# decoder built, and the from-scratch level-1 deform step (float32_fine) is traced (built, not
# executed) in test_scratch_level1_trainer.
# --curriculum and --global-transform need the DALI pipeline (parser error with
# --overfit-one-group; DALI is not installed locally): checked at the parser level.
# The deform-only knobs (float32_fine decoder, checkpointing 'full', lean sampler / losses, pair
# recompute, val chunks) need 'deform' in --task-ratio, so they are in the parser and Trainer tests only.
SCRATCH_KNOBS = ["--level", "1", "--deform-decoder", "true", "--amp", "true", "--split-group-conv", "true",
                 "--optimizer", "adamw", "--weight-decay", "0.1", "--lr", "3e-4"]
SCRATCH_DEFORM_KNOBS = ["--deform-decoder-dtype", "float32_fine", "--gradient-checkpointing", "full", "--lean-sampler", "true", "--pair-recompute", "true",
                        "--lean-losses", "true", "--val-image-chunk", "1", "--val-pair-chunk", "1"]
SCRATCH_L1 = (["--overfit-one-group", "--overfit-synthetic",
               "--filters", "4,16,16,16,16,16", "--reductions", "16,8", "--dense", "64,32",
               "--decoder-widths", "8,8,8,4,4", "--group-size", "2",       # steps = --overfit-max-steps
               "--task-ratio", "rigid:1,affine:1", "--task-start-step", "affine:2", "--task-start-ramp-steps", "1",
               "--log-every", "1", "--val-png", "false"] + SCRATCH_KNOBS + COMMON)


class TestScratchLevel1Main(unittest.TestCase):
    """Plan item 8: the from-scratch level-1 training path through main.py, resumable."""

    def test_scratch_level1_parse(self):
        """The DALI-path recipe (d12hs, N 8 / 4, delayed deform, curriculum, delayed G, no cascade, no
        BN recal) parses at level 1 into the expected train config."""
        argv = (["--data", "x", "--steps", "16000", "--group-size", "8", "--deform-group-size", "4",
                 "--filters", "16,64,128,192,256,384", "--decoder-widths", "128,96,64,16,8",
                 "--group-config", "B", "--lambda-smooth", "15",
                 "--curriculum", "true", "--curriculum-ramp-steps", "3000", "--global-transform", "true",
                 "--global-start-step", "3000", "--global-ramp-steps", "3000"] + SCRATCH_KNOBS
                + SCRATCH_DEFORM_KNOBS + ["--task-ratio", "rigid:1,affine:1,deform:2", "--task-start-step", "deform:3000",
                   "--task-start-ramp-steps", "500"])
        a = A.parse_args(argv)
        self.assertEqual((a.level, a.cascade_from, a.bn_recal_batches), (1, None, 0))
        tc = A.train_config_from_args(a)
        self.assertEqual((tc.level, tc.group_size, tc.deform_group_size), (1, 8, 4))
        self.assertEqual(tc.task_start_steps, {"deform": 3000})
        self.assertEqual(tc.task_start_ramp_steps, 500)
        self.assertEqual((tc.lean_sampler, tc.lean_losses, tc.pair_recompute, tc.amp), (True, True, True, True))
        mc = A.model_config_from_args(a)
        self.assertEqual((mc.group_config, mc.norm, mc.filters), ("B", "batch", (16, 64, 128, 192, 256, 384)))
        a = A.parse_args(argv + ["--norm", "instance", "--group-config", "none"])
        self.assertEqual((A.model_config_from_args(a).norm, A.model_config_from_args(a).group_config),
                         ("instance", "none"))

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_abl_scr1_")
        self.addCleanup(tf.keras.mixed_precision.set_global_policy, "float32")      # --amp true

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        free_memory()

    def run_dir(self, name):
        return os.path.join(self.tmp, "results", name)

    def _records(self, name):
        with open(os.path.join(self.run_dir(name), "logs", "metrics.jsonl")) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_scratch_level1_and_resume(self):
        res = os.path.join(self.tmp, "results")
        base = SCRATCH_L1 + ["--results", res, "--run-name", "s1", "--val-every", "4"]
        t0 = time.time()
        self.assertIn(MAIN.main(base + ["--overfit-max-steps", "4"]), (0, 1))
        free_memory()
        d = self.run_dir("s1")
        with open(os.path.join(d, "config.json")) as f:
            cfg1 = json.load(f)
        self.assertNotIn("cascade", cfg1)
        self.assertIsNone(cfg1["args"]["cascade_from"])
        self.assertEqual((cfg1["args"]["level"], cfg1["args"]["bn_recal_batches"]), (1, 0))
        self.assertEqual(cfg1["model_config"]["norm"], "batch")
        self.assertEqual(cfg1["args"]["optimizer"], "adamw")
        recs = self._records("s1")
        with open(os.path.join(d, "overfit_summary.json")) as f:
            steps1 = json.load(f)["task_curve"]
        self.assertEqual(len(steps1), 4)
        self.assertEqual(steps1[:2], ["rigid", "rigid"])     # --task-start-step affine:2
        self.assertTrue(any(k.startswith("last-4") for k in os.listdir(os.path.join(d, "ckpt", "last"))))
        val4 = [r["data"] for r in recs if r.get("step") == 4 and "val_rigid/center_rms_mm" in r["data"]]
        self.assertTrue(val4)
        self.assertTrue(np.isfinite(val4[-1]["val_rigid/center_rms_mm"]))
        self.assertTrue(np.isfinite(val4[-1]["val_affine/center_rms_mm"]))
        self.assertFalse(any(r["data"].get("event") == "bn_recalibration" for r in recs))
        # Resume: same run name, more steps -> continues at step 4, config unchanged.
        self.assertIn(MAIN.main(base + ["--overfit-max-steps", "6"]), (0, 1))
        free_memory()
        with open(os.path.join(d, "config.json")) as f:
            cfg2 = json.load(f)
        self.assertEqual(cfg2["model_config"], cfg1["model_config"])
        with open(os.path.join(d, "overfit_summary.json")) as f:
            s = json.load(f)
        self.assertEqual((s["steps_run"], s["end_step"]), (2, 6))
        recs = self._records("s1")
        params = [r["data"] for r in recs if r.get("step") == "PARAMETER"]
        self.assertEqual([(p["resumed"], p["resumed_step"]) for p in params], [(False, 0), (True, 4)])
        self.assertTrue(any(k.startswith("last-6") for k in os.listdir(os.path.join(d, "ckpt", "last"))))
        _say("from-scratch level 1 (4 steps + resume to 6): %.0f s; steps/tasks of run 1 %s"
             % (time.time() - t0, steps1))

    def test_scratch_level1_instance_none(self):
        res = os.path.join(self.tmp, "results")
        argv = SCRATCH_L1 + ["--results", res, "--run-name", "s1in", "--val-every", "0",
                             "--norm", "instance", "--group-config", "none", "--overfit-max-steps", "3"]
        self.assertIn(MAIN.main(argv), (0, 1))
        free_memory()
        with open(os.path.join(self.run_dir("s1in"), "config.json")) as f:
            cfg = json.load(f)
        self.assertEqual((cfg["model_config"]["norm"], cfg["model_config"]["group_config"]), ("instance", "none"))
        with open(os.path.join(self.run_dir("s1in"), "overfit_summary.json")) as f:
            s = json.load(f)
        self.assertEqual(s["end_step"], 3)
        self.assertTrue(np.isfinite(s["final_val"]["val_rigid/center_rms_mm"]))
        self.assertTrue(np.isfinite(s["final_val"]["val_affine/center_rms_mm"]))

    def test_scratch_level1_trainer(self):
        """The from-scratch level-1 Trainer (cascade None, three tasks, float32_fine, AMP, lean
        sampler / losses, pair recompute, min pairs n): linear steps applied, the deform step traced
        (see the section comment), for configs B (batch norm) and none + instance norm."""
        groups = {t: synthetic_group(1, 2, t, seed=i) for i, t in enumerate(("rigid", "affine"))}
        batches = {"rigid": groups["rigid"]["batch"], "affine": groups["affine"]["batch"],
                   "deform": fd.synthetic_deform_group(1, 2, seed=0)["batch"]}
        for kw in (dict(group_config="B"), dict(group_config="none", norm="instance")):
            with self.subTest(**kw), policy_scope("mixed_float16"):
                cfg = tiny_config(**dict(STAGE_KW, **dict(TS.TINY_L1, **kw)))
                m1 = GroupRegLevelModel(1, cfg, gradient_checkpointing="full", lean_sampler=True,
                                        split_group_conv=True, deform_decoder_dtype="float32_fine").build_model(1)
                tr = Trainer(m1, O.make_optimizer("adamw", lr=3e-4, weight_decay=0.1, amp=True),
                             {t: fd.FixedGroupSource(b) for t, b in batches.items()}, groups["rigid"]["S"],
                             chain_config(task_ratio=dict(RATIO3), lean_losses=True, pair_recompute=True,
                                          min_pairs="n", task_start_steps={"deform": 2},
                                          task_start_ramp_steps=1))
                self.assertEqual(tr._n_stages, 0)
                for t in ("rigid", "affine"):
                    _step_until_finite(self, tr, t, batches[t])
                tr._steps["deform"].get_concrete_function()
                self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})
                del tr, m1
                free_memory()


if __name__ == "__main__":
    unittest.main()
