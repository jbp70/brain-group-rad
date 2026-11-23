"""Global average pooling in the rigid/affine decoder (``--linear-pool gap``, linear-gap;
`.agents/tasks/linear-gap/report.md`).

'gap' replaces the 1x1 reductions + flatten of `LinearDecoder` by a float32 spatial mean of the
encoder bottleneck [N, 5, 7, 6, f5] -> [N, f5]; dense blocks and heads unchanged. Tests:
* TestGapModel: d12hs config B builds at levels 4 / 2 / 1 (no reduction variables, dense_0 kernel
  [f5, 2048]), trainable counts == `width_presets.count_params` per part at every level, flatten
  vs gap counts printed; the pooled vector is the float32 mean (also under mixed_float16, where
  it is cast to float16); --dense-group-concat-first with gap; parser / config round trip.
* TestGapTraining: level-4 three-task Trainer steps under mixed_float16 with the final recipe's
  knobs (ckpt 'full', lean sampler, split group conv, config B); a linear step moves the encoder.
* TestGapCascade: level-2 cascade from a gap level-4 source (rigid, affine, deform steps; frozen
  stage unchanged); flatten / gap mixes rejected (`check_cascade_config`, `FrozenChain`).
* TestGapMain: main.py overfit-synthetic level 4 with --linear-pool gap -> --cascade-from at
  level 2 (config.json round trip), flatten at level 2 refused, --init-weights gap -> gap accepted
  and gap -> flatten refused.
* TestGapPredict: checkpoint round trip through `inference.loader.load_model` (bit-identical A),
  StreamedLevel == model for gap (levels 4 / 2, float32 and mixed_float16), predict.py run on a
  gap level-2 cascade (rigid; affine+deform default, --levels 2, --tta flip8 + --recenter 1) and
  check-streaming.
Default 'flatten' bit-identity is covered by the goldens (tests.test_trainer, test_predict_tta_golden).
"""
from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory limit)
from tests import test_ablation_configs as AB
from tests import test_predict_streaming as TS
from tests.predict_helpers import save_cascade_run, save_level4_run, write_subjects
from tests.runtime_helpers import TINY_ARGS, free_memory, synthetic_group, tiny_config
from tests.test_cascade import N2, ND2, STAGE_CKPT, STAGE_KW, cascade_config, randomize
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf

import main as MAIN
from inference import cli as CLI
from inference import loader as LD
from inference import predict as PR
from models import cascade as CS
from models import width_presets as W
from models.config import ModelConfig
from models.group_net import GroupRegLevelModel
from models.precision import policy_scope
from runtime import args as A
from runtime import checkpoint as C
from runtime import fixed_data as fd
from runtime import optimizers as O
from runtime.run import Trainer, TrainConfig
from runtime.schedulers import ScheduleConfig

D12HS = dict(filters=(16, 64, 128, 192, 256, 384), decoder_widths=(128, 96, 64, 16, 8))
GAP = dict(linear_pool="gap", group_config="B")
COMMON = ["--log-backend", "jsonl", "--quiet", "true", "--seed", "0"]


def _say(msg):
    print("\n[linear-gap] " + msg, file=sys.stderr)


def _vals(variables):
    return [v.numpy().copy() for v in variables]


def _same(before, variables):
    return all(np.array_equal(a, v.numpy()) for a, v in zip(before, variables))


def _parts(model):
    c = model.count_params_by_part()
    return {"encoder": c["encoder"], "linear_decoder": c["linear_decoder"],
            "deform_decoder": c.get("deform_decoder", 0), "total": c["trainable"]}


class TestGapModel(unittest.TestCase):
    def tearDown(self):
        tf.keras.mixed_precision.set_global_policy("float32")
        free_memory()

    def test_build_levels_d12hs(self):
        """Config B, d12hs widths: gap at levels 4 / 2 / 1 (N 1, eval forward) has the same
        variables at every level, no reductions, dense_0 kernel [384, 2048]; counts == the TF-free
        mirror exactly. The deform decoder is built at level 4 only (level-independent variables;
        the level-1 deform forward does not fit the local 6000 MB cap at d12hs)."""
        rows = {}
        for pool in ("flatten", "gap"):
            cfg = ModelConfig(group_config="B", linear_pool=pool, deform_decoder=True, **D12HS)
            m = GroupRegLevelModel(4, cfg).build_model(1)
            got, want = _parts(m), W.count_params(cfg)
            self.assertEqual(got, dict(want), pool)
            rows[pool] = got
            if pool == "gap":
                self.assertEqual(m.linear_decoder.reductions, [])
                self.assertIsNone(m.linear_decoder.flatten)
                self.assertFalse([v.name for v in m.weights if "reduction" in v.name])
                self.assertEqual(tuple(m.linear_decoder.dense_blocks[0].dense.kernel.shape), (384, 2048))
            else:
                self.assertEqual(len(m.linear_decoder.reductions), 4)
                self.assertEqual(tuple(m.linear_decoder.dense_blocks[0].dense.kernel.shape), (5040, 2048))
            del m
            free_memory()
        lin_cfg = ModelConfig(group_config="B", linear_pool="gap", **D12HS)
        names = None
        for level in (4, 2, 1):
            m = GroupRegLevelModel(level, lin_cfg).build_model(1)
            got = _parts(m)
            self.assertEqual(got, dict(W.count_params(lin_cfg)), level)
            lv = [(GroupRegLevelModel._local_name(v), tuple(v.shape)) for v in m.weights]
            if names is None:
                names = lv
            self.assertEqual(lv, names, level)
            x = tf.zeros((1,) + m.input_shape_level + (1,))
            out = m(x, task="affine", training=False)
            self.assertEqual(tuple(out["A"].shape), (1, 4, 4))
            self.assertTrue(np.isfinite(out["A"].numpy()).all())
            del m, out
            free_memory()
        _say("d12hs config B trainable parameters (all levels): flatten %s | gap %s | linear decoder "
             "%d -> %d (-%d)" % (rows["flatten"], rows["gap"], rows["flatten"]["linear_decoder"],
                                  rows["gap"]["linear_decoder"],
                                  rows["flatten"]["linear_decoder"] - rows["gap"]["linear_decoder"]))

    def test_pool_is_float32_mean(self):
        rng = np.random.default_rng(0)
        b = rng.normal(0.0, 30.0, size=(3, 5, 7, 6, 16)).astype(np.float32)
        ref = b.astype(np.float64).mean(axis=(1, 2, 3))
        dec = GroupRegLevelModel(4, tiny_config(**GAP)).linear_decoder
        got = dec._global_average_pool(tf.constant(b))
        self.assertEqual(got.dtype, tf.float32)
        np.testing.assert_allclose(got.numpy(), ref, rtol=0, atol=1e-4)
        h = tf.cast(tf.constant(b), tf.float16)
        got16 = dec._global_average_pool(h)
        self.assertEqual(got16.dtype, tf.float16)
        ref16 = h.numpy().astype(np.float64).mean(axis=(1, 2, 3))
        # float32 accumulation: only the final float16 rounding (rel 2^-11) remains.
        np.testing.assert_allclose(got16.numpy().astype(np.float64), ref16, rtol=1e-3, atol=1e-3)
        with policy_scope("mixed_float16"):
            m = GroupRegLevelModel(4, tiny_config(**GAP)).build_model(2)
            x = TS._images(4, 2)
            out = m(tf.constant(x), task="affine", training=True)
            self.assertEqual(out["A"].dtype, tf.float32)
            self.assertTrue(np.isfinite(out["A"].numpy()).all())
            self.assertEqual(out["bottleneck"].dtype, tf.float16)

    def test_dense_group_concat_first(self):
        for g in ("A", "B"):
            cfg = tiny_config(linear_pool="gap", group_config=g, dense_group_concat_first=True)
            m = GroupRegLevelModel(4, cfg).build_model(2)
            d0 = m.linear_decoder.dense_blocks[0]
            self.assertTrue(d0.group)
            self.assertEqual(tuple(d0.dense.kernel.shape), (2 * cfg.filters[-1], cfg.dense[0]))
            out = m(tf.constant(TS._images(4, 2)), task="rigid", training=False)
            self.assertTrue(np.isfinite(out["A"].numpy()).all())
            self.assertEqual(_parts(m), dict(W.count_params(cfg)))
            del m
            free_memory()
        with self.assertRaises(ValueError):                 # unchanged rule for none / E
            ModelConfig(linear_pool="gap", group_config="none", dense_group_concat_first=True)

    def test_parser(self):
        base = ["--data", "x", "--level", "4"]
        a = A.parse_args(base + ["--linear-pool", "gap", "--group-config", "B"])
        cfg = A.model_config_from_args(a)
        self.assertEqual((a.linear_pool, cfg.linear_pool), ("gap", "gap"))
        self.assertEqual(A.args_to_dict(a)["linear_pool"], "gap")
        self.assertEqual(A.model_config_from_args(A.parse_args(base)).linear_pool, "flatten")
        self.assertEqual(ModelConfig.from_dict(cfg.to_dict()), cfg)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            A.parse_args(base + ["--linear-pool", "max"])
        self.assertIn("--linear-pool", err.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            A.parse_args(["--help"])
        self.assertIn("--linear-pool", out.getvalue())


class TestGapTraining(unittest.TestCase):
    def tearDown(self):
        tf.keras.mixed_precision.set_global_policy("float32")
        free_memory()

    def test_level4_three_task_amp(self):
        n = 2
        groups = {t: synthetic_group(4, n, t, seed=i) for i, t in enumerate(("rigid", "affine"))}
        batches = {"rigid": groups["rigid"]["batch"], "affine": groups["affine"]["batch"],
                   "deform": fd.synthetic_deform_group(4, n, seed=0)["batch"]}
        cfg = tiny_config(seed=1, deform_decoder=True, integration_steps=3, **GAP)
        tcfg = TrainConfig(level=4, group_size=n, deform_group_size=n, steps=100, val_every=0, log_every=0,
                           task_ratio=dict(AB.RATIO3), seed=0, amp=True, lean_sampler=True,
                           n_classes=AB.N_CLASSES,
                           schedule=ScheduleConfig("constant", base_lr=1e-3, total_steps=100, warmup_steps=0))
        with policy_scope("mixed_float16"):
            model = GroupRegLevelModel(4, cfg, gradient_checkpointing="full", lean_sampler=True,
                                       split_group_conv=True).build_model(n)
            self.assertEqual(model.linear_decoder.reductions, [])
            head = model.deform_decoder.svf_head
            head.kernel.assign(np.random.default_rng(0).normal(0, 1e-3, head.kernel.shape).astype(np.float32))
            for h in (model.linear_decoder.rigid_head, model.linear_decoder.scale_shear_head):
                h.kernel.assign(np.random.default_rng(1).normal(0, 1e-3, h.kernel.shape).astype(np.float32))
            tr = Trainer(model, O.make_optimizer("adam", lr=1e-3, amp=True),
                         {t: fd.FixedGroupSource(b) for t, b in batches.items()}, groups["rigid"]["S"], tcfg)
            enc = model.encoder.trainable_variables
            dense = model.linear_decoder.dense_blocks[0].trainable_variables
            losses = {}
            for t in ("rigid", "affine", "deform"):
                enc_before, dense_before = _vals(enc), _vals(dense)
                losses[t] = float(AB._step_until_finite(self, tr, t, batches[t])["loss"])
                self.assertFalse(_same(enc_before, enc), t)          # gradient reaches the encoder
                if t != "deform":
                    self.assertFalse(_same(dense_before, dense), t)
        self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})
        _say("level-4 gap mixed_float16 three-task: losses %s" % {k: "%.4g" % v for k, v in losses.items()})


class TestGapCascade(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        groups = {t: synthetic_group(2, N2, t, seed=i) for i, t in enumerate(("rigid", "affine"))}
        cls.S = groups["rigid"]["S"]
        cls.batches = {"rigid": groups["rigid"]["batch"], "affine": groups["affine"]["batch"],
                       "deform": fd.synthetic_deform_group(2, ND2, seed=0)["batch"]}

    def tearDown(self):
        free_memory()

    def test_cascade_steps(self):
        cfg = tiny_config(**dict(STAGE_KW, **GAP))
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        randomize(m4, 3, 0.02)
        m2 = GroupRegLevelModel(2, cfg, gradient_checkpointing=STAGE_CKPT).build_model(2)
        m2.copy_weights_from(m4)
        frozen = CS.FrozenStage.from_model(m4)
        self.assertEqual(frozen.model.config.linear_pool, "gap")
        tr = Trainer(m2, O.make_optimizer("adam", lr=1e-3),
                     {t: fd.FixedGroupSource(b) for t, b in self.batches.items()}, self.S,
                     cascade_config(), cascade=frozen)
        fz_before = _vals(frozen.model.weights)
        tv_before = _vals(m2.trainable_variables)
        losses = {}
        for t in ("rigid", "affine", "deform"):
            res = tr.train_step(t, self.batches[t])
            self.assertTrue(res["finite"], t)
            losses[t] = float(res["loss"])
        self.assertTrue(_same(fz_before, frozen.model.weights))
        self.assertFalse(_same(tv_before, m2.trainable_variables))
        self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})
        _say("level-2 gap cascade: losses %s" % {k: "%.4g" % v for k, v in losses.items()})

    def test_mixed_pools_rejected(self):
        gap, flat = (tiny_config(**dict(STAGE_KW, group_config="B", linear_pool=p)) for p in ("gap", "flatten"))
        for mine, theirs in ((gap, flat), (flat, gap)):
            with self.assertRaises(ValueError) as cm:
                C.check_cascade_config(mine, theirs)
            self.assertIn("linear_pool", str(cm.exception))
        C.check_cascade_config(gap, gap.replace(seed=7))
        m4 = GroupRegLevelModel(4, flat).build_model(1)
        m2 = GroupRegLevelModel(2, gap).build_model(1)
        with self.assertRaises(ValueError) as cm:
            CS.FrozenChain.from_models(m4, m2)
        self.assertIn("linear_pool", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            m2.copy_weights_from(m4)
        self.assertIn("linear_pool", str(cm.exception))


class TestGapMain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_gap_main_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        free_memory()

    def run_dir(self, name):
        return os.path.join(self.tmp, "results", name)

    def test_level4_to_level2_and_warm_start(self):
        res = os.path.join(self.tmp, "results")
        common = ["--overfit-one-group", "--overfit-synthetic", "--group-size", "4", "--task-ratio", "rigid:1",
                  "--deform-decoder", "true", "--log-every", "1", "--val-every", "0", "--val-png", "false",
                  "--results", res, "--overfit-max-steps", "1", "--group-config", "B"] + TINY_ARGS + COMMON
        gap = common + ["--linear-pool", "gap"]
        self.assertIn(MAIN.main(gap + ["--level", "4", "--run-name", "src4"]), (0, 1))
        free_memory()
        src = os.path.join(self.run_dir("src4"), "ckpt", "last")
        self.assertIn(MAIN.main(gap + ["--level", "2", "--cascade-from", src, "--run-name", "c2"]), (0, 1))
        free_memory()
        self.assertIn(MAIN.main(gap + ["--level", "4", "--init-weights", src, "--run-name", "ws4"]), (0, 1))
        free_memory()
        for name in ("src4", "c2", "ws4"):
            with open(os.path.join(self.run_dir(name), "config.json")) as f:
                rec = json.load(f)
            self.assertEqual(rec["model_config"]["linear_pool"], "gap", name)
            self.assertEqual(rec["args"]["linear_pool"], "gap", name)
            self.assertEqual(ModelConfig.from_dict(rec["model_config"]).linear_pool, "gap", name)
            self.assertNotIn("linear_decoder/reduction_0", rec.get("param_counts") or {}, name)
        with open(os.path.join(self.run_dir("c2"), "config.json")) as f:
            rec = json.load(f)
        self.assertEqual((rec["init_weights"]["via"], rec["init_weights"]["mode"]), ("cascade", "full"))
        with open(os.path.join(self.run_dir("ws4"), "config.json")) as f:
            self.assertEqual(json.load(f)["init_weights"]["mode"], "full")
        with open(os.path.join(self.run_dir("c2"), "overfit_summary.json")) as f:
            s = json.load(f)
        self.assertTrue(np.isfinite(s["final_val"]["val_rigid/center_rms_mm"]))
        # flatten at level 2 from a gap source: refused, naming the field.
        with self.assertRaises(SystemExit) as cm:
            MAIN.main(common + ["--level", "2", "--cascade-from", src, "--run-name", "bad2"])
        self.assertIn("linear_pool", str(cm.exception.code))
        free_memory()
        # flatten warm-started from a gap source: refused (not a decoder-only field).
        with self.assertRaises((ValueError, SystemExit)) as cm:
            MAIN.main(common + ["--level", "4", "--init-weights", src, "--run-name", "bad4"])
        exc = cm.exception
        self.assertIn("linear_pool", str(exc.code if isinstance(exc, SystemExit) else exc))


class TestGapPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgr_gap_pred_")
        cls.proc = os.path.join(cls.tmp, "proc")
        write_subjects(cls.proc, 3, textured=True, seed=2)
        cfg = tiny_config(**dict(STAGE_KW, **GAP))
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        randomize(m4, 3, 0.02)
        frozen = CS.FrozenStage.from_model(m4)
        m2 = GroupRegLevelModel(2, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
        m2.copy_weights_from(m4)
        randomize(m2, 4, 0.01)
        cls.ckpt = save_cascade_run(cls.tmp, "l2_gap", m2, frozen)
        del m4, m2, frozen
        free_memory()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        free_memory()

    def tearDown(self):
        free_memory()

    def test_checkpoint_round_trip(self):
        cfg = tiny_config(seed=1, **GAP)
        prefix, model = save_level4_run(self.tmp, "l4_gap", cfg, seed=5)
        loaded = LD.load_model(prefix, "rigid", precision="checkpoint", device="gpu")
        self.assertEqual(loaded.model.config, model.config)
        self.assertEqual(loaded.model.config.linear_pool, "gap")
        x = tf.constant(TS._images(4, 3))
        for task in ("rigid", "affine"):
            a = model(x, task=task, training=False)["A_raw"].numpy()
            b = loaded.model(x, task=task, training=False)["A_raw"].numpy()
            np.testing.assert_array_equal(a, b, task)
        del loaded, model

    def test_streaming_equivalence(self):
        chk = TS.TestStreamedLevel._check_model
        for level in (4, 2):
            for policy, fp16 in (("float32", False), ("mixed_float16", True)):
                m = TS._model(level, policy=policy, **GAP)
                self.assertEqual(m.linear_decoder.reductions, [])
                worst = chk(self, "L%d gap %s" % (level, policy), m, TS._images(level), (1, 2, TS.N),
                            ("rigid", "affine", "deform"), fp16=fp16)
                _say("streaming L%d gap %s: worst %s" % (level, policy,
                                                         {k: "%.2e" % v for k, v in worst.items()}))
                del m
                free_memory()

    def _run(self, name, branches, *extra):
        out = os.path.join(self.tmp, "out_" + name)
        args = CLI.parse_args(["run", "--checkpoint", self.ckpt, "--processed-dir", self.proc, "--out", out,
                               "--branches", branches] + list(extra))
        code = PR.run_predict(args, configure_device=False)
        free_memory()
        with open(os.path.join(out, "predict.json")) as f:
            return code, json.load(f)

    def test_run_variants(self):
        code, rec = self._run("rigid", "rigid", "--check-internal", "1")
        self.assertEqual(code, 0, rec.get("message"))
        self.assertEqual(rec["nonfinite"]["total"], 0)
        self.assertTrue(rec["internal_check"]["passed"])
        code, rec = self._run("default", "affine+deform", "--check-internal", "1")
        self.assertEqual(code, 0, rec.get("message"))
        self.assertEqual(rec["nonfinite"]["total"], 0)
        self.assertTrue(rec["internal_check"]["passed"])
        code, rec = self._run("lv2", "affine+deform", "--levels", "2", "--allow-untrained-skip",
                              "--check-internal", "0")
        self.assertEqual(code, 0, rec.get("message"))
        self.assertEqual(rec["nonfinite"]["total"], 0)
        code, rec = self._run("tta_rc", "affine+deform", "--tta", "flip8", "--recenter", "1",
                              "--check-internal", "0")
        self.assertEqual(code, 0, rec.get("message"))
        self.assertEqual((rec["tta"], rec["recenter"]["status"]), ("flip8", "ok"))
        self.assertEqual(rec["nonfinite"]["total"], 0)
        code, rec = self._run("tta_rigid", "rigid", "--tta", "flip8", "--check-internal", "0")
        self.assertEqual(code, 0, rec.get("message"))
        self.assertEqual(rec["nonfinite"]["total"], 0)

    def test_check_streaming(self):
        out = os.path.join(self.tmp, "check_gap.json")
        code = PR.run_check_streaming(CLI.parse_args(
            ["check-streaming", "--checkpoint", self.ckpt, "--processed-dir", self.proc,
             "--out", out, "--n-subjects", "3"]), configure_device=False)
        with open(out) as f:
            rec = json.load(f)
        self.assertEqual(code, 0, rec.get("message"))
        self.assertEqual(rec["status"], "pass")
        _say("check-streaming gap L2 N 3: %s" % rec["differences"])


if __name__ == "__main__":
    unittest.main()
