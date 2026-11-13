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
Measured values are printed to stderr.
"""
from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory limit)
from tests.predict_helpers import save_cascade_run, write_subjects
from tests.runtime_helpers import TINY_ARGS, free_memory, synthetic_group, tiny_config
from tests.test_cascade import N2, ND2, STAGE_CKPT, STAGE_KW, cascade_config, randomize
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

import main as MAIN
from inference import cli as CLI
from inference import predict as PR
from models import cascade as CS
from models.config import ModelConfig
from models.group_net import GroupRegLevelModel
from runtime import checkpoint as C
from runtime import fixed_data as fd
from runtime import optimizers as O
from runtime.run import Trainer

COMMON = ["--log-backend", "jsonl", "--quiet", "true", "--seed", "0"]


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


if __name__ == "__main__":
    unittest.main()
