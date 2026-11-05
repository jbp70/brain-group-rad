"""Post-training synthetic-contrast evaluation (synth-contrast design section 8, FEAT-005).

Config round trip (int dict keys of *_parsed, TrainConfig equality, old configs without synth_*),
the 11 evaluation presets, the ``group_images_fn`` hooks of the validation-set builders (default
None = today's sets), per-preset set names / identity caches, the sanity-gate comparison, and a
tiny end-to-end `eval_synth.py` run on a level-4 run directory (shared sets, CSV, run directory
unchanged, failed sanity gate -> exit 5).
"""
import contextlib
import csv
import io
import json
import os
import shutil
import tempfile
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.runtime_helpers import TINY_ARGS, fine_labels_from_train18, free_memory, write_fake_export
import numpy as np
import tensorflow as tf

import eval_synth as ES
import main as MAIN
from data_loading import params as P
from data_loading import synth as SY
from preprocessing import luts
from runtime import args as A
from runtime import checkpoint as C
from runtime import eval_run as E
from runtime import fixed_data as fd

COMMON = ["--log-backend", "jsonl", "--quiet", "true", "--seed", "0"]
SHAPE = P.LEVEL_SHAPE[4]


def _write_run_config(run_dir, args, drop=()):
    os.makedirs(run_dir, exist_ok=True)
    d = A.args_to_dict(args)
    for k in drop:
        d.pop(k, None)
    C.write_config(os.path.join(run_dir, "config.json"), A.model_config_from_args(args), d)


class TestConfigRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="evalsyn_cfg_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_int_keys_and_train_config(self):
        args = A.parse_args(["--data", "x", "--level", "2", "--cascade-from", "src", "--task-ratio",
                             "rigid:1,deform:1", "--cascade-skip-deform", "4:0.15", "--group-size", "4",
                             "--deform-group-size", "2"])
        self.assertEqual(args.cascade_skip_stage_parsed, {"rigid": {}, "deform": {4: 0.15}})
        run = os.path.join(self.tmp, "run")
        _write_run_config(run, args)
        with open(os.path.join(run, "config.json")) as f:
            self.assertEqual(json.load(f)["args"]["cascade_skip_stage_parsed"]["deform"], {"4": 0.15})
        ns, raw = E.load_run_config(run)
        self.assertEqual(ns.cascade_skip_stage_parsed, {"rigid": {}, "deform": {4: 0.15}})
        self.assertEqual(E.train_config_for_eval(ns), A.train_config_from_args(args))
        self.assertIsNone(E.train_config_for_eval(ns).synth)
        self.assertIsNone(E.train_config_for_eval(ns).val_png_dir)
        from models.config import ModelConfig
        self.assertEqual(ModelConfig.from_dict(raw["model_config"]), A.model_config_from_args(args))

    def test_synth_on_and_old_config(self):
        args = A.parse_args(["--data", "x", "--level", "4", "--task-ratio", "rigid:1", "--group-size", "4",
                             "--synth-prob", "0.5", "--intensity-aug-device", "tf"])
        run = os.path.join(self.tmp, "on")
        _write_run_config(run, args)
        ns, _ = E.load_run_config(run)
        cfg = E.train_config_for_eval(ns)
        self.assertEqual(cfg, A.train_config_from_args(args))
        self.assertEqual(cfg.synth.prob, 0.5)
        old = os.path.join(self.tmp, "old")                     # a config written before synth_*
        _write_run_config(old, args, drop=[k for k in vars(args) if k.startswith("synth_")])
        ns_old, raw = E.load_run_config(old)
        self.assertFalse(any(k.startswith("synth_") for k in raw["args"]))
        self.assertIsNone(E.train_config_for_eval(ns_old).synth)

    def test_missing_config(self):
        with self.assertRaises(E.EvalError):
            E.load_run_config(os.path.join(self.tmp, "nope"))


class TestPresets(unittest.TestCase):
    CFG = P.SynthConfig()

    def test_names(self):
        self.assertEqual(SY.PRESET_NAMES, ES.PRESET_NAMES)
        self.assertEqual(len(SY.PRESET_NAMES), 11)
        for name, spec in SY.EVAL_PRESETS.items():
            if spec["kind"] == "named":
                self.assertEqual(len(spec["class_means"]), P.N_TRAIN_LABELS, name)
                self.assertEqual(spec["class_means"][0], 0.0)

    def test_shapes_and_determinism(self):
        ref = SY.draw_synth_params(tf.random.Generator.from_seed(0), 3, P.SynthConfig(prob=1.0))
        for name in SY.PRESET_NAMES:
            with self.subTest(name):
                a = SY.preset_params(name, 3, 7, self.CFG, 4)
                b = SY.preset_params(name, 3, 7, self.CFG, 4)
                c = SY.preset_params(name, 3, 8, self.CFG, 4)
                self.assertEqual(sorted(a), sorted(ref))
                for k in ref:
                    self.assertEqual(a[k].shape, ref[k].shape, k)
                    self.assertEqual(a[k].dtype, ref[k].dtype, k)
                    np.testing.assert_array_equal(a[k].numpy(), b[k].numpy(), k)
                self.assertTrue(bool(a["flag"].numpy()))
                self.assertFalse(np.array_equal(a["jitter"].numpy(), c["jitter"].numpy()))
                self.assertEqual(float(a["mean"].numpy()[0]), 0.0)

    def test_random_uses_training_distribution(self):
        g = luts.synthfine_to_train18_table().astype(np.int64)
        cortex = luts.synthfine_cortex_ids()
        other = np.setdiff1d(np.arange(1, len(luts.LUTS["synthfine"]["map"])), cortex)
        for k in range(4):
            prm = SY.preset_params("random_%d" % k, 2, 0, self.CFG, 4)
            mean, cls = prm["mean"].numpy(), prm["cls_mean"].numpy()
            dev = np.abs(mean - np.clip(cls[g], 0, 1))
            self.assertLessEqual(float(dev[other].max()), 0.1 + 1e-6)
            self.assertLessEqual(float(dev[cortex].max()), 0.25 + 1e-6)
            self.assertGreater(float(dev[cortex].max()), 0.1)        # cortex uses hier_dev_cortex 0.25
        r0 = SY.preset_params("random_0", 2, 0, self.CFG, 4)["mean"].numpy()
        r1 = SY.preset_params("random_1", 2, 0, self.CFG, 4)["mean"].numpy()
        self.assertFalse(np.array_equal(r0, r1))

    def test_pet_parcel_overrides(self):
        tau = SY.preset_params("tau_like", 2, 0, self.CFG, 4)
        mean = tau["mean"].numpy()
        ent = SY.parcel_fine_ids("entorhinal")
        self.assertEqual(len(ent), 2)
        np.testing.assert_allclose(mean[ent], 0.9)
        np.testing.assert_allclose(mean[SY.parcel_fine_ids("precuneus")], 0.30)   # tau cortex class
        self.assertTrue(np.all(np.abs(tau["fwhm_mm"].numpy() - 6.0) <= 0.6 + 1e-6))   # 6 mm x U(0.9, 1.1)
        self.assertTrue(bool(tau["pet_family"].numpy()))
        amy = SY.preset_params("amyloid_pos", 2, 0, self.CFG, 4)["mean"].numpy()
        np.testing.assert_allclose(amy[SY.parcel_fine_ids("precuneus")], 0.9)
        g = luts.synthfine_to_train18_table()
        self.assertTrue(np.all(amy[g == 6][1:] < amy[g == 1][1:].min()))  # cerebellar cortex < WM
        fdg = SY.preset_params("fdg_like", 2, 0, self.CFG, 4)
        self.assertTrue(np.all(np.abs(fdg["fwhm_mm"].numpy() - 5.0) <= 0.5 + 1e-6))
        t2 = SY.preset_params("t2_like", 2, 0, self.CFG, 4)
        self.assertFalse(bool(t2["pet_family"].numpy()))
        self.assertTrue(np.all(t2["fwhm_mm"].numpy() <= 3.0 * 1.1 + 1e-6))
        with self.assertRaises(ValueError):
            SY.preset_params("nope", 2, 0, self.CFG, 4)

    def test_synthesize_with_preset(self):
        rng = np.random.default_rng(0)
        fine = []
        for _ in range(2):
            _, lab = fd.synthetic_label_phantom(SHAPE, rng)
            fine.append(fine_labels_from_train18(lab, rng))
        fine = np.stack(fine)[..., None]
        with tf.device("/cpu:0"):
            for name in ("t1_like", "tau_like", "random_2"):
                prm = SY.preset_params(name, 2, 0, self.CFG, 4)
                x = SY.synthesize_group(fine, prm, self.CFG, 4).numpy()
                self.assertEqual(x.shape, (2,) + tuple(SHAPE) + (1,))
                self.assertTrue(np.all(np.isfinite(x)) and x.min() >= 0 and x.max() <= 1)
                self.assertTrue(np.all(x[fine == 0] == 0))           # re-masked (named presets)


class TestHooks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        imgs, labs = [], []
        for _ in range(5):
            img, lab = fd.synthetic_label_phantom(SHAPE, rng)
            imgs.append(img)
            labs.append(lab)
        self.images = np.stack(imgs)[..., None].astype(np.float32)
        self.labels = np.stack(labs)[..., None]

    def assert_sets_equal(self, a, b, keys):
        self.assertEqual(len(a.groups), len(b.groups))
        for ga, gb in zip(a.groups, b.groups):
            for k in keys:
                np.testing.assert_array_equal(ga[k], gb[k], k)

    def test_linear_default_and_hook(self):
        gt = {"rotation_deg": 20.0, "translation_mm": 5.0}
        ref = fd.build_validation_sets(self.images, ["rigid", "affine"], 4, 3, None, None, seed=5,
                                       global_transform=gt)
        none = fd.build_validation_sets(self.images, ["rigid", "affine"], 4, 3, None, None, seed=5,
                                        global_transform=gt, group_images_fn=None)
        ident = fd.build_validation_sets(self.images, ["rigid", "affine"], 4, 3, None, None, seed=5,
                                         global_transform=gt, group_images_fn=lambda g, idx: self.images[idx])
        self.assertEqual(sorted(ref), ["affine", "affine_g", "rigid", "rigid_g"])
        for name in ref:
            self.assert_sets_equal(ref[name], none[name], ("images", "T", "T_target", "A_star", "index"))
            self.assert_sets_equal(ref[name], ident[name], ("images", "T", "T_target", "A_star", "index"))
        calls = []

        def const(g, idx):
            calls.append((g, tuple(idx)))
            return np.full((len(idx),) + tuple(SHAPE) + (1,), 0.5, np.float32)

        rep = fd.ValidationSet.build(self.images, "rigid", 4, 3, seed=5, group_images_fn=const)
        self.assert_sets_equal(ref["rigid"], rep, ("T", "T_target", "A_star", "index"))
        self.assertEqual([c[0] for c in calls], list(range(len(rep.groups))))
        for (g, idx), gr, gp in zip(calls, ref["rigid"].groups, rep.groups):
            self.assertEqual(idx, tuple(gr["index"]))          # the hook gets the group's cases
            self.assertFalse(np.array_equal(gr["images"], gp["images"]))

    def test_deform_default_and_hook(self):
        ref = fd.DeformValidationSet.build(self.images, self.labels, 4, 2, seed=5)
        none = fd.DeformValidationSet.build(self.images, self.labels, 4, 2, seed=5, group_images_fn=None)
        self.assert_sets_equal(ref, none, ("images", "labels", "index"))
        rep = fd.DeformValidationSet.build(self.images, self.labels, 4, 2, seed=5,
                                           group_images_fn=lambda g, idx: 1.0 - self.images[idx])
        self.assert_sets_equal(ref, rep, ("labels", "index"))
        for gr, gp in zip(ref.groups, rep.groups):
            np.testing.assert_array_equal(gp["images"], 1.0 - gr["images"])
            self.assertEqual(gp["images"].dtype, np.float32)


class TestGateCompare(unittest.TestCase):
    def test_tolerances(self):
        logged = {"val_rigid/pair_rms_mm": 3.0, "val_deform/template_dice": 0.8, "val_deform/ncc_template": 0.5,
                  "val_rigid/time_s": 10.0}
        ok = E.compare_validation({"val_rigid/pair_rms_mm": 3.005, "val_deform/template_dice": 0.8005,
                                   "val_deform/ncc_template": 0.9, "val_rigid/time_s": 99.0}, logged)
        self.assertTrue(ok["pass"])
        self.assertEqual(ok["n_compared"], 2)
        self.assertAlmostEqual(ok["max_mm_diff"], 0.005, places=9)
        bad_mm = E.compare_validation({"val_rigid/pair_rms_mm": 3.02, "val_deform/template_dice": 0.8}, logged)
        self.assertFalse(bad_mm["pass"])
        self.assertEqual(bad_mm["worst"]["mm"], "val_rigid/pair_rms_mm")
        bad_dice = E.compare_validation({"val_rigid/pair_rms_mm": 3.0, "val_deform/template_dice": 0.802}, logged)
        self.assertFalse(bad_dice["pass"])
        self.assertFalse(E.compare_validation({"val_rigid/pair_rms_mm": 3.0}, {})["pass"])
        nan = E.compare_validation({"val_rigid/pair_rms_mm": float("nan")}, logged)
        self.assertFalse(nan["pass"])


def _snapshot(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        for n in dn + fn:
            p = os.path.join(dp, n)
            st = os.stat(p)
            out[os.path.relpath(p, root)] = (st.st_size if n in fn else -1, st.st_mtime_ns)
    return out


class TestEndToEnd(unittest.TestCase):
    """Tiny level-4 run (rigid + deform, TINY widths, overfit-synthetic, 2 steps) + fake export with
    synthfine labels."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="evalsyn_e2e_")
        cls.data = write_fake_export(cls.tmp, n_train=2, n_val=3, labels=True, fine_labels=True)
        res = os.path.join(cls.tmp, "results")
        argv = (["--overfit-one-group", "--overfit-synthetic", "--level", "4", "--group-size", "2",
                 "--deform-group-size", "2", "--task-ratio", "rigid:1,deform:1", "--deform-residual-affine",
                 "false", "--log-every", "1", "--val-every", "0", "--val-png", "false", "--schedule",
                 "constant", "--warmup-steps", "0", "--overfit-max-steps", "2", "--results", res,
                 "--run-name", "tiny"] + TINY_ARGS + COMMON)
        rc = MAIN.main(argv)
        free_memory()
        assert rc in (0, 1), rc
        cls.run_dir = os.path.join(res, "tiny")
        assert os.path.isdir(os.path.join(cls.run_dir, "ckpt", "last"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        free_memory()

    def tearDown(self):
        free_memory()

    def test_eval_two_runs_shared_sets(self):
        before = _snapshot(self.run_dir)
        out = os.path.join(self.tmp, "out_ok")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = ES.main(["--runs", self.run_dir, self.run_dir, "--level", "4", "--presets", "random_0,t2_like",
                          "--data", self.data, "--out", out, "--sanity-gate", "off"])
        self.assertEqual(rc, 0)
        self.assertEqual(_snapshot(self.run_dir), before)              # nothing written into the run
        with open(os.path.join(out, E.JSON_NAME)) as f:
            rep = json.load(f)
        self.assertEqual(rep["presets"], ["random_0", "t2_like"])
        labels = [r["label"] for r in rep["runs"]]
        self.assertEqual(labels, ["tiny", "tiny#2"])
        self.assertEqual(rep["n_specs"], 1)
        self.assertEqual(rep["exit_code"], 0)
        for p in ("random_0", "t2_like"):
            res = rep["results"][p]
            self.assertEqual(sorted(res["runs"]), sorted(labels))
            self.assertTrue(all(res["shared_sets"].values()))
            sha = [res["runs"][lb]["sha256"] for lb in labels]
            self.assertEqual(sha[0], sha[1])
            self.assertEqual(sorted(sha[0]), ["synth_%s_deform" % p, "synth_%s_rigid" % p])
            m = res["runs"]["tiny"]["metrics"]
            self.assertIn("val_synth_%s_rigid/pair_rms_mm" % p, m)
            self.assertIn("val_synth_%s_deform/template_dice" % p, m)
            self.assertIn("val_synth_%s_deform/ncc_template_identity" % p, m)
            self.assertIsNotNone(res["ru_maxrss_kb"])
        self.assertNotEqual(rep["results"]["random_0"]["runs"]["tiny"]["sha256"]["synth_random_0_rigid"],
                            rep["results"]["t2_like"]["runs"]["tiny"]["sha256"]["synth_t2_like_rigid"])
        with open(os.path.join(out, E.CSV_NAME)) as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], ["preset", "metric", "tiny", "tiny#2", "diff"])
        self.assertGreater(len(rows), 10)
        same = [r for r in rows[1:] if r[1].endswith("/pair_rms_mm")]
        self.assertTrue(same and all(abs(float(r[4])) < 1e-4 for r in same))   # same model twice
        # compare_synth_eval: one JSON prints the table; two JSONs with equal sets pass.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "compare_synth_eval", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                               "scripts", "compare_synth_eval.py"))
        CMP = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(CMP)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(CMP.main([os.path.join(out, E.JSON_NAME)]), 0)
            self.assertEqual(CMP.main([os.path.join(out, E.JSON_NAME), os.path.join(out, E.JSON_NAME)]), 0)
        self.assertIn("val_synth_t2_like_deform/template_dice", buf.getvalue())
        bad = dict(rep)
        bad["results"] = json.loads(json.dumps(rep["results"]))
        bad["results"]["t2_like"]["runs"]["tiny"]["sha256"]["synth_t2_like_rigid"] = "0"
        bad_path = os.path.join(self.tmp, "bad.json")
        with open(bad_path, "w") as f:
            json.dump(bad, f)
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(CMP.main([os.path.join(out, E.JSON_NAME), bad_path]), 1)
        self.assertIn("preset t2_like", err.getvalue())

    def test_identity_cache_per_preset_and_determinism(self):
        ns, raw = E.load_run_config(self.run_dir)
        entry = E.build_trainer(self.run_dir, ns, raw, self.data)
        spec = E.val_spec(ns, self.data)
        data = E.load_val_data(spec)
        real = E.real_val_sets(spec, data)
        vals, shas = {}, {}
        cfg = P.SynthConfig()
        for p in ("random_0", "t2_like"):
            sets, shas[p] = E.synthetic_val_sets(p, spec, data, cfg)
            self.assertEqual(sorted(sets), ["synth_%s_deform" % p, "synth_%s_rigid" % p])
            for name, vs in sets.items():                   # same groups / transforms as the real sets
                rn = name[len("synth_%s_" % p):]
                for gs, gr in zip(vs.groups, real[rn].groups):
                    np.testing.assert_array_equal(gs["index"], gr["index"])
                    for k in ("T", "labels"):
                        if k in gr:
                            np.testing.assert_array_equal(gs[k], gr[k])
                    self.assertFalse(np.array_equal(gs["images"], gr["images"]))
            entry["trainer"].val_sets = sets
            vals[p] = entry["trainer"].validate()
        cache = entry["trainer"]._identity_cache
        self.assertIn(("synth_random_0_deform", 2), cache)
        self.assertIn(("synth_t2_like_deform", 2), cache)
        a = vals["random_0"]["val_synth_random_0_deform/ncc_template_identity"]
        b = vals["t2_like"]["val_synth_t2_like_deform/ncc_template_identity"]
        self.assertNotEqual(a, b)
        _, again = E.synthetic_val_sets("random_0", spec, data, cfg)
        self.assertEqual(again, shas["random_0"])           # deterministic CPU synthesis

    def test_failed_gate_exits_5_and_cli_errors(self):
        before = _snapshot(self.run_dir)
        out = os.path.join(self.tmp, "out_gate")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = ES.main(["--runs", self.run_dir, "--level", "4", "--presets", "t1_like", "--data", self.data,
                          "--out", out])
        # The overfit run logged its validation of the fixed training group, not of the export's
        # val sets: the gate sees mm-scale differences and fails.
        self.assertEqual(rc, E.EXIT_GATE_FAILED)
        with open(os.path.join(out, E.JSON_NAME)) as f:
            rep = json.load(f)
        gate = rep["sanity_gate"]["tiny"]
        self.assertFalse(gate["pass"])
        self.assertGreater(gate["n_compared"], 0)
        self.assertTrue(gate["max_mm_diff"] > E.GATE_TOL_MM or gate["max_dice_diff"] > E.GATE_TOL_DICE, gate)
        self.assertEqual(rep["exit_code"], E.EXIT_GATE_FAILED)
        self.assertEqual(rep["results"], {})
        self.assertEqual(_snapshot(self.run_dir), before)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(ES.main(["--runs", self.run_dir, "--level", "2", "--presets", "t1_like",
                                      "--data", self.data, "--out", out, "--sanity-gate", "off"]), 2)
        self.assertIn("level 4", err.getvalue())
        for bad in (["--runs", self.run_dir, "--out", os.path.join(self.run_dir, "x")],
                    ["--runs", self.run_dir, "--out", out, "--presets", "nope"],
                    ["--runs", self.run_dir, self.run_dir, self.run_dir, "--out", out]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                ES.parse_args(bad)
            self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
