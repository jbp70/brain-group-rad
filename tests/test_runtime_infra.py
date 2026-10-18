"""Args (runtime.args), logging, checkpoints and TF-side fixed data (runtime.fixed_data)."""
import argparse
import json
import os
import shutil
import tempfile
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.runtime_helpers import free_memory, narrow_config, write_fake_export, NARROW_ARGS
import numpy as np
import tensorflow as tf

from data_loading import params as P
from data_loading.postprocess import LinearPostprocess
from models.group_net import GroupRegLevelModel
from runtime import args as A
from runtime import checkpoint as C
from runtime import fixed_data as fd
from runtime import metrics as M
from runtime import optimizers as O
from runtime.logging import RunLogger, make_run_dir


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_m4_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        free_memory()


class TestArgs(unittest.TestCase):
    def test_str2bool(self):
        for v in ("yes", "True", "t", "1", "on", True):
            self.assertTrue(A.str2bool(v))
        for v in ("no", "False", "f", "0", "off", False):
            self.assertFalse(A.str2bool(v))
        with self.assertRaises(argparse.ArgumentTypeError):
            A.str2bool("maybe")

    def test_level_defaults_and_overfit_dropout(self):
        for level, n in ((4, 32), (2, 8), (1, 2)):
            self.assertEqual(A.parse_args(["--data", "x", "--level", str(level)]).group_size, n)
        self.assertEqual(A.parse_args(["--data", "x"]).dropout, 0.15)
        self.assertEqual(A.parse_args(["--overfit-one-group", "--overfit-synthetic"]).dropout, 0.0)
        self.assertEqual(A.parse_args(["--overfit-one-group", "--overfit-synthetic", "--dropout", "0.2"]).dropout, 0.2)
        a = A.parse_args(["--data", "x", "--val-every", "100"])
        self.assertEqual(a.ckpt_every, 100)
        self.assertTrue(a.run_name.startswith("l4_rigid1_"))

    def test_model_config_round_trip(self):
        a = A.parse_args(["--data", "x", "--group-config", "B", "--translation-scale", "10"] + NARROW_ARGS)
        cfg = A.model_config_from_args(a)
        self.assertEqual(cfg.group_config, "B")
        self.assertEqual(cfg.translation_scale, 10.0)
        self.assertEqual(cfg.filters, (16, 32, 32, 48, 64, 96))
        self.assertEqual(type(cfg).from_dict(cfg.to_dict()), cfg)
        tc = A.train_config_from_args(a)
        self.assertEqual((tc.level, tc.group_size, tc.lambda_pair, tc.p_pair), (4, 32, 1.0, 0.1))
        self.assertEqual(tc.schedule.base_lr, 3e-3)
        self.assertEqual(A.ranges_from_args(a)["rigid"], dict(P.DEFAULT_LINEAR_RANGES["rigid"]))
        json.dumps(A.args_to_dict(a))

    def test_loss_scale_and_head_activation_flags(self):
        a = A.parse_args(["--data", "x"])
        self.assertEqual((a.loss_scale_mm2, a.head_activation), (1.0, "relu"))
        self.assertEqual(A.train_config_from_args(a).loss_scale_mm2, 1.0)
        self.assertEqual(A.model_config_from_args(a).head_activation, "relu")
        a = A.parse_args(["--data", "x", "--loss-scale-mm2", "100", "--head-activation", "none"])
        self.assertEqual(A.train_config_from_args(a).loss_scale_mm2, 100.0)
        self.assertEqual(A.model_config_from_args(a).head_activation, "none")
        d = A.args_to_dict(a)
        self.assertEqual((d["loss_scale_mm2"], d["head_activation"]), (100.0, "none"))
        import contextlib
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            for bad in (["--loss-scale-mm2", "0"], ["--loss-scale-mm2", "-1"], ["--head-activation", "gelu"]):
                with self.assertRaises(SystemExit):
                    A.parse_args(["--data", "x"] + bad)

    def test_ratio_and_required_data(self):
        self.assertEqual(dict(A.parse_task_ratio("rigid:3, affine:1")), {"rigid": 3.0, "affine": 1.0})
        with self.assertRaises(NotImplementedError) as cm:
            A.parse_task_ratio("rigid:1,deform:2")
        self.assertIn("milestone 6", str(cm.exception))
        with self.assertRaises(ValueError):
            A.parse_task_ratio("rigid:0")
        devnull = open(os.devnull, "w")
        try:
            import contextlib
            with contextlib.redirect_stderr(devnull):
                with self.assertRaises(SystemExit):
                    A.parse_args([])                      # --data missing
                with self.assertRaises(SystemExit):
                    A.parse_args(["--data", "x", "--task-ratio", "deform:1"])
        finally:
            devnull.close()


class TestLogging(TempDirCase):
    def test_jsonl_and_tensorboard(self):
        paths = make_run_dir(self.tmp, "run")
        for k in ("logs", "tb", "ckpt_last", "ckpt_best"):
            self.assertTrue(os.path.isdir(paths[k]), k)
        log = RunLogger(paths, backend="jsonl", quiet=True)
        log.log_params({"lr": 0.01, "name": "x", "arr": np.arange(3)})
        for s in (1, 2, 3):
            log.log_metrics(s, {"loss": np.float32(1.5 * s), "rms": tf.constant(2.0 * s), "nan": float("nan")})
        log.close()
        with open(log.metrics_path) as f:
            lines = [json.loads(line) for line in f]
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0]["step"], "PARAMETER")
        self.assertEqual(lines[0]["data"]["arr"], [0, 1, 2])
        self.assertEqual([r["step"] for r in lines[1:]], [1, 2, 3])
        self.assertEqual(lines[3]["data"]["loss"], 4.5)
        self.assertEqual(lines[3]["data"]["rms"], 6.0)
        self.assertIsInstance(lines[3]["data"]["loss"], float)
        self.assertEqual(lines[1]["data"]["nan"], "nan")
        self.assertEqual(log.backend_name, "jsonl")
        self.assertTrue(any(n.startswith("events.out.tfevents") for n in os.listdir(paths["tb"])))


class TestCheckpoint(TempDirCase):
    def make(self):
        model = GroupRegLevelModel(4, narrow_config(dropout=0.0)).build_model(3)
        opt = O.make_optimizer("sgdw", lr=0.01)
        O.build_optimizer(opt, model.trainable_variables)
        step = tf.Variable(0, dtype=tf.int64)
        gen = tf.random.Generator.from_seed(5)
        ck = C.RunCheckpoint(os.path.join(self.tmp, "last"), os.path.join(self.tmp, "best"), model, opt,
                             step, extra={"gen": gen})
        return model, opt, step, gen, ck

    def train_two_steps(self, model, opt, step):
        x = tf.random.stateless_uniform((3,) + P.LEVEL_SHAPE[4] + (1,), seed=(1, 2))
        for _ in range(2):
            with tf.GradientTape() as tape:
                a = model(x, task="rigid", training=True)["A"]
                loss = tf.reduce_sum(tf.square(a[:, :3, 3] - 1.0))
            tv = model.trainable_variables
            gv = [(g, v) for g, v in zip(tape.gradient(loss, tv), tv) if g is not None]
            opt.apply_gradients(gv)
            step.assign_add(1)

    def test_round_trip_bit_exact(self):
        model, opt, step, gen, ck = self.make()
        self.train_two_steps(model, opt, step)
        gen.uniform([3])
        ck.save_last()
        ref_model = [v.numpy() for v in model.variables]
        ref_opt = [v.numpy() for v in opt.variables()]
        ref_gen = gen.state.numpy()
        model2, opt2, step2, gen2, ck2 = self.make()
        self.assertTrue(ck2.restore_last())
        self.assertEqual(int(step2.numpy()), 2)
        self.assertEqual(int(opt2.iterations.numpy()), 2)
        self.assertEqual(len(ref_opt), len(opt2.variables()))
        for a, b in zip(ref_model, model2.variables):
            self.assertTrue(np.array_equal(a, b.numpy()), b.name)
        for a, b in zip(ref_opt, opt2.variables()):
            self.assertTrue(np.array_equal(a, b.numpy()), b.name)
        np.testing.assert_array_equal(gen2.state.numpy(), ref_gen)
        bn = [v for v in model2.non_trainable_variables if "moving_mean" in v.name]
        self.assertTrue(any(np.any(v.numpy() != 0) for v in bn))   # statistics moved and restored

    def test_best_and_keep_last(self):
        model, opt, step, gen, ck = self.make()
        for s, val in ((1, 5.0), (2, 6.0), (3, 4.0)):
            step.assign(s)
            saved = ck.maybe_save_best(val)           # Trainer order: validate, best, then last
            ck.save_last()
            self.assertEqual(saved, val != 6.0)
        self.assertFalse(ck.maybe_save_best(float("nan")))
        self.assertTrue(ck.latest_best.endswith("best-3"))
        self.assertEqual(len(ck.last.checkpoints), 2)
        self.assertTrue(ck.latest_last.endswith("last-3"))
        model2, opt2, step2, gen2, ck2 = self.make()
        ck2.restore_last()
        self.assertEqual(float(ck2.best_metric.numpy()), 4.0)

    def test_config_json_and_compatibility(self):
        path = os.path.join(self.tmp, "config.json")
        cfg = narrow_config()
        C.write_config(path, cfg, {"lr": 0.01, "filters": [1, 2]})
        with open(path) as f:
            d = json.load(f)
        self.assertEqual(d["args"]["lr"], 0.01)
        self.assertIn("tensorflow", d["versions"])
        C.check_config_compatible(path, cfg)
        with self.assertRaises(ValueError) as cm:
            C.check_config_compatible(path, cfg.replace(dropout=0.3, group_config="B"))
        self.assertIn("dropout", str(cm.exception))
        self.assertIn("group_config", str(cm.exception))


class TestFixedData(TempDirCase):
    def test_transform_ranges(self):
        rng = np.random.default_rng(0)
        T, tgt = fd.random_linear_transforms(rng, 200, "rigid")
        r = P.DEFAULT_LINEAR_RANGES["rigid"]
        self.assertLessEqual(float(np.max(M.rotation_angle_deg(tgt[:, :3, :3]))), r["rotation_deg"] + 1e-6)
        self.assertLessEqual(float(np.max(np.abs(tgt[:, :3, 3]))), r["translation_mm"])
        rr = tgt[:, :3, :3]
        np.testing.assert_allclose(rr @ np.swapaxes(rr, 1, 2), np.tile(np.eye(3), (200, 1, 1)), atol=1e-12)
        np.testing.assert_allclose(T[:, :3, 3], tgt[:, :3, 3])
        k = np.linalg.inv(tgt) @ T                                # Rt^-1 T = nuisance K (upper triangular)
        self.assertLess(float(np.max(np.abs(np.tril(k[:, :3, :3], -1)))), 1e-12)
        self.assertGreater(float(np.max(np.abs(k[:, :3, :3] - np.eye(3)))), 1e-4)
        Tn, tn = fd.random_linear_transforms(rng, 5, "rigid", nuisance=False)
        np.testing.assert_allclose(Tn, tn)
        Ta, ta = fd.random_linear_transforms(rng, 5, "affine")
        np.testing.assert_allclose(Ta, ta)

    def test_dali_batch_round_trip(self):
        rng = np.random.default_rng(1)
        for task in ("rigid", "affine"):
            T, tgt = fd.random_linear_transforms(rng, 4, task)
            imgs = np.zeros((4,) + P.LEVEL_SHAPE[4] + (1,), np.float32)
            batch = fd.to_dali_batch(imgs, T, tgt, task, 4)
            self.assertEqual(len(batch), 3 if task == "rigid" else 2)
            self.assertEqual(tuple(batch[1].shape), (4, 3, 4))
            out = LinearPostprocess(task, 4, flip_prob=0.0)(*batch)
            np.testing.assert_allclose(out["T_target"].numpy(), tgt, atol=1e-4)
            np.testing.assert_allclose(out["T_render"].numpy(), T, atol=1e-4)
            a = out["A_star"].numpy().astype(np.float64)
            np.testing.assert_allclose(a @ tgt, np.broadcast_to((a @ tgt)[:1], a.shape), atol=1e-4)

    def test_load_level_images(self):
        root = write_fake_export(self.tmp)
        x, files = fd.load_level_images(root, 4, "train", n=2, seed=3)
        self.assertEqual(x.shape, (2,) + P.LEVEL_SHAPE[4] + (1,))
        self.assertEqual(x.dtype, np.float32)
        y, files2 = fd.load_level_images(root, 4, "train", n=2, seed=3)
        self.assertEqual(files, files2)
        np.testing.assert_array_equal(x, y)
        v, vf = fd.load_level_images(root, 4, "val")
        self.assertEqual(len(vf), 2)
        with self.assertRaises(ValueError):
            fd.load_level_images(root, 4, "train", n=5)

    def test_validation_set_deterministic(self):
        rng = np.random.default_rng(2)
        imgs = np.stack([fd.smooth_phantom(P.LEVEL_SHAPE[4], rng)[0][..., None] for _ in range(5)])
        a = fd.ValidationSet.build(imgs, "rigid", 4, group_size=3, seed=7)
        b = fd.ValidationSet.build(imgs, "rigid", 4, group_size=3, seed=7)
        self.assertEqual(len(a.groups), 2)                 # ceil(5 / 3)
        self.assertEqual(a.group_size, 3)
        covered = set()
        for ga, gb in zip(a.groups, b.groups):
            np.testing.assert_array_equal(ga["index"], gb["index"])
            np.testing.assert_array_equal(ga["images"], gb["images"])
            np.testing.assert_array_equal(ga["T"], gb["T"])
            self.assertEqual(len(set(ga["index"].tolist())), 3)
            covered |= set(ga["index"].tolist())
        self.assertEqual(covered, set(range(5)))
        c = fd.ValidationSet.build(imgs, "rigid", 4, group_size=3, seed=8)
        self.assertFalse(np.array_equal(c.groups[0]["T"], a.groups[0]["T"]))
        with self.assertRaises(ValueError):
            fd.ValidationSet.build(imgs, "rigid", 4, group_size=6)

    def test_validation_set_independent_of_task_order(self):
        # Review issue 4: the rigid set must not depend on the task's position in the ratio.
        rng = np.random.default_rng(3)
        imgs = np.stack([fd.smooth_phantom(P.LEVEL_SHAPE[4], rng)[0][..., None] for _ in range(5)])
        ranges = A.ranges_from_args(A.parse_args(["--data", "x"]))
        alone = fd.build_validation_sets(imgs, list(A.parse_task_ratio("rigid:1")), 4, 3, ranges=ranges, seed=7)
        mixed = fd.build_validation_sets(imgs, list(A.parse_task_ratio("affine:1,rigid:2")), 4, 3,
                                         ranges=ranges, seed=7)
        self.assertEqual(sorted(mixed), ["affine", "rigid"])
        for ga, gb in zip(alone["rigid"].groups, mixed["rigid"].groups):
            np.testing.assert_array_equal(ga["index"], gb["index"])
            np.testing.assert_array_equal(ga["T"], gb["T"])
            np.testing.assert_array_equal(ga["images"], gb["images"])
        self.assertFalse(np.array_equal(mixed["affine"].groups[0]["T"], mixed["rigid"].groups[0]["T"]))
        # group g depends on (seed, task, g) only, not on how many groups are built
        one = fd.ValidationSet.build(imgs, "rigid", 4, group_size=3, n_groups=1, ranges=ranges["rigid"], seed=7)
        np.testing.assert_array_equal(one.groups[0]["T"], alone["rigid"].groups[0]["T"])


class TestDLLoggerBackend(TempDirCase):
    def test_dllogger_backend(self):
        try:
            import dllogger  # noqa: F401
        except ImportError:
            self.skipTest("dllogger is not importable here (the container may have it)")
        paths = make_run_dir(self.tmp, "dll")
        log = RunLogger(paths, backend="dllogger", tensorboard=False, quiet=True)
        self.assertEqual(log.backend_name, "dllogger")
        log.log_params({"lr": 0.01, "arr": np.arange(2)})
        log.log_metrics(1, {"loss": np.float32(1.5), "rms": tf.constant(2.0)})
        log.log_event(1, {"event": "x"})
        log.close()
        self.assertTrue(os.path.isfile(log.metrics_path))
        with open(log.metrics_path) as f:
            text = f.read()
        self.assertIn("loss", text)
        self.assertIn("PARAMETER", text)


if __name__ == "__main__":
    unittest.main()
