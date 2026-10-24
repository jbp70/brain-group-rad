"""T15 (milestone 8, design section 11): tests/gpu/measure_level_memory.py runs end to end on a
tiny level-2 cascade configuration (GPU only; skipped without a GPU).

The local GTX 1080 runs under the 6000 MB cap: a level-2 TINY deform step with T = 7 and no
checkpointing does not fit there (FEAT-002), so the test passes --integration-steps 3 and
--gradient-checkpointing both to get a measured 'ok' result; 'oom' is still accepted.

Milestone 9 phase 1 (`.agents/tasks/milestone-9/design-m9-phase1.md` 5.1, 5.2, 5.6):
test_tiny_level1 (level-1 TINY N 2 with the frozen-chain stand-in, AMP, full checkpointing, lean
knobs; 'ok' or 'oom'; predicted fields, cascade_kind, frozen peak; a second invocation merges),
test_incremental_write (per-configuration atomic write, patched measure), test_merge_old_record
(m8-format records merge with re-measured ones; T 6 kept separately; no GPU needed).
"""
import tests.tf_setup  # noqa: F401  (TF32 off, GPU memory setup) -- must come first

import contextlib
import glob
import importlib.util
import io
import json
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

import tensorflow as tf

from tests.runtime_helpers import free_memory

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "measure_level_memory.py")
_spec = importlib.util.spec_from_file_location("measure_level_memory", _PATH)
mlm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mlm)


@unittest.skipIf(not tf.config.list_physical_devices("GPU"), "needs a GPU")
class TestMeasureLevelMemory(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mlm_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        tf.keras.mixed_precision.set_global_policy("float32")
        free_memory()

    def test_tiny_level2_cascade(self):
        out = os.path.join(self.tmp, "memory.json")
        rc = mlm.main(["--level", "2", "--tasks", "deform,rigid", "--n", "2", "--widths", "tiny",
                       "--cascade", "true", "--gradient-checkpointing", "both", "--integration-steps", "3",
                       "--steps", "2", "--warmup-steps", "1", "--val-pass", "true",
                       "--gpu-memory-limit-mb", "6000", "--out", out])
        self.assertEqual(rc, 0)
        with open(out) as f:
            res = json.load(f)["results"]
        by_task = {r["task"]: r for r in res}
        self.assertEqual(sorted(by_task), ["deform", "rigid"])
        for r in res:
            print("[T15] %s status %s peak_mb %s warmup_peak_mb %s resident_mb %s step_ms %s frozen_ms %s "
                  "val_status %s val_peak_mb %s val_ms %s" % (
                      r["task"], r["status"], r.get("peak_mb"), r.get("warmup_peak_mb"), r.get("resident_mb"),
                      r.get("step_ms"), r.get("frozen_ms"), r.get("val_status"), r.get("val_peak_mb"),
                      r.get("val_ms")))
            self.assertIn(r["status"], ("ok", "oom"), r.get("error"))
            self.assertEqual((r["level"], r["n"], r["widths"], r["cascade"]), (2, 2, "tiny", True))
            if r["status"] == "ok":
                for k in ("peak_mb", "warmup_peak_mb", "resident_mb", "step_ms", "step_ms_mean", "frozen_ms"):
                    self.assertIsInstance(r[k], float, k)
                self.assertEqual(len(r["step_ms_all"]), 2)
                self.assertTrue(r["finite"])
                self.assertEqual(r["trace_counts"][r["task"]], 1)
        d, rg = by_task["deform"], by_task["rigid"]
        self.assertEqual(d["gradient_checkpointing"], "both")
        self.assertEqual(rg["gradient_checkpointing"], "none")
        self.assertIn(d["val_status"], ("ok", "oom"), d.get("val_error"))
        if d["val_status"] == "ok":
            self.assertIsInstance(d["val_peak_mb"], float)
            self.assertIsInstance(d["val_ms"], float)
        self.assertIsNone(rg["val_peak_mb"])
        self.assertIsNone(rg["val_status"])
        for r in res:                                   # milestone-9 fields (design 5.1)
            self.assertEqual(r["build_n"], 2)
            self.assertEqual(r["cascade_kind"], "frozen4")
            self.assertEqual(r["integration_steps"], 3)
            self.assertIn("predicted_peak_mb", r)
            self.assertIn("largest_tensor_elems", r)

    def test_tiny_level1(self):
        """Design 5.6: level-1 TINY N 2 with the frozen-chain stand-in, AMP, full checkpointing and
        both lean knobs, under the 6000 MB cap: 'ok' or 'oom', never a crash; a second invocation
        merges."""
        out = os.path.join(self.tmp, "memory_l1.json")
        args = ["--level", "1", "--tasks", "deform,rigid", "--n", "2", "--widths", "tiny", "--cascade", "true",
                "--gradient-checkpointing", "full", "--lean-sampler", "true", "--lean-losses", "true",
                "--integration-steps", "3", "--amp", "true", "--steps", "1", "--warmup-steps", "1",
                "--val-pass", "true", "--val-pair-chunk", "1", "--gpu-memory-limit-mb", "6000", "--out", out]
        t0 = time.time()
        self.assertEqual(mlm.main(args), 0)
        print("[m9 tiny level 1] first invocation %.1f s" % (time.time() - t0))
        with open(out) as f:
            res = json.load(f)["results"]
        self.assertEqual(sorted(r["task"] for r in res), ["deform", "rigid"])
        for r in res:
            print("[m9 tiny level 1] %s status %s peak_mb %s pred %s (%s) largest %s frozen %s/%s val %s/%s" % (
                r["task"], r["status"], r.get("peak_mb"), r.get("predicted_peak_mb"), r.get("predicted_phase"),
                r.get("largest_tensor_elems"), r.get("frozen_status"), r.get("frozen_peak_mb"),
                r.get("val_status"), r.get("val_peak_mb")))
            self.assertIn(r["status"], ("ok", "oom"), r.get("error"))
            self.assertEqual((r["level"], r["n"], r["widths"], r["cascade"]), (1, 2, "tiny", True))
            self.assertEqual(r["cascade_kind"], "standin_chain_4_2")
            self.assertEqual(r["build_n"], 2)
            self.assertIsInstance(r["predicted_peak_mb"], float)
            self.assertIsInstance(r["largest_tensor_elems"], int)
            self.assertIn("frozen_peak_mb", r)
            self.assertIn("frozen_status", r)
            if r["status"] == "ok":
                self.assertEqual(r["frozen_status"], "ok")
                self.assertIsInstance(r["frozen_peak_mb"], float)
                self.assertIsInstance(r["pred_over_meas"], float)
                self.assertIsInstance(r["frozen_params"], int)
        d = [r for r in res if r["task"] == "deform"][0]
        rg = [r for r in res if r["task"] == "rigid"][0]
        self.assertEqual((d["gradient_checkpointing"], d["lean_sampler"], d["lean_losses"]), ("full", True, True))
        self.assertEqual((rg["gradient_checkpointing"], rg["lean_sampler"], rg["lean_losses"]), ("none", False, False))
        self.assertEqual(d["val_pair_chunk"], 1)
        # second invocation (rigid only, other N) merges into the same file
        args2 = list(args)
        args2[args2.index("--tasks") + 1] = "rigid"
        args2[args2.index("--n") + 1] = "3"
        self.assertEqual(mlm.main(args2), 0)
        with open(out) as f:
            res2 = json.load(f)["results"]
        self.assertEqual(sorted((r["task"], r["n"]) for r in res2), [("deform", 2), ("rigid", 2), ("rigid", 3)])
        print("[m9 tiny level 1] both invocations %.1f s" % (time.time() - t0))

    def test_incremental_write(self):
        """Design 5.1 (v5): the output is merged and written atomically after every configuration."""
        out = os.path.join(self.tmp, "inc.json")
        old = dict(_m9_record(n=3), status="ok", peak_mb=1.0)
        with open(out, "w") as f:
            json.dump({"environment": {}, "results": [old]}, f)
        snaps = []
        orig_write = mlm.write_merged

        def fake_measure(base, warmup_steps, steps, val_pass, seed=0):
            return dict(base, status="ok", peak_mb=100.0 * base["n"], warmup_peak_mb=1.0, resident_mb=1.0,
                        step_ms=1.0, frozen_ms=None, frozen_peak_mb=None, frozen_status=None, val_peak_mb=None,
                        val_status=None, val_ms=None)

        def snap_write(path, env, records):
            orig_write(path, env, records)
            with open(path) as f:
                snaps.append(json.load(f)["results"])
            self.assertEqual(glob.glob(path + ".tmp.*"), [])

        with mock.patch.object(mlm, "measure", fake_measure), mock.patch.object(mlm, "write_merged", snap_write):
            rc = mlm.main(["--level", "4", "--tasks", "deform", "--n", "2,3", "--widths", "narrow",
                           "--cascade", "false", "--integration-steps", "7", "--gpu-memory-limit-mb", "6000",
                           "--out", out])
        self.assertEqual(rc, 0)
        self.assertEqual(len(snaps), 2)
        self.assertEqual(sorted((r["n"], r["peak_mb"]) for r in snaps[0]), [(2, 200.0), (3, 1.0)])
        self.assertEqual(sorted((r["n"], r["peak_mb"]) for r in snaps[1]), [(2, 200.0), (3, 300.0)])
        self.assertEqual(glob.glob(out + ".tmp.*"), [])


@unittest.skipIf(not tf.config.list_physical_devices("GPU"), "main() needs a visible GPU")
class TestIsolate(unittest.TestCase):
    """FEAT-004 leak fix: `--isolate true` runs every configuration (one N) in its own child process
    (patched subprocess.run here: no GPU work). In one process `resident_mb` grew from 73 to 329 MB
    over 48 narrow hold-out configurations; one process per configuration keeps it flat (73.4-73.6 MB
    for the same configurations measured one per process, FEAT-004 report)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mlm_iso_")
        self.out = os.path.join(self.tmp, "iso.json")
        self.calls = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fake_run(self, oom_lean=False, crash=False):
        def run(argv, stdin=None):
            self.calls.append(list(argv))
            a = dict(zip(argv[2::2], argv[3::2]))           # after [python, script]
            if crash:
                return mock.Mock(returncode=1)
            ls = a["--lean-sampler"] == "true"
            base = _m9_record(n=int(a["--n"]), gradient_checkpointing=a["--gradient-checkpointing"],
                              lean_sampler=ls, integration_steps=int(a["--integration-steps"]))
            st = "oom" if (oom_lean and ls) else "ok"
            # every child is a fresh process: the same resident memory for every configuration
            mlm.write_merged(a["--out"], {"child": True},
                             [dict(base, status=st, peak_mb=100.0 * base["n"] if st == "ok" else None,
                                   resident_mb=73.5, child_index=len(self.calls))])
            return mock.Mock(returncode=0)
        return run

    def _main(self, extra=()):
        argv = ["--level", "4", "--tasks", "deform", "--n", "2,3", "--widths", "narrow", "--cascade", "false",
                "--gradient-checkpointing", "full", "--lean-sampler", "false,true", "--integration-steps", "7",
                "--isolate", "true", "--gpu-memory-limit-mb", "6000", "--out", self.out] + list(extra)
        with mock.patch.object(mlm, "configure_gpu", side_effect=AssertionError("parent touched the GPU")):
            return mlm.main(argv)

    def _results(self):
        with open(self.out) as f:
            return json.load(f)["results"]

    def test_one_child_per_configuration(self):
        with mock.patch.object(mlm.subprocess, "run", self._fake_run()):
            self.assertEqual(self._main(), 0)
        self.assertEqual(len(self.calls), 4)
        for c in self.calls:
            self.assertEqual(c[1], os.path.abspath(mlm.__file__))
            a = dict(zip(c[2::2], c[3::2]))
            self.assertEqual(a["--isolate"], "false")
            self.assertIn(a["--n"], ("2", "3"))
            self.assertIn(a["--lean-sampler"], ("true", "false"))
            self.assertEqual((a["--gradient-checkpointing"], a["--integration-steps"], a["--out"]),
                             ("full", "7", self.out))
        res = self._results()
        self.assertEqual(len(res), 4)
        self.assertEqual(sorted(r["child_index"] for r in res), [1, 2, 3, 4])   # each from its own child
        self.assertEqual({r["resident_mb"] for r in res}, {73.5})

    def test_skip_after_child_oom(self):
        with mock.patch.object(mlm.subprocess, "run", self._fake_run(oom_lean=True)):
            self.assertEqual(self._main(), 0)
        self.assertEqual(len(self.calls), 3)                 # lean N 3 not launched
        st = {(r["lean_sampler"], r["n"]): r["status"] for r in self._results()}
        self.assertEqual(st, {(False, 2): "ok", (False, 3): "ok", (True, 2): "oom", (True, 3): "skipped"})

    def test_child_crash_gives_error_record(self):
        with mock.patch.object(mlm.subprocess, "run", self._fake_run(crash=True)):
            self.assertEqual(self._main(["--lean-sampler", "false", "--n", "2"]), 0)
        res = self._results()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["status"], "error")
        self.assertIn("child exit code 1", res[0]["error"])

    def test_child_argv_custom_widths(self):
        a = mock.Mock(build_n=2, warmup_steps=1, steps=3, gpu_memory_limit_mb=0, coefficients="c.json", seed=0,
                      out="o.json")
        base = _m9_record(widths=mlm.custom_key((16, 64, 128, 192, 256, 384), (128, 96, 64, 32, 8)))
        argv = mlm.child_argv(a, base, False)
        i = argv.index("--widths")
        self.assertEqual(argv[i:i + 6], ["--widths", "custom", "--filters", "16,64,128,192,256,384",
                                         "--decoder-widths", "128,96,64,32,8"])


def _m9_record(**kw):
    r = {"level": 4, "task": "deform", "n": 2, "widths": "narrow", "cascade": False,
         "gradient_checkpointing": "none", "amp": False, "lambda_ncc_tmpl": 0.0, "lambda_ncc_pair": 0.0,
         "integration_steps": 7, "lean_sampler": False, "lean_losses": False, "split_group_conv": False,
         "val_pair_chunk": 16, "cascade_kind": "none", "build_n": 2}
    r.update(kw)
    return r


class TestMergeOldRecord(unittest.TestCase):
    """Design 5.1 (review F8, v3 F6): an m8-format record (no milestone-9 fields, integration_steps
    0) and a re-measured T 7 record of the same configuration merge into one (the new one); a T 6
    record is kept separately."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mlm_merge_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_merge_old_record(self):
        out = os.path.join(self.tmp, "memory.json")
        m8 = {"level": 2, "task": "deform", "n": 4, "widths": "default", "cascade": True,
              "gradient_checkpointing": "both", "amp": True, "lambda_ncc_tmpl": 1.0, "lambda_ncc_pair": 1.0,
              "integration_steps": 0, "status": "ok", "peak_mb": 14000.0}
        with open(out, "w") as f:
            json.dump({"environment": {"old": True}, "results": [m8]}, f)
        new7 = dict(m8, integration_steps=7, lean_sampler=False, lean_losses=False, split_group_conv=False,
                    val_pair_chunk=16, cascade_kind="frozen4", build_n=2, peak_mb=13000.0)
        self.assertEqual(mlm.config_key(m8), mlm.config_key(new7))
        mlm.write_merged(out, {"new": True}, [new7])
        with open(out) as f:
            d = json.load(f)
        self.assertEqual(len(d["results"]), 1)
        self.assertEqual(d["results"][0]["integration_steps"], 7)
        self.assertEqual(d["results"][0]["peak_mb"], 13000.0)
        self.assertEqual(d["environment"], {"new": True})
        new6 = dict(new7, integration_steps=6, peak_mb=12500.0)
        mlm.write_merged(out, {"new": True}, [new6])
        with open(out) as f:
            d = json.load(f)
        self.assertEqual(sorted(r["integration_steps"] for r in d["results"]), [6, 7])
        self.assertEqual(glob.glob(out + ".tmp.*"), [])
        # a different knob value is a different configuration
        self.assertNotEqual(mlm.config_key(new7), mlm.config_key(dict(new7, lean_sampler=True)))
        self.assertNotEqual(mlm.config_key(new7), mlm.config_key(dict(new7, cascade_kind="standin_chain_4_2")))

    def test_custom_widths_and_classify(self):
        key = mlm.custom_key((16, 32, 128, 192, 256, 384), (128, 96, 64, 16, 8))
        self.assertEqual(key, "custom:f=16-32-128-192-256-384;d=128-96-64-16-8")
        cfg = mlm.model_config(key, 0)
        self.assertEqual(tuple(cfg.filters), (16, 32, 128, 192, 256, 384))
        self.assertEqual(tuple(cfg.decoder_widths), (128, 96, 64, 16, 8))
        self.assertEqual(tuple(mlm.model_config("fr50", 6).filters), (16, 32, 128, 192, 256, 384))
        self.assertEqual(mlm.model_config("fr50", 6).integration_steps, 6)
        self.assertEqual(mlm.classify_error(tf.errors.ResourceExhaustedError(None, None, "OOM")), "oom")
        self.assertEqual(mlm.classify_error(tf.errors.InvalidArgumentError(
            None, None, "num_elements exceeds 2147483647")), "limit")
        self.assertEqual(mlm.classify_error(ValueError("tensor too large for INT_MAX")), "limit")
        self.assertEqual(mlm.classify_error(tf.errors.InternalError(None, None, "cuDNN failed")), "error")
        self.assertEqual(mlm.classify_error(KeyError("2147483647")), "error")


class TestCliErrors(unittest.TestCase):
    """R2.6 / AC5 for the tool's new options: invalid values fail via ap.error naming the option.
    --isolate true skips the in-process GPU setup; every check runs before any GPU work."""

    def test_invalid_values(self):
        base = ["--isolate", "true", "--out", os.path.join(tempfile.gettempdir(), "mlm_unused.json")]
        cases = [
            (["--build-n", "0"], "--build-n"),
            (["--val-pair-chunk", "0"], "--val-pair-chunk"),
            (["--integration-steps", "7,-1"], "--integration-steps"),
            (["--integration-steps", "seven"], "--integration-steps"),
            (["--gradient-checkpointing", "fully"], "--gradient-checkpointing"),
            (["--lean-sampler", "maybe"], "--lean-sampler"),
            (["--lean-losses", "false,2"], "--lean-losses"),
            (["--split-group-conv", "nope"], "--split-group-conv"),
            (["--amp", "fp16"], "--amp"),
            (["--val-pass", "sometimes"], "--val-pass"),
            (["--widths", "f0d2"], "--widths"),
            (["--widths", "custom"], "--widths custom"),
            (["--widths", "custom", "--filters", "16,64,128,192,256", "--decoder-widths", "128,96,64,32,8"],
             "--filters"),
            (["--level", "4", "--cascade", "true"], "--cascade true"),
            (["--isolate", "perhaps"], "--isolate"),
        ]
        for extra, msg in cases:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit, msg=str(extra)):
                    mlm.main(base + extra)
            self.assertIn(msg, err.getvalue(), extra)


if __name__ == "__main__":
    unittest.main()
