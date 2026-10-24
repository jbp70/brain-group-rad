"""Milestone 9 phase 1 (design 2.4 / 5.3): tests/gpu/probe_tensor_limit.py.

* graph mode on the level-2 N = 12 default configuration (no execution): the deform decoder's D2
  group concat [12, 80, 112, 96, 256] (2.64e9 elements) is listed first, above 2^31 - 1, and the
  hypothesis line is printed;
* run mode with a tiny --sizes list on the local GPU: every op and direction has a status and the
  summary has verified_max_elements {fp32, fp16} x {fwd, bwd} (skipped without a GPU);
* m8-errors mode on a small fake JSON, in a subprocess that checks TensorFlow is not imported.
"""
import tests.tf_setup  # noqa: F401  (TF32 off, GPU memory setup) -- must come first

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import tensorflow as tf

from tests.runtime_helpers import free_memory

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(REPO, "tests", "gpu", "probe_tensor_limit.py")
_spec = importlib.util.spec_from_file_location("probe_tensor_limit", _PATH)
ptl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ptl)


class TestProbeTensorLimit(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ptl_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        tf.keras.mixed_precision.set_global_policy("float32")
        free_memory()

    def test_graph_mode_level2_n12(self):
        out = os.path.join(self.tmp, "graph.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ptl.main(["--mode", "graph", "--level", "2", "--n", "12", "--widths", "default",
                           "--task", "rigid", "--out", out])
        text = buf.getvalue()
        print(text)
        self.assertEqual(rc, 0)
        self.assertIn(ptl.HYPOTHESIS, text)
        with open(out) as f:
            res = json.load(f)
        rows = res["large_outputs"]
        self.assertGreaterEqual(len(rows), 1)
        first = rows[0]
        self.assertEqual(first["op_type"], "ConcatV2")
        self.assertIn("deform_decoder/d2", first["scope"])
        self.assertEqual(first["shape"], [12, 80, 112, 96, 256])
        self.assertEqual(first["elements"], 12 * 80 * 112 * 96 * 256)
        self.assertGreater(first["elements"], 2 ** 31 - 1)
        self.assertEqual(first["dtype"], "float32")
        self.assertEqual(first["bytes"], 4 * first["elements"])

    def test_sizes_and_shapes(self):
        self.assertEqual(ptl.parse_sizes("fp32:4100,fp16:4096"), [("fp32", 4096), ("fp16", 4096)])
        with self.assertRaises(ValueError):
            ptl.parse_sizes("fp64:4096")
        for e in [int(x.split(":")[1]) for x in ptl.DEFAULT_SIZES] + [4096]:
            shp = ptl.shape5(e)
            n = 1
            for d in shp:
                n *= d
            self.assertEqual(n, e, shp)
            self.assertEqual(shp[-1], 8)

    def test_verified_max(self):
        recs = [{"dtype": "fp32", "direction": "fwd", "elements": 8, "status": "ok"},
                {"dtype": "fp32", "direction": "fwd", "elements": 16, "status": "ok"},
                {"dtype": "fp32", "direction": "fwd", "elements": 16, "status": "oom"},
                {"dtype": "fp32", "direction": "bwd", "elements": 16, "status": "ok"},
                {"dtype": "fp16", "direction": "fwd", "elements": 8, "status": "skipped_mem"}]
        v = ptl.verified_max(recs)
        self.assertEqual(v, {"fp32": {"fwd": 8, "bwd": 16}, "fp16": {"fwd": None, "bwd": None}})

    @unittest.skipIf(not tf.config.list_physical_devices("GPU"), "needs a GPU")
    def test_run_mode_tiny(self):
        out = os.path.join(self.tmp, "tensor_limit.json")
        rc = ptl.main(["--mode", "run", "--sizes", "fp32:4096,fp16:8192,fp32:%d" % (2 ** 40),
                       "--gpu-memory-limit-mb", "6000", "--out", out])
        self.assertEqual(rc, 0)
        with open(out) as f:
            res = json.load(f)
        recs = res["records"]
        want = {(dt, e, "fwd", op) for dt, e in (("fp32", 4096), ("fp16", 8192), ("fp32", 2 ** 40))
                for op in ptl.FWD_OPS}
        want |= {(dt, e, "bwd", op) for dt, e in (("fp32", 4096), ("fp16", 8192), ("fp32", 2 ** 40))
                 for op in ptl.BWD_OPS}
        self.assertEqual({(r["dtype"], r["elements"], r["direction"], r["op"]) for r in recs}, want)
        for r in recs:
            self.assertIn(r["status"], ("ok", "oom", "error", "skipped_mem"), r)
            if r["elements"] == 2 ** 40:
                self.assertEqual(r["status"], "skipped_mem", r)
            else:
                self.assertEqual(r["status"], "ok", r)
        v = res["verified_max_elements"]
        self.assertEqual(sorted(v), ["fp16", "fp32"])
        for dt in v:
            self.assertEqual(sorted(v[dt]), ["bwd", "fwd"])
        self.assertEqual(v["fp32"], {"fwd": 4096, "bwd": 4096})
        self.assertEqual(v["fp16"], {"fwd": 8192, "bwd": 8192})

    def test_m8_errors_no_tf(self):
        fake = os.path.join(self.tmp, "m8.json")
        res = [{"level": 2, "task": "rigid", "n": 12, "widths": "default", "cascade": True,
                "gradient_checkpointing": "none", "amp": False, "integration_steps": 0, "status": "error",
                "error": "InternalError('Failed to call ThenRnnForward ... d2')"},
               {"level": 2, "task": "deform", "n": 4, "widths": "default", "cascade": True,
                "gradient_checkpointing": "both", "amp": True, "integration_steps": 0, "status": "ok",
                "peak_mb": 14800.5, "warmup_peak_mb": 15000.25}]
        with open(fake, "w") as f:
            json.dump({"environment": {}, "results": res}, f)
        code = ("import importlib.util, sys\n"
                "spec = importlib.util.spec_from_file_location('ptl', %r)\n"
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
                "rc = m.main(['--mode', 'm8-errors', '--m8-json', %r])\n"
                "assert 'tensorflow' not in sys.modules, 'tensorflow imported'\n"
                "sys.exit(rc)\n" % (_PATH, fake))
        r = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, cwd=REPO)
        print(r.stdout)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("2 records, 1 not ok", r.stdout)
        self.assertIn("ThenRnnForward ... d2", r.stdout)
        self.assertIn("14800.5", r.stdout)
        self.assertIn("15000.2", r.stdout)
        missing = subprocess.run([sys.executable, _PATH, "--mode", "m8-errors", "--m8-json",
                                  os.path.join(self.tmp, "nope.json")], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, universal_newlines=True, cwd=REPO)
        self.assertEqual(missing.returncode, 1)


if __name__ == "__main__":
    unittest.main()
