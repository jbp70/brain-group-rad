"""scripts/step_times_from_metrics.py on fake run dirs (milestone 9 design 4.5, 4.6). No TF."""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("step_times_from_metrics",
                                                  os.path.join(REPO, "scripts", "step_times_from_metrics.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ST = _load()


def _write_run(root, name, windows, config_args=None, bad_lines=0):
    run = os.path.join(root, name)
    os.makedirs(os.path.join(run, "logs"))
    with open(os.path.join(run, "logs", "metrics.jsonl"), "w") as fh:
        for step, data in windows:
            fh.write(json.dumps({"step": step, "time": 0.0, "data": data}) + "\n")
        fh.write(json.dumps({"step": 1500, "time": 0.0, "data": {"event": "bn_recalibration"}}) + "\n")
        for _ in range(bad_lines):
            fh.write("{not json\n")
    if config_args is not None:
        with open(os.path.join(run, "config.json"), "w") as fh:
            json.dump({"model_config": {}, "args": config_args}, fh)
    return run


class TestStepTimes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgr_step_times_")
        # Cascade-style run: per-task window means and the u_res diagnostic (25-step windows).
        w = []
        for i, step in enumerate(range(25, 2025, 25)):
            d = {"time/compute_ms": 2000.0 + i, "train_rigid/compute_ms": 700.0 + i,
                 "train_deform/compute_ms": 4900.0 + 2 * i, "train_deform/diag_u_res_max_vox": 1.0 + 0.01 * i}
            w.append((step, d))
        cls.cascade = _write_run(cls.tmp, "cascade", w, {"group_size": 16, "global_start_step": 2000}, bad_lines=2)
        cls.plain = _write_run(cls.tmp, "plain", [(s, {"time/compute_ms": 1650.0}) for s in range(25, 2025, 25)])
        cls.missing = os.path.join(cls.tmp, "does_not_exist")
        cls.out = os.path.join(cls.tmp, "out", "step_times.json")
        cls.rc = ST.main(["--runs", cls.cascade, cls.plain, cls.missing, "--min-step", "1000",
                          "--extra-keys", "diag_u_res_max_vox", "--args-keys", "group_size,global_start_step,lr",
                          "--out", cls.out])
        with open(cls.out) as fh:
            cls.res = json.load(fh)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_no_tensorflow(self):
        if "tensorflow" not in sys.modules:          # alone / bgr_preproc38 (stage 1 of run_tests_wsl.sh)
            return
        # Full-suite discovery: other test modules import TF first, so check the module's imports.
        import ast
        tree = ast.parse(open(os.path.join(REPO, "scripts", "step_times_from_metrics.py")).read())
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        names += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
        self.assertFalse(any(x.split(".")[0] == "tensorflow" for x in names))

    def test_exit_zero_and_runs(self):
        self.assertEqual(self.rc, 0)
        self.assertEqual([r["run"] for r in self.res["runs"]], [self.cascade, self.plain, self.missing])

    def test_cascade_per_task(self):
        r = self.res["runs"][0]
        self.assertTrue(r["per_task"])
        # Windows with step >= 1000: steps 1000..2000 -> indices 39..79 (41 windows); events excluded.
        self.assertEqual(r["n_windows"], 41)
        self.assertEqual(r["unparsable_lines"], 2)
        t = r["timing"]
        self.assertEqual(set(t), {"time/compute_ms", "train_rigid/compute_ms", "train_deform/compute_ms"})
        self.assertAlmostEqual(t["train_rigid/compute_ms"]["median"], 700.0 + 59)
        self.assertAlmostEqual(t["train_deform/compute_ms"]["median"], 4900.0 + 2 * 59)
        self.assertEqual(t["train_rigid/compute_ms"]["n_windows"], 41)
        self.assertAlmostEqual(t["train_rigid/compute_ms"]["iqr"][0], 700.0 + 49)
        self.assertAlmostEqual(t["train_rigid/compute_ms"]["iqr"][1], 700.0 + 69)
        ex = r["extra"]["diag_u_res_max_vox"]
        self.assertAlmostEqual(ex["max"], 1.0 + 0.01 * 79)
        self.assertAlmostEqual(ex["median"], 1.0 + 0.01 * 59)
        self.assertEqual(ex["keys"], ["train_deform/diag_u_res_max_vox"])
        self.assertEqual(r["args"], {"group_size": 16, "global_start_step": 2000, "lr": None})

    def test_plain_mixed_only(self):
        r = self.res["runs"][1]
        self.assertFalse(r["per_task"])
        self.assertEqual(set(r["timing"]), {"time/compute_ms"})
        self.assertAlmostEqual(r["timing"]["time/compute_ms"]["median"], 1650.0)
        self.assertIsNone(r["extra"]["diag_u_res_max_vox"])
        self.assertEqual(r["config"], "missing")
        self.assertEqual(r["args"], {"group_size": None, "global_start_step": None, "lr": None})

    def test_missing_run(self):
        r = self.res["runs"][2]
        self.assertEqual((r["status"], r["metrics"], r["config"]), ("missing", "missing", "missing"))
        self.assertIsNone(r["extra"]["diag_u_res_max_vox"])


if __name__ == "__main__":
    unittest.main()
