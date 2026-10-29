"""fp32-decoder-width FEAT-003: scripts/collect_memory_runs.py (stdlib collector of the m11 memory
job). No TF: fake run directories (metrics.jsonl, config.json, logs) in a temp dir."""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "collect_memory_runs.py")
HEADER = "name\tlevel\tdtype\twidths\tn\tmax_pairs\trc\telapsed_s\tlog\n"


def _load():
    spec = importlib.util.spec_from_file_location("collect_memory_runs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CM = _load()


class TestCollectMemoryRuns(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="collect_mem_")
        self.runs = os.path.join(self.tmp, "runs")
        os.makedirs(self.runs)
        self.rows = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_run(self, name, linear_steps=(), peaks=None, args=None, counts=None, malformed=False):
        logs = os.path.join(self.runs, name, "logs")
        os.makedirs(logs)
        with open(os.path.join(logs, "metrics.jsonl"), "w") as f:
            if malformed:
                f.write("{not json\n")
            f.write(json.dumps({"step": 0, "time": 0.0, "data": {"event": "start"}}) + "\n")
            for step in range(1, 9):
                d = {"time/data_ms": 100.0 * step, "time/compute_ms": 1000.0 * step,
                     "gpu_peak_mb": (peaks or {}).get(step, 1000.0 + step), "nonfinite_total": step // 4}
                if step in linear_steps:
                    d["train_rigid/loss"] = 1.0
                else:
                    d["train_deform/n_pairs"] = 4 + (step % 3)
                f.write(json.dumps({"step": step, "time": float(step), "data": d}) + "\n")
        with open(os.path.join(self.runs, name, "config.json"), "w") as f:
            json.dump({"args": args or {"filters": None, "decoder_widths": None},
                       "param_counts": counts or {"trainable": 29630000, "deform_decoder": 1500000}}, f)

    def add_row(self, name, rc, log_text="", level=1, dtype="float32", widths="default", n=4, max_pairs=0):
        log = os.path.join(self.tmp, name + ".log")
        with open(log, "w") as f:
            f.write(log_text)
        self.rows.append("\t".join(str(x) for x in (name, level, dtype, widths, n, max_pairs, rc, 321, log)))

    def collect(self, *extra):
        status = os.path.join(self.tmp, "status.tsv")
        with open(status, "w") as f:
            f.write(HEADER + "".join(r + "\n" for r in self.rows))
        out = os.path.join(self.tmp, "out", "fp32_width_memory.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = CM.main(["--status", status, "--runs", self.runs, "--out", out, "--job", "123"] + list(extra))
        self.assertEqual(rc, 0)
        with open(out) as f:
            res = json.load(f)
        return res, {r["name"]: r for r in res["records"]}, buf.getvalue()

    def test_ok_record(self):
        self.make_run("l1_f32_d12h", peaks={5: 33000.0},
                      args={"filters": [16, 64, 128, 192, 256, 384], "decoder_widths": [128, 96, 64, 16, 8]},
                      malformed=True)
        self.add_row("l1_f32_d12h", 0, "step 8 done\n", widths="d12h")
        res, recs, out = self.collect()
        self.assertEqual((res["job"], res["skip_steps"]), ("123", 2))
        r = recs["l1_f32_d12h"]
        self.assertEqual(r["status"], "ok")
        self.assertEqual((r["level"], r["dtype"], r["widths"], r["n"], r["max_pairs"], r["rc"], r["elapsed_s"]),
                         (1, "float32", "d12h", 4, 0, 0, 321))
        self.assertEqual(r["steps_logged"], 8)
        self.assertEqual(r["gpu_peak_mb"], 33000.0)
        self.assertEqual(r["gpu_peak_gib"], round(33000.0 / 1024, 2))
        # steps 3..8: (100 s + 1000 s) * step / 1000 -> 1.1 * mean(3..8) = 1.1 * 5.5
        self.assertAlmostEqual(r["mean_deform_step_s"], 1.1 * 5.5, places=3)
        self.assertIsNone(r["mean_linear_step_s"])
        self.assertEqual(r["pairs"], 6)
        self.assertEqual(r["nonfinite_total"], 2)
        self.assertEqual((r["param_count"], r["deform_decoder_params"]), (29630000, 1500000))
        self.assertEqual(r["filters"], [16, 64, 128, 192, 256, 384])
        self.assertEqual(r["decoder_widths"], [128, 96, 64, 16, 8])
        self.assertIsNone(r["error"])
        self.assertIn("l1_f32_d12h", out)
        self.assertIn("peakGiB", out)
        self.assertIn("32.23", out)
        self.assertIn("29.63", out)

    def test_group_size_fallback_row(self):
        """m11 fallback names (_n<N>) keep the job's real N from status.tsv; the table shows the name."""
        self.make_run("l4_f32fine_d12hs_n48")
        self.add_row("l4_f32fine_d12hs_n48", 0, level=4, dtype="float32_fine", widths="d12hs", n=48, max_pairs=64)
        _, recs, out = self.collect()
        r = recs["l4_f32fine_d12hs_n48"]
        self.assertEqual((r["level"], r["n"], r["max_pairs"], r["widths"], r["status"]), (4, 48, 64, "d12hs", "ok"))
        self.assertIn("l4_f32fine_d12hs_n48 ", out)

    def test_skip_steps_and_linear_records(self):
        self.make_run("mix", linear_steps=(4, 6))
        self.add_row("mix", 0)
        _, recs, _ = self.collect("--skip-steps", "4")
        r = recs["mix"]
        # deform steps > 4: 5, 7, 8; linear steps > 4: 6
        self.assertAlmostEqual(r["mean_deform_step_s"], 1.1 * (5 + 7 + 8) / 3.0, places=3)
        self.assertAlmostEqual(r["mean_linear_step_s"], 1.1 * 6, places=3)
        self.assertEqual(r["pairs"], 6)

    def test_failure_classification(self):
        for name in ("oom", "ok_bfc_warning", "timeout", "abort"):
            self.make_run(name)
        self.add_row("oom", 1, "x\ntensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when "
                               "allocating tensor with shape[4,160,224,192,64]\nend\n")
        self.add_row("ok_bfc_warning", 0, "Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.1GiB; "
                                          "the caller indicates that this is not a failure\n")
        self.add_row("timeout", 124, "still running\n")
        self.add_row("abort", 134, "Fatal Python error: Aborted\n")
        self.add_row("skipped", -1)
        self.add_row("no_source", -2)
        _, recs, out = self.collect()
        self.assertEqual({k: v["status"] for k, v in recs.items()},
                         {"oom": "oom", "ok_bfc_warning": "ok", "timeout": "timeout", "abort": "abort",
                          "skipped": "skipped_time", "no_source": "skipped_source"})
        self.assertTrue(recs["oom"]["error"].startswith("tensorflow.python.framework.errors_impl.ResourceExhausted"))
        self.assertEqual(recs["abort"]["error"], "Fatal Python error: Aborted")
        self.assertIn("skipped_time", out)

    def test_oom_without_run_dir(self):
        self.add_row("gone", 1, "RuntimeError: CUDA_ERROR_OUT_OF_MEMORY: out of memory\n")
        self.add_row("gone_abort", 134, "")
        _, recs, _ = self.collect()
        r = recs["gone"]
        self.assertEqual(r["status"], "oom")
        for k in ("pairs", "gpu_peak_mb", "gpu_peak_gib", "mean_deform_step_s", "mean_linear_step_s",
                  "nonfinite_total", "param_count", "deform_decoder_params", "filters", "decoder_widths"):
            self.assertIsNone(r[k], k)
        self.assertEqual(r["steps_logged"], 0)
        self.assertEqual(recs["gone_abort"]["status"], "abort")
        self.assertEqual(r["error"], "RuntimeError: CUDA_ERROR_OUT_OF_MEMORY: out of memory")

    def test_rc0_without_steps_is_abort(self):
        self.add_row("empty", 0, "")
        _, recs, _ = self.collect()
        self.assertEqual(recs["empty"]["status"], "abort")

    def test_long_error_truncated(self):
        self.add_row("long", 1, "ValueError: " + "x" * 1000 + "\n")
        _, recs, _ = self.collect()
        self.assertEqual(len(recs["long"]["error"]), CM.ERROR_MAX_CHARS)

    def test_missing_status_file(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = CM.main(["--status", os.path.join(self.tmp, "missing.tsv"), "--runs", self.runs,
                          "--out", os.path.join(self.tmp, "o.json")])
        self.assertEqual(rc, 2)
        self.assertIn("not found", err.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "o.json")))


if __name__ == "__main__":
    unittest.main()
