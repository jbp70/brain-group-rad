"""MIRIAD patient-level prediction / evaluation: evaluation/groups.py and stub runs of
slurm/predict_miriad.sh and slurm/eval_miriad.sh. Stdlib + numpy only (no TF); the bash parts are
skipped when bash is missing.

Run alone: python -m unittest tests.test_miriad_eval -v
"""
import contextlib
import csv
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

from evaluation import groups as G

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICT = os.path.join(REPO, "slurm", "predict_miriad.sh")
EVAL = os.path.join(REPO, "slurm", "eval_miriad.sh")
BM_MIRIAD = os.path.join(REPO, "slurm", "brainmorph_miriad.sh")
LOGS = "/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs"
SIF = "/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif"
BASH = shutil.which("bash")
SCANS = ["miriad_188_AD_M_01_MR_1", "miriad_188_AD_M_01_MR_2", "miriad_188_AD_M_02_MR_1",
         "miriad_188_AD_M_03_MR_1", "miriad_188_AD_M_05_MR_1",
         "miriad_241_HC_M_01_MR_1", "miriad_241_HC_M_02_MR_1", "miriad_241_HC_M_04_MR_1",
         "miriad_300_HC_F_01_MR_1"]                                   # single scan: no group file
SCRIPT_ENV = ("CKPT", "TASK", "NAME", "PROCESSED", "OUT", "PATIENTS", "LABEL_NAME", "N_CHECK", "CHECK_CHUNKS",
              "GRID_REF", "LEVELS", "EXTRA", "SCRATCH_ROOT", "METHOD", "PRED", "FWD", "INV", "ATLAS",
              "ATLAS_SEG", "CONV", "IDENTITY", "GROUPS_DIR", "PAIRS", "OVERWRITE", "WORKERS", "PROJ", "REPO",
              "SIF", "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "STUB_FAIL", "KEYMORPH_DIR", "WEIGHTS", "ALIGNS",
              "METHOD_PREFIX")


def _processed(root):
    for s in SCANS:
        d = os.path.join(root, s + "_T1w")
        os.makedirs(d)
        open(os.path.join(d, "image.nii.gz"), "w").close()
    os.makedirs(os.path.join(root, "not_a_scan"))
    return root


def _summary(n, tl, td, pl, pd, cen, sharp):
    return {"n_subjects": n, "n_pairs": n * (n - 1),
            "subject_metrics": {"template_lncc": {"mean": tl, "n": n}, "template_dice": {"mean": td, "n": n},
                                "fold_fwd_pct": {"mean": 0.0, "n": n}},
            "pair_metrics": {"pair_lncc": {"mean": pl, "n": n}, "pair_dice": {"mean": pd, "n": n}},
            "group": {"centrality_mean_mm": cen, "centrality_max_mm": 2 * cen, "atlas_intensity_std": sharp,
                      "mean_logjac_abs": 0.1, "atlas_mask_voxels": 10}}


class TestGroups(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="miriad_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_find_and_write_groups(self):
        p = _processed(os.path.join(self.tmp, "p"))
        g = G.find_groups(p)
        self.assertEqual(sorted(g), ["miriad_188_AD_M", "miriad_241_HC_M", "miriad_300_HC_F"])
        self.assertEqual(g["miriad_188_AD_M"], sorted(s + "_T1w" for s in SCANS[:5]))
        out = os.path.join(self.tmp, "groups")
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(G.main(["write-groups", "--processed-dir", p, "--out-dir", out]), 0)
        self.assertIn("3 groups (2 written, min size 2), 9 scans", buf.getvalue())
        self.assertEqual(sorted(os.listdir(out)), ["groups.csv", "miriad_188_AD_M.txt", "miriad_241_HC_M.txt"])
        with open(os.path.join(out, "miriad_241_HC_M.txt")) as f:
            self.assertEqual(f.read().split(), sorted(s + "_T1w" for s in SCANS[5:8]))
        with open(os.path.join(out, "groups.csv")) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([(r["group"], r["n_scans"], r["written"]) for r in rows],
                         [("miriad_188_AD_M", "5", "1"), ("miriad_241_HC_M", "3", "1"), ("miriad_300_HC_F", "1", "0")])

    def test_write_groups_none_found(self):
        os.makedirs(os.path.join(self.tmp, "empty"))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(G.main(["write-groups", "--processed-dir", os.path.join(self.tmp, "empty"),
                                     "--out-dir", os.path.join(self.tmp, "g")]), 2)

    def test_aggregate_patient_weighting(self):
        root = os.path.join(self.tmp, "eval")
        # Patient sizes differ: the group mean must weight patients equally (not scans).
        for name, s in (("miriad_188_AD_M", _summary(10, 0.6, 0.8, 0.5, 0.7, 1.0, 0.10)),
                        ("miriad_241_HC_M", _summary(2, 0.8, 0.9, 0.7, 0.8, 3.0, 0.20)),
                        ("_aggregate", _summary(2, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0))):   # skipped
            os.makedirs(os.path.join(root, name))
            with open(os.path.join(root, name, "summary.json"), "w") as f:
                json.dump(s, f)
        out = os.path.join(root, "_aggregate")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(G.main(["aggregate", "--root", root, "--method", "m", "--out", out]), 0)
        with open(os.path.join(out, "summary.json")) as f:
            s = json.load(f)
        self.assertEqual(s["n_groups"], 2)
        self.assertEqual(s["n_subjects_total"], 12)
        m = s["metrics"]
        self.assertAlmostEqual(m["template_lncc"]["mean"], 0.7)
        self.assertAlmostEqual(m["pair_dice"]["mean"], 0.75)
        self.assertAlmostEqual(m["centrality_mean_mm"]["mean"], 2.0)
        self.assertAlmostEqual(m["atlas_intensity_std"]["mean"], 0.15)
        for k in ("fold_fwd_pct", "mean_logjac_abs", "atlas_mask_voxels"):
            self.assertNotIn(k, m)
        self.assertEqual(s["by_diagnosis"]["AD"]["n_groups"], 1)
        self.assertAlmostEqual(s["by_diagnosis"]["HC"]["metrics"]["template_dice"]["mean"], 0.9)
        with open(os.path.join(out, "per_group.csv")) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["group"] for r in rows], ["miriad_188_AD_M", "miriad_241_HC_M"])
        self.assertEqual(rows[0]["diagnosis"], "AD")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(G.main(["aggregate", "--root", root, "--method", "m", "--out", out,
                                     "--all-metrics"]), 0)
        with open(os.path.join(out, "summary.json")) as f:
            self.assertIn("fold_fwd_pct", json.load(f)["metrics"])

    def test_aggregate_missing(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(G.main(["aggregate", "--root", self.tmp, "--method", "m",
                                     "--out", os.path.join(self.tmp, "o")]), 2)


def _read(path):
    with open(path) as f:
        return f.read()


class TestStatic(unittest.TestCase):

    def test_headers(self):
        pt_sif = SIF.replace("tf10_nv.sif", "pt_nv.sif")
        for path, part, gpu, sif in ((PREDICT, "rtx6000", True, SIF), (EVAL, "basic", False, SIF),
                                     (BM_MIRIAD, "rtx8000", True, pt_sif)):
            text = _read(path)
            with self.subTest(os.path.basename(path)):
                self.assertNotIn("\r", text)
                self.assertTrue(all(ord(c) < 128 for c in text))
                self.assertNotIn("entre", text.lower())
                for line in ("#SBATCH -A qtim", "#SBATCH -p %s" % part, "#SBATCH -N 1", "#SBATCH --ntasks-per-node=1",
                             "#SBATCH --output=%s/%%x-%%j.out" % LOGS, "#SBATCH --error=%s/%%x-%%j.err" % LOGS):
                    self.assertIn(line + "\n", text)
                self.assertEqual("#SBATCH --gpus=1\n" in text, gpu)
                self.assertRegex(text, r"(?m)^#SBATCH --time=\d\d:\d\d:\d\d$")
                self.assertRegex(text, r"(?m)^#SBATCH --mem=\d+G$")
                self.assertIn(sif, text)
                self.assertIn("DO NOT submit without Jay's approval", text)
                self.assertNotRegex(text, r"(?m)^\s*sbatch ")
                # bash's GROUPS is a special (read-only in effect) variable: never used as a setting.
                self.assertNotRegex(text, r"(?m)^GROUPS=")
                for ln in text.splitlines():
                    if '"${RUN[@]}"' in ln and not ln.lstrip().startswith("#"):
                        self.assertIn("</dev/null", ln)

    @unittest.skipIf(BASH is None, "bash not available")
    def test_bash_syntax(self):
        for path in (PREDICT, EVAL, BM_MIRIAD):
            r = subprocess.run([BASH, "-n", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
            self.assertEqual(r.returncode, 0, r.stderr)


# Stub `singularity`: drop the container options and run the rest (bash -c '...' NAME REPO args) on
# the host. Stub `python`: evaluation.groups runs for real; predict.py / evaluation are logged and
# write their marker outputs (predict.json / summary.json).
STUB_PY = r'''#!%(py)s
import json, os, subprocess, sys
a = sys.argv[1:]
with open(%(log)r, "a") as f:
    f.write(" ".join(a) + "\n")
if a[:2] == ["-m", "evaluation.groups"]:
    sys.exit(subprocess.call([%(py)r] + a))
def opt(k):
    return a[a.index(k) + 1]
fail = os.environ.get("STUB_FAIL", "")
if a[0] == "predict.py" and a[1] == "run":
    out = opt("--out")
    if fail and fail in out:
        sys.exit(3)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "predict.json"), "w").write("{}")
elif a[0].endswith("run_brainmorph.py"):
    pass
elif a[0] == "predict.py":
    open(opt("--out"), "w").write("{}")
elif a[:2] == ["-m", "evaluation"]:
    out = opt("--out")
    os.makedirs(out, exist_ok=True)
    n = len([x for x in open(opt("--subjects-file")) if x.strip()])
    json.dump({"n_subjects": n, "n_pairs": n * (n - 1), "subject_metrics": {"template_lncc": {"mean": 0.5, "n": n}},
               "pair_metrics": {}, "group": {"centrality_mean_mm": 1.0}}, open(os.path.join(out, "summary.json"), "w"))
'''
STUB_SING = '''#!/bin/sh
while [ $# -gt 0 ] && [ "$1" != "bash" ]; do shift; done
exec "$@"
'''


@unittest.skipIf(BASH is None, "bash not available")
class TestStubRuns(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="miriad_stub_")
        self.log = os.path.join(self.tmp, "calls.log")
        bindir = os.path.join(self.tmp, "bin")
        os.makedirs(bindir)
        for name, body in (("python", STUB_PY % {"py": sys.executable, "log": self.log}),
                           ("singularity", STUB_SING), ("git", "#!/bin/sh\nexit 0\n"),
                           ("nvidia-smi", "#!/bin/sh\nexit 0\n")):
            p = os.path.join(bindir, name)
            with open(p, "w") as f:
                f.write(body)
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.processed = _processed(os.path.join(self.tmp, "processed"))
        env = {k: v for k, v in os.environ.items() if k not in SCRIPT_ENV}
        env.update(PATH=bindir + os.pathsep + os.environ.get("PATH", ""), PROJ=self.tmp, REPO=REPO,
                   SIF=os.path.join(self.tmp, "x.sif"), SCRATCH_ROOT=os.path.join(self.tmp, "scratch"),
                   PROCESSED=self.processed, PYTHONPATH=REPO)
        self.env = env

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_script(self, path, **extra):
        env = dict(self.env)
        env.update({k: v for k, v in extra.items() if v is not None})
        if os.path.exists(self.log):
            os.remove(self.log)
        r = subprocess.run([BASH, path], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, stdin=subprocess.DEVNULL)
        calls = _read(self.log).splitlines() if os.path.exists(self.log) else []
        return r, calls

    def test_predict_requires_task(self):
        r, calls = self.run_script(PREDICT, CKPT="c")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("set TASK", r.stderr)
        r, calls = self.run_script(PREDICT, CKPT="c", TASK="deform")
        self.assertEqual(r.returncode, 2)
        self.assertEqual(calls, [])

    def test_predict_then_eval(self):
        out = os.path.join(self.tmp, "results", "predict_miriad", "ours_rigid")
        r, calls = self.run_script(PREDICT, CKPT="/ck/last", TASK="rigid")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(calls[0].split()[:3], ["-m", "evaluation.groups", "write-groups"])
        gate = [c for c in calls if "check-streaming" in c]
        self.assertEqual(len(gate), 1)
        for k in ("--precision float32", "--n-subjects 4", "--branches rigid", "--label-name synthseg",
                  "--chunk-l1 3", "--subjects-file %s/groups/_gate.txt" % out):
            self.assertIn(k, gate[0])
        with open(os.path.join(out, "groups", "_gate.txt")) as f:
            self.assertEqual(f.read().split(), sorted(s + "_T1w" for s in SCANS[:4]))  # first patient with >= 4
        runs = [c for c in calls if c.startswith("predict.py run")]
        self.assertEqual(len(runs), 2)
        for c, g in zip(runs, ("miriad_188_AD_M", "miriad_241_HC_M")):
            for k in ("--branches rigid", "--checkpoint /ck/last", "--subjects-file %s/groups/%s.txt" % (out, g),
                      "--out %s/%s/fields" % (out, g), "--grid-reference %s/data/target_grid/target_grid.json"
                      % self.tmp, "--overwrite"):
                self.assertIn(k, c)
        self.assertIn("2 predicted, 0 already done, 0 failed (of 2)", r.stdout)
        # resume: nothing is predicted again
        r, calls = self.run_script(PREDICT, CKPT="/ck/last", TASK="rigid")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse([c for c in calls if c.startswith("predict.py run")])
        self.assertIn("0 predicted, 2 already done", r.stdout)

        ev_out = os.path.join(self.tmp, "results", "eval_miriad", "ours_rigid")
        r, calls = self.run_script(EVAL, METHOD="ours_rigid")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        ev = [c for c in calls if c.startswith("-m evaluation run")]
        self.assertEqual(len(ev), 2)
        g = "miriad_188_AD_M"
        for k in ("--fwd %s/%s/fields/{subject_dir}_fwd.nii.gz" % (out, g),
                  "--inv %s/%s/fields/{subject_dir}_inv.nii.gz" % (out, g),
                  "--atlas %s/%s/fields/atlas.nii.gz" % (out, g), "--atlas-seg %s/%s/fields/atlas_seg.nii.gz" % (out, g),
                  "--components xyz --signs +++ --units mm", "--label-name synthseg", "--pairs all",
                  "--subjects-file %s/groups/%s.txt" % (out, g), "--out %s/%s" % (ev_out, g)):
            self.assertIn(k, ev[0])
        self.assertNotIn("{group}", " ".join(ev))
        agg = [c for c in calls if "aggregate" in c]
        self.assertEqual(len(agg), 1)
        with open(os.path.join(ev_out, "_aggregate", "summary.json")) as f:
            s = json.load(f)
        self.assertEqual(s["n_groups"], 2)
        self.assertEqual(s["n_subjects_total"], 8)

    def test_predict_failure_continues(self):
        r, calls = self.run_script(PREDICT, CKPT="c", TASK="affine", STUB_FAIL="miriad_188_AD_M")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(len([c for c in calls if c.startswith("predict.py run")]), 2)
        out = os.path.join(self.tmp, "results", "predict_miriad", "ours_affine")
        self.assertIn("miriad_188_AD_M: predict.py rc 3", _read(os.path.join(out, "failed.txt")))
        self.assertTrue(os.path.isfile(os.path.join(out, "miriad_241_HC_M", "fields", "predict.json")))

    def test_predict_patients_subset(self):
        r, calls = self.run_script(PREDICT, CKPT="c", TASK="rigid", PATIENTS="miriad_241_HC_M", N_CHECK="3")
        self.assertEqual(r.returncode, 0, r.stderr)
        runs = [c for c in calls if c.startswith("predict.py run")]
        self.assertEqual(len(runs), 1)
        self.assertIn("miriad_241_HC_M.txt", runs[0])

    def test_brainmorph_miriad(self):
        bm = os.path.join(self.tmp, "baselines", "brainmorph")
        os.makedirs(os.path.join(bm, "keymorph", "keymorph"))
        os.makedirs(os.path.join(bm, "weights"))
        for p in (os.path.join(bm, "keymorph", "keymorph", "model.py"),
                  os.path.join(bm, "weights", "foundation-numkey512-numlevels6.pth.tar")):
            open(p, "w").close()
        r, calls = self.run_script(BM_MIRIAD, PATIENTS="miriad_241_HC_M")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        out = os.path.join(self.tmp, "results", "predict_miriad")
        self.assertEqual(calls[0].split()[:3], ["-m", "evaluation.groups", "write-groups"])
        self.assertTrue(os.path.isfile(os.path.join(out, "_brainmorph_keypoints", "groups", "miriad_241_HC_M.txt")))
        self.assertIn("--self-test", calls[1])
        main = calls[2]
        for k in ("--groups-dir %s/_brainmorph_keypoints/groups" % out, "--aligns rigid affine",
                  "--groups miriad_241_HC_M", "--label-name synthseg", "--num-keypoints 512", "--variant L",
                  "--out %s" % out, "--method-prefix brainmorph"):
            self.assertIn(k, main)
        self.assertIn("METHOD=brainmorph_affine sbatch slurm/eval_miriad.sh", r.stdout)
        r, calls = self.run_script(BM_MIRIAD, WEIGHTS=os.path.join(self.tmp, "missing.pth"))
        self.assertEqual(r.returncode, 2)
        self.assertEqual(calls, [])

    def test_eval_identity(self):
        r, calls = self.run_script(EVAL, METHOD="identity", IDENTITY="1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        ev = [c for c in calls if c.startswith("-m evaluation run")]
        self.assertEqual(len(ev), 2)
        for c in ev:
            self.assertIn("--identity", c)
            for k in ("--fwd", "--atlas", "--components"):
                self.assertNotIn(k, c)
        self.assertTrue(re.search(r"--subjects-file \S+/eval_miriad/identity/groups/miriad_188_AD_M\.txt", ev[0]))


if __name__ == "__main__":
    unittest.main()
