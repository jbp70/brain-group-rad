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

import nibabel as nib
import numpy as np

from evaluation import fields as F
from evaluation import groups as G
from evaluation.evaluate import main as eval_main

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICT = os.path.join(REPO, "slurm", "predict_miriad.sh")
EVAL = os.path.join(REPO, "slurm", "eval_miriad.sh")
BM_MIRIAD = os.path.join(REPO, "slurm", "brainmorph_miriad.sh")
SM_MIRIAD = os.path.join(REPO, "slurm", "synthmorph_miriad.sh")
ANTS_MIRIAD = os.path.join(REPO, "slurm", "ants_miriad.sh")
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
              "METHOD_PREFIX", "MATRICES", "FREESURFER_HOME", "TYPE", "ITERATIONS", "PARALLEL_GROUPS",
              "ANTS_ARGS", "KEEP_WORK", "SKIP_SELFTEST", "ANTSPATH", "TMPDIR")


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

    def test_compare(self):
        root = os.path.join(self.tmp, "eval")
        pats = ["miriad_%d_%s_M" % (i, "AD" if i % 2 else "HC") for i in range(10)]
        rng = np.random.RandomState(0)
        for meth, gain in (("identity", 0.0), ("bm_rigid", 0.1)):
            d = os.path.join(root, meth, "_aggregate")
            os.makedirs(d)
            plist = pats + (["miriad_99_AD_M"] if meth == "bm_rigid" else [])   # extra patient left out
            with open(os.path.join(d, "per_group.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["group", "diagnosis", "n_subjects", "template_lncc", "pair_dice", "atlas_intensity_std",
                            "centrality_mean_mm"])
                for i, p in enumerate(plist):
                    base = 0.5 + 0.01 * i
                    w.writerow([p, G._diagnosis(p), 5, base + gain + 0.001 * rng.rand(), base + gain,
                                0.2 - gain / 2, 0.0 if meth == "identity" else 0.5])
        out = os.path.join(root, "_compare")
        with contextlib.redirect_stdout(io.StringIO()) as buf, contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(G.main(["compare", "--root", root, "--methods", "identity", "bm_rigid",
                                     "--out", out]), 0)
        self.assertIn("miriad_99_AD_M", err.getvalue())
        with open(os.path.join(out, "compare.csv")) as f:
            rows = list(csv.DictReader(f))
        r = {(x["subset"], x["metric"], x["method"]): x for x in rows}
        lncc = r[("all", "template_lncc", "bm_rigid")]
        self.assertEqual(lncc["n"], "10")
        self.assertAlmostEqual(float(lncc["diff_vs_ref"]), 0.1, places=2)
        self.assertEqual(float(lncc["frac_better"]), 1.0)
        self.assertLess(float(lncc["p_wilcoxon"]), 0.01)
        sharp = r[("all", "atlas_intensity_std", "bm_rigid")]
        self.assertEqual(sharp["better"], "lower")
        self.assertEqual(float(sharp["frac_better"]), 1.0)     # lower std = better
        self.assertEqual(r[("AD", "pair_dice", "identity")]["n"], "5")
        self.assertNotIn("diff_vs_ref", {k for k, v in r[("all", "pair_dice", "identity")].items() if v})
        md = _read(os.path.join(out, "compare.md"))
        for k in ("### all (10 patients)", "### AD (5 patients)", "template_lncc (higher)",
                  "bm_rigid - identity"):
            self.assertIn(k, md)
        self.assertIn("### all", buf.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(G.main(["compare", "--root", root, "--methods", "identity", "nope", "--out", out]), 2)

    def test_aggregate_missing(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(G.main(["aggregate", "--root", self.tmp, "--method", "m",
                                     "--out", os.path.join(self.tmp, "o")]), 2)


def _read(path):
    with open(path) as f:
        return f.read()


class TestSynthMorphGroups(unittest.TestCase):
    """baselines/synthmorph groups mode with a stand-in detector (numpy only; the real models run
    in FreeSurfer's fspython on the cluster)."""

    def test_groups_mode(self):
        from baselines.synthmorph import run_synthmorph_groupwise as SM
        tmp = tempfile.mkdtemp(prefix="sm_groups_")
        try:
            rng = np.random.RandomState(3)
            shape = (30, 34, 28)
            proc = os.path.join(tmp, "processed")
            x, y, z = np.indices(shape).astype(np.float32)
            c = (np.asarray(shape) - 1) / 2.0
            r = np.sqrt(((x - c[0]) / 10) ** 2 + ((y - c[1]) / 12) ** 2 + ((z - c[2]) / 9) ** 2)
            img = (np.clip(0.5 + 0.3 * np.cos(r * 5), 0.05, None) * (r < 1)).astype(np.float32)
            lab = np.where(r < 1, np.where(x < c[0], 2, 41), 0).astype(np.int32)
            names, maps = [], []
            for i in range(4):
                m = np.eye(4)
                m[:3, :3] += rng.uniform(-0.08, 0.08, (3, 3))
                m[:3, 3] = c - m[:3, :3] @ c + rng.uniform(-2, 2, 3)
                name = "S%d_T1w" % i
                os.makedirs(os.path.join(proc, name))
                nib.save(nib.Nifti1Image(img, np.eye(4)), os.path.join(proc, name, "image.nii.gz"))
                nib.save(nib.Nifti1Image(lab, np.eye(4)), os.path.join(proc, name, "labels_dkt31aseg.nii.gz"))
                names.append(name)
                maps.append(m)
            kp = rng.uniform(4, np.asarray(shape) - 4, (30, 3))
            for name, cond, val in SM.groups_mode_check(tmp, proc, names, maps, kp):
                self.assertTrue(cond, "%s: %s" % (name, val))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestMatrices(unittest.TestCase):
    """evaluation --matrices: fields built from 4x4 voxel matrices == the same fields stored densely."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="miriad_mat_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _rigid(rng, shape):
        a = rng.uniform(-0.1, 0.1, 3)
        cx, cy, cz = np.cos(a)
        sx, sy, sz = np.sin(a)
        r = (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
             @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))
        c = (np.asarray(shape) - 1) / 2.0
        m = np.eye(4)
        m[:3, :3] = r
        m[:3, 3] = c - r @ c + rng.uniform(-2, 2, 3)
        return m

    def test_matrix_field_and_load(self):
        shape = (6, 7, 5)
        m = self._rigid(np.random.RandomState(0), shape)
        u = F.matrix_field(m, shape)
        x = np.array([2.0, 3.0, 1.0])
        np.testing.assert_allclose(x + u[2, 3, 1], (m @ np.r_[x, 1])[:3], atol=1e-5)
        p = os.path.join(self.tmp, "m.npz")
        np.savez(p, subjects=np.array(["a_T1w", "b_T1w"]), fwd_vox=np.stack([m, np.eye(4)]))
        mats = F.load_matrices(p)
        np.testing.assert_allclose(mats["a_T1w"][1], np.linalg.inv(m))
        np.savez(p, subjects=np.array(["a_T1w"]), fwd_vox=m[None], inv_vox=np.eye(4)[None])
        with self.assertRaises(ValueError):
            F.load_matrices(p)
        with self.assertRaises(ValueError):          # neither templates nor matrices
            F.FieldSource(None, None, F.Convention(), shape, (1, 1, 1))
        np.savez(p, subjects=np.array(["a_T1w"]), fwd_vox=m[None])
        with self.assertRaises(ValueError):          # identity and matrices are exclusive
            F.FieldSource(None, None, F.Convention(), shape, (1, 1, 1), identity=True, matrices=p)
        src = F.FieldSource(None, None, F.Convention(), shape, (1, 1, 1), matrices=p)
        np.testing.assert_allclose(src.get("fwd", {"subject": "a", "dir": "a_T1w"}), u)
        with self.assertRaises(KeyError):
            src.get("fwd", {"subject": "b", "dir": "b_T1w"})

    def test_evaluation_matrices_equal_dense(self):
        shape = (20, 22, 18)
        aff = np.eye(4)
        rng = np.random.RandomState(1)
        x, y, z = np.indices(shape).astype(np.float32)
        c = (np.asarray(shape) - 1) / 2.0
        r = np.sqrt(((x - c[0]) / 7) ** 2 + ((y - c[1]) / 8) ** 2 + ((z - c[2]) / 6) ** 2)
        base = (np.clip(0.5 + 0.3 * np.cos(r * 5) + 0.1 * (x - c[0]) / 7, 0.05, None) * (r < 1)).astype(np.float32)
        lab0 = np.where(r < 1, np.where(x < c[0], 2, 41), 0).astype(np.uint16)
        lab0[(r < 1) & (r > 0.6) & (y < c[1])] = 1002
        proc, fld = os.path.join(self.tmp, "proc"), os.path.join(self.tmp, "fields")
        os.makedirs(fld)
        mats, names = [], []
        for i in range(3):
            m = self._rigid(rng, shape)   # fwd: atlas voxel -> subject voxel
            u_sub2atlas = F.matrix_field(np.linalg.inv(m), shape)
            name = "S%d_T1w" % i
            d = os.path.join(proc, name)
            os.makedirs(d)
            nib.save(nib.Nifti1Image(F.warp(base, u_sub2atlas, 1), aff), os.path.join(d, "image.nii.gz"))
            nib.save(nib.Nifti1Image(F.warp(lab0, u_sub2atlas, 0), aff), os.path.join(d, "labels_synthseg.nii.gz"))
            nib.save(nib.Nifti1Image(F.matrix_field(m, shape), aff), os.path.join(fld, name + "_fwd.nii.gz"))
            nib.save(nib.Nifti1Image(F.matrix_field(np.linalg.inv(m), shape), aff),
                     os.path.join(fld, name + "_inv.nii.gz"))
            mats.append(m)
            names.append(name)
        np.savez(os.path.join(fld, "matrices.npz"), subjects=np.array(names), fwd_vox=np.stack(mats))
        common = ["run", "--processed-dir", proc, "--label-name", "synthseg", "--method", "m", "--pairs", "all"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(eval_main(common + ["--out", os.path.join(self.tmp, "dense"),
                                                 "--fwd", fld + "/{subject_dir}_fwd.nii.gz",
                                                 "--inv", fld + "/{subject_dir}_inv.nii.gz"]), 0)
            self.assertEqual(eval_main(common + ["--out", os.path.join(self.tmp, "mat"),
                                                 "--matrices", os.path.join(fld, "matrices.npz")]), 0)
        s = {}
        for k in ("dense", "mat"):
            with open(os.path.join(self.tmp, k, "summary.json")) as f:
                s[k] = json.load(f)
        self.assertEqual(s["mat"]["matrices"], os.path.join(fld, "matrices.npz"))
        self.assertIsNone(s["mat"]["convention"])
        for sec in ("subject_metrics", "pair_metrics"):
            for k in ("template_lncc", "template_dice", "pair_lncc", "pair_dice"):
                if k in s["dense"][sec]:
                    self.assertAlmostEqual(s["mat"][sec][k]["mean"], s["dense"][sec][k]["mean"], places=5)
        for k in ("centrality_mean_mm", "atlas_intensity_std"):
            self.assertAlmostEqual(s["mat"]["group"][k], s["dense"]["group"][k], places=5)
        self.assertGreater(s["mat"]["pair_metrics"]["pair_dice"]["mean"], 0.85)


class TestStatic(unittest.TestCase):

    def test_headers(self):
        pt_sif = SIF.replace("tf10_nv.sif", "pt_nv.sif")
        for path, part, gpu, sif in ((PREDICT, "rtx6000", True, SIF), (EVAL, "basic", False, SIF),
                                     (BM_MIRIAD, "rtx8000", True, pt_sif),
                                     (SM_MIRIAD, "rtx8000", True, "$FREESURFER_HOME/bin/fspython"),
                                     (ANTS_MIRIAD, "basic", False, SIF)):
            text = _read(path)
            with self.subTest(os.path.basename(path)):
                self.assertNotIn("\r", text)
                self.assertTrue(all(ord(c) < 128 for c in text))
                self.assertNotIn("entre", text.lower())
                for line in ("#SBATCH -A qtim", "#SBATCH -p %s" % part, "#SBATCH -N 1", "#SBATCH --ntasks-per-node=1",
                             "#SBATCH --output=%s/%%x-%%j.out" % LOGS, "#SBATCH --error=%s/%%x-%%j.err" % LOGS):
                    self.assertIn(line + "\n", text)
                self.assertEqual("#SBATCH --gpus=1\n" in text, gpu)
                self.assertRegex(text, r"(?m)^#SBATCH --time=\d{2,3}:\d\d:\d\d$")
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
        for path in (PREDICT, EVAL, BM_MIRIAD, SM_MIRIAD, ANTS_MIRIAD):
            r = subprocess.run([BASH, "-n", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
            self.assertEqual(r.returncode, 0, r.stderr)


# Stub `singularity`: drop the container options and run the rest (bash -c '...' NAME REPO args) on
# the host. Stub `python`: evaluation.groups runs for real; predict.py / evaluation are logged and
# write their marker outputs (predict.json / summary.json).
STUB_PY = r'''#!%(py)s
import json, os, subprocess, sys
a = sys.argv[1:]
if a and a[0] == "-u":
    a = a[1:]
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
elif a[0].endswith(("run_brainmorph.py", "run_synthmorph_groupwise.py", "run_ants_groupwise.py")):
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
                   PROCESSED=self.processed, PYTHONPATH=REPO, TMPDIR=os.path.join(self.tmp, "tmpdir"))
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

    def test_eval_uses_matrices_when_no_dense_fields(self):
        pred = os.path.join(self.tmp, "results", "predict_miriad", "brainmorph_rigid")
        gdir = os.path.join(pred, "groups")
        with contextlib.redirect_stdout(io.StringIO()):
            G.main(["write-groups", "--processed-dir", self.processed, "--out-dir", gdir])
        for g in ("miriad_188_AD_M", "miriad_241_HC_M"):
            os.makedirs(os.path.join(pred, g, "fields"))
            open(os.path.join(pred, g, "fields", "matrices.npz"), "w").close()
        open(os.path.join(pred, "miriad_241_HC_M", "fields", "miriad_241_HC_M_01_MR_1_T1w_fwd.nii.gz"), "w").close()
        r, calls = self.run_script(EVAL, METHOD="brainmorph_rigid")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        ev = [c for c in calls if c.startswith("-m evaluation run")]
        self.assertIn("--matrices %s/miriad_188_AD_M/fields/matrices.npz" % pred, ev[0])
        for k in ("--fwd", "--components"):
            self.assertNotIn(k, ev[0])
        self.assertIn("--atlas %s/miriad_188_AD_M/fields/atlas.nii.gz" % pred, ev[0])
        self.assertNotIn("--matrices", ev[1])             # dense fwd files present -> dense path
        self.assertIn("--fwd", ev[1])
        r, calls = self.run_script(EVAL, METHOD="brainmorph_rigid", OVERWRITE="1", MATRICES="none")
        self.assertFalse([c for c in calls if "--matrices" in c])

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

    def test_synthmorph_miriad(self):
        fs = os.path.join(self.tmp, "fs")
        os.makedirs(os.path.join(fs, "bin"))
        os.makedirs(os.path.join(fs, "models"))
        for m in ("synthmorph.rigid.1.h5", "synthmorph.affine.2.h5"):
            open(os.path.join(fs, "models", m), "w").close()
        fspy = os.path.join(fs, "bin", "fspython")
        with open(fspy, "w") as f:   # drop -u, then behave like the stub python
            f.write('#!/bin/sh\n[ "$1" = "-u" ] && shift\nexec python "$@"\n')
        os.chmod(fspy, 0o755)
        r, calls = self.run_script(SM_MIRIAD, FREESURFER_HOME=fs, PATIENTS="miriad_188_AD_M")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        out = os.path.join(self.tmp, "results", "predict_miriad")
        self.assertEqual(calls[0].split()[:3], ["-m", "evaluation.groups", "write-groups"])
        self.assertTrue(os.path.isfile(os.path.join(out, "_synthmorph_keypoints", "groups", "miriad_188_AD_M.txt")))
        self.assertIn("--self-test", calls[1])
        for k in ("--groups-dir %s/_synthmorph_keypoints/groups" % out, "--aligns rigid affine",
                  "--groups miriad_188_AD_M", "--label-name synthseg", "--out %s" % out,
                  "--method-prefix synthmorph", "--num-iters 5"):
            self.assertIn(k, calls[2])
        self.assertIn("METHOD=synthmorph_rigid sbatch slurm/eval_miriad.sh", r.stdout)
        os.remove(os.path.join(fs, "models", "synthmorph.rigid.1.h5"))
        r, calls = self.run_script(SM_MIRIAD, FREESURFER_HOME=fs)
        self.assertEqual(r.returncode, 2)
        self.assertEqual(calls, [])

    def test_ants_miriad(self):
        r, calls = self.run_script(ANTS_MIRIAD, TYPE="affine", PATIENTS="miriad_188_AD_M", ITERATIONS="1",
                                   SLURM_CPUS_PER_TASK="24")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        out = os.path.join(self.tmp, "results", "predict_miriad")
        self.assertEqual(calls[0].split()[:3], ["-m", "evaluation.groups", "write-groups"])
        self.assertTrue(os.path.isfile(os.path.join(out, "_ants_work", "groups", "miriad_188_AD_M.txt")))
        self.assertIn("--self-test --self-test-linear-only", calls[1])
        for k in ("--type affine", "--groups-dir %s/_ants_work/groups" % out, "--groups miriad_188_AD_M",
                  "--label-name synthseg", "--iterations 1", "--cpus 24", "--parallel-groups 2", "--out %s" % out):
            self.assertIn(k, calls[2])
        self.assertNotIn("--keep-work", calls[2])
        self.assertIn("METHOD=ants_affine sbatch slurm/eval_miriad.sh", r.stdout)
        r, calls = self.run_script(ANTS_MIRIAD, TYPE="deformable")
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
