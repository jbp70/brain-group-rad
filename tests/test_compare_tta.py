"""T13 (TTA design 9.3, FEAT-004): scripts/compare_tta.py paired statistics. numpy + scipy only, no TF.

Run alone: python -m unittest tests.test_compare_tta -v
"""
import contextlib
import csv
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from tests import test_evaluation as TE

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("compare_tta", os.path.join(REPO, "scripts", "compare_tta.py"))
CT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CT)


def _write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = CT.main(argv)
    return rc, out.getvalue(), err.getvalue()


def _by_metric(res):
    return {r["metric"]: r for r in res["primary"] + res["secondary"] + res["per_label"] if not r.get("missing")}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_cmp_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def subjects_csv(self, arm, subjects, cols):
        names = list(cols)
        _write(os.path.join(self.tmp, arm, "per_subject.csv"), ["subject"] + names,
               [[s] + ["%.9g" % cols[c][k] for c in names] for k, s in enumerate(subjects)])

    def compare(self, *extra, a="A", b="B", out="out"):
        rc, so, se = _run(["--a", os.path.join(self.tmp, a), "--b", os.path.join(self.tmp, b),
                           "--out", os.path.join(self.tmp, out), "--n-boot", "2000"] + list(extra))
        res = None
        if rc == 0:
            with open(os.path.join(self.tmp, out, "compare.json")) as f:
                res = json.load(f)
        return rc, res, so + se


class TestSubjectStats(Base):
    def setUp(self):
        super().setUp()
        rng = np.random.RandomState(0)
        self.subs = ["S%02d" % k for k in range(30)]
        self.a = {"template_dice": 0.7 + 0.01 * rng.randn(30), "fold_fwd_pct": 0.001 + 0.0001 * rng.rand(30),
                  "inv_err_fwd_mm": 0.5 + 0.05 * rng.rand(30)}
        self.noise = 0.002 * rng.randn(30)

    def test_known_shift(self):
        b = dict(self.a, template_dice=self.a["template_dice"] + 0.01 + self.noise)
        self.subjects_csv("A", self.subs, self.a)
        self.subjects_csv("B", self.subs, b)
        rc, res, txt = self.compare()
        self.assertEqual(rc, 0, txt)
        r = _by_metric(res)["template_dice"]
        self.assertEqual(r["family"], "primary")
        self.assertEqual(r["n"], 30)
        self.assertAlmostEqual(r["mean_diff"], 0.01 + float(self.noise.mean()), places=6)
        self.assertLess(r["ci_low"], 0.01)
        self.assertGreater(r["ci_high"], 0.01)
        self.assertGreater(r["ci_low"], 0.0)
        self.assertLess(r["wilcoxon_p"], 0.05)
        self.assertLess(r["holm_p"], 0.05)
        self.assertEqual((r["improved"], r["worse"], r["tied"]), (30, 0, 0))
        self.assertGreater(r["d_z"], 1.0)
        self.assertAlmostEqual(r["rank_biserial"], 1.0)
        self.assertEqual(r["sesoi"], 0.002)
        self.assertIn("measurable impact (B better); practically relevant", r["decision"])
        # unchanged metrics: zero differences, no impact; P2 marked as needing the all-pairs evaluation
        fold = _by_metric(res)["fold_fwd_pct"]
        self.assertEqual(fold["mean_diff"], 0.0)
        self.assertAlmostEqual(fold["sesoi"], max(0.25 * float(np.mean(self.a["fold_fwd_pct"])), 0.0002))
        p2 = [r for r in res["primary"] if r["metric"] == "pair_dice"][0]
        self.assertTrue(p2["missing"])
        self.assertEqual(p2["decision"], "needs all-pairs evaluation")
        self.assertEqual(_by_metric(res)["inv_err_fwd_mm"]["family"], "secondary")
        with open(os.path.join(self.tmp, "out", "compare.md")) as f:
            md = f.read()
        self.assertIn(CT.SCOPE, md)
        with open(os.path.join(self.tmp, "out", "compare.csv")) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(sorted(r["metric"] for r in rows), ["fold_fwd_pct", "inv_err_fwd_mm", "template_dice"])

    def test_lower_is_better_direction(self):
        b = dict(self.a, inv_err_fwd_mm=self.a["inv_err_fwd_mm"] - 0.1)
        self.subjects_csv("A", self.subs, self.a)
        self.subjects_csv("B", self.subs, b)
        rc, res, txt = self.compare()
        self.assertEqual(rc, 0, txt)
        r = _by_metric(res)["inv_err_fwd_mm"]
        self.assertEqual((r["direction"], r["improved"], r["worse"]), ("lower", 30, 0))
        self.assertIn("measurable impact (B better)", r["decision"])
        self.assertIsNone(r["sesoi"])

    def test_identical_inputs(self):
        self.subjects_csv("A", self.subs, self.a)
        self.subjects_csv("B", self.subs, self.a)
        rc, res, txt = self.compare()
        self.assertEqual(rc, 0, txt)
        for r in _by_metric(res).values():
            self.assertEqual(r["mean_diff"], 0.0)
            self.assertEqual(r["wilcoxon_p"], 1.0)
            self.assertEqual(r["sign_p"], 1.0)
            self.assertEqual(r["holm_p"], 1.0)
            self.assertIsNone(r["d_z"])
            self.assertIsNone(r["rank_biserial"])
            self.assertEqual(r["tied"], 30)
            self.assertIn("wilcoxon: all differences zero", r["notes"])
            self.assertIn("sign test: all differences zero", r["notes"])
            self.assertTrue(r["decision"].startswith("no measurable impact"), r["decision"])
        self.assertIn("no practically relevant effect", _by_metric(res)["template_dice"]["decision"])

    def test_direct_helpers_all_zero(self):
        self.assertEqual(CT.wilcoxon_p(np.zeros(10)), (1.0, CT.ALL_ZERO))
        self.assertEqual(CT.sign_p(np.zeros(10)), (1.0, CT.ALL_ZERO))
        p, note = CT.wilcoxon_p(np.r_[np.zeros(5), np.linspace(0.1, 1.0, 20)])
        self.assertIsNone(note)
        self.assertLess(p, 0.05)

    def test_holm(self):
        got = CT.holm([0.01, 0.04, 0.03, None])
        np.testing.assert_allclose(got[:3], [0.03, 0.06, 0.06])
        self.assertIsNone(got[3])
        np.testing.assert_allclose(CT.holm([0.2, 0.9]), [0.4, 0.9])
        self.assertEqual(CT.holm([0.5, 0.9]), [1.0, 1.0])
        self.assertEqual(CT.holm([0.6]), [0.6])

    def test_mismatched_subjects_and_missing_file(self):
        self.subjects_csv("A", self.subs, self.a)
        self.subjects_csv("B", self.subs[:-1] + ["X99"], self.a)
        rc, _, txt = self.compare()
        self.assertEqual(rc, 2)
        self.assertIn("subject sets differ", txt)
        self.assertIn("X99", txt)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "out", "compare.json")))
        rc, _, txt = self.compare(b="missing")
        self.assertEqual(rc, 2)
        self.assertIn("missing", txt)

    def test_per_label_branch(self):
        rng = np.random.RandomState(1)
        self.subjects_csv("A", self.subs, self.a)
        self.subjects_csv("B", self.subs, self.a)
        rc, res, txt = self.compare()
        self.assertEqual(rc, 0, txt)
        self.assertEqual(res["per_label"], [])
        self.assertIn("per-label data absent", res["notes"])
        labs = ["2", "41", "1002"]
        va = 0.8 + 0.01 * rng.randn(30, 3)
        vb = va.copy()
        vb[:, 0] += 0.02                                       # only label 2 changes
        vb[:, 1] += 0.0005 * np.tile([1.0, -1.0], 15)          # balanced signs: no effect
        for arm, v in (("A", va), ("B", vb)):
            _write(os.path.join(self.tmp, arm, "per_subject_label.csv"),
                   ["subject"] + ["template_dice_" + x for x in labs],
                   [[s] + ["%.9g" % x for x in v[k]] for k, s in enumerate(self.subs)])
        rc, res, txt = self.compare(out="out2")
        self.assertEqual(rc, 0, txt)
        pl = {r["label"]: r for r in res["per_label"]}
        self.assertEqual(sorted(pl), sorted(labs))
        self.assertEqual(pl["2"]["family"], "per_label")
        self.assertLess(pl["2"]["holm_p"], 0.05)
        self.assertGreater(pl["41"]["holm_p"], 0.05)
        self.assertEqual(pl["1002"]["holm_p"], 1.0)
        self.assertNotIn("per-label data absent", res["notes"])
        with open(os.path.join(self.tmp, "out2", "compare.md")) as f:
            self.assertIn("3 labels; Holm p < 0.05: 1 (B better 1, B worse 0)", f.read())


class TestPairStats(Base):
    def test_cluster_bootstrap_recovers_shift(self):
        rng = np.random.RandomState(2)
        n = 30
        D = 0.01 + 0.002 * rng.randn(n, n)
        W = 1.0 - np.eye(n)
        est, lo, hi = CT.cluster_bootstrap_ci(D, W, 4000, 0)
        self.assertAlmostEqual(est, float((D * W).sum() / W.sum()), places=12)
        self.assertLess(abs(est - 0.01), 0.001)
        self.assertLess(lo, 0.01)
        self.assertGreater(hi, 0.01)

    def test_cluster_wider_than_naive_with_subject_effect(self):
        rng = np.random.RandomState(3)
        n = 30
        u = 0.01 * rng.randn(n)
        D = u[:, None] + u[None, :] + 0.0005 * rng.randn(n, n)
        W = 1.0 - np.eye(n)
        _, lo, hi = CT.cluster_bootstrap_ci(D, W, 4000, 0)
        vals = D[W > 0]
        g = np.random.default_rng(0)
        boot = vals[g.integers(0, vals.size, (4000, vals.size))].mean(axis=1)
        nlo, nhi = np.percentile(boot, [2.5, 97.5])
        self.assertGreater(hi - lo, 3.0 * (nhi - nlo))

    def test_pair_metric_and_pair_mismatch(self):
        rng = np.random.RandomState(4)
        subs = ["S%d" % k for k in range(12)]
        a = {"template_dice": 0.7 + 0.01 * rng.randn(12), "fold_fwd_pct": np.zeros(12)}
        self.subjects_csv("A", subs, a)
        self.subjects_csv("B", subs, a)
        pairs = [(i, j) for j in range(12) for i in range(12) if i != j]
        pa = 0.6 + 0.01 * rng.randn(len(pairs))
        pb = pa + 0.005
        for arm, v in (("A", pa), ("B", pb)):
            _write(os.path.join(self.tmp, arm, "per_pair.csv"), ["moving", "fixed", "pair_dice", "pair_fold_pct"],
                   [[subs[i], subs[j], "%.9g" % x, "0"] for (i, j), x in zip(pairs, v)])
        rc, res, txt = self.compare()
        self.assertEqual(rc, 0, txt)
        r = _by_metric(res)["pair_dice"]
        self.assertEqual((r["family"], r["source"], r["n"], r["n_pairs"]), ("primary", "pair", 12, 132))
        self.assertAlmostEqual(r["mean_diff"], 0.005, places=6)
        self.assertAlmostEqual(r["pair_mean_diff"], 0.005, places=6)
        self.assertEqual(r["ci_method"], "subject-cluster bootstrap")
        self.assertAlmostEqual(r["ci_low"], 0.005, places=6)
        self.assertIn("measurable impact (B better); practically relevant", r["decision"])
        self.assertEqual(len([x for x in res["primary"] if not x.get("missing")]), 3)
        self.assertEqual(_by_metric(res)["pair_fold_pct"]["family"], "secondary")
        # a pair missing in B -> exit 2
        _write(os.path.join(self.tmp, "B", "per_pair.csv"), ["moving", "fixed", "pair_dice"],
               [[subs[i], subs[j], "0.6"] for (i, j) in pairs[1:]])
        rc, _, txt = self.compare(out="out2")
        self.assertEqual(rc, 2)
        self.assertIn("pair sets differ", txt)


class TestLinearKind(Base):
    def test_linear_eval_layout(self):
        rng = np.random.RandomState(5)
        subs = ["S%d" % k for k in range(10)]
        cols = ["center_rms_mm", "center_rot_deg", "center_trans_mm", "pair_rms_mm_mean"]
        va = 2.0 + 0.3 * rng.rand(10, 4)
        vb = va.copy()
        vb[:, 3] -= 0.6 + 0.01 * rng.rand(10)
        pairs = [(i, j) for j in range(10) for i in range(10) if i != j]
        qa = 3.0 + 0.3 * rng.rand(len(pairs), 2)
        qb = qa - np.array([0.6, 0.0])
        for arm, v, q in (("A", va, qa), ("B", vb, qb)):
            _write(os.path.join(self.tmp, arm, "per_subject.csv"), ["subject", "subject_dir"] + cols,
                   [[s, s + "_T1w"] + ["%.9g" % x for x in v[k]] for k, s in enumerate(subs)])
            _write(os.path.join(self.tmp, arm, "per_pair.csv"), ["moving", "fixed", "pair_rms_mm", "pair_rot_deg"],
                   [[subs[i], subs[j]] + ["%.9g" % x for x in q[p]] for p, (i, j) in enumerate(pairs)])
            with open(os.path.join(self.tmp, arm, "summary.json"), "w") as f:
                json.dump({"note": "x"}, f)
        rc, res, txt = self.compare("--kind", "linear", "--label-a", "none", "--label-b", "flip8")
        self.assertEqual(rc, 0, txt)
        self.assertEqual(res["primary"], [])
        m = _by_metric(res)
        self.assertEqual(sorted(m), sorted(cols + ["pair_rms_mm", "pair_rot_deg"]))
        self.assertNotIn("subject_dir", m)
        for r in m.values():
            self.assertEqual(r["family"], "linear")
        ep = m["pair_rms_mm_mean"]
        self.assertIn("linear endpoint", ep["notes"])
        self.assertEqual(ep["sesoi"], 0.5)
        self.assertEqual(m["pair_rms_mm"]["sesoi"], 0.5)
        self.assertIsNone(m["center_rms_mm"]["sesoi"])
        self.assertIn("measurable impact (B better); practically relevant", ep["decision"])
        self.assertEqual(m["pair_rms_mm"]["ci_method"], "subject-cluster bootstrap")
        self.assertEqual(m["center_rot_deg"]["wilcoxon_p"], 1.0)
        with open(os.path.join(self.tmp, "out", "compare.md")) as f:
            md = f.read()
        self.assertIn(CT.SCOPE, md)
        self.assertIn("flip8 (B) vs none (A)", md)


class TestEndToEndEvaluation(Base):
    """Two `evaluation run` outputs (--pairs all) on the tiny grid of tests/test_evaluation.py."""

    def test_two_evaluation_runs(self):
        atlas, alab = TE._atlas()
        proc = os.path.join(self.tmp, "processed")
        for arm, extra in (("fa", 0.0), ("fb", 0.4)):
            for i, s in enumerate(TE.SHIFTS):
                name = "SUBJ-%d" % i
                inv = np.broadcast_to(-s, TE.SHAPE + (3,)).astype(np.float32)
                if arm == "fa":
                    d = os.path.join(proc, name + "_T1w")
                    TE._save(os.path.join(d, "image.nii.gz"), TE.F.warp(atlas, inv, 1))
                    TE._save(os.path.join(d, "labels_dkt31aseg.nii.gz"), TE.F.warp(alab, inv, 0), np.uint16)
                    with open(os.path.join(d, "meta.json"), "w") as f:
                        json.dump({"row": {"subject": name}}, f)
                fwd = np.broadcast_to(s + np.float32(extra * (i % 2)), TE.SHAPE + (3,)).astype(np.float32)
                TE._save(os.path.join(self.tmp, arm, name + "_fwd.nii.gz"), fwd)
                TE._save(os.path.join(self.tmp, arm, name + "_inv.nii.gz"), -fwd)
        for arm in ("fa", "fb"):
            with contextlib.redirect_stdout(io.StringIO()):
                rc = TE.main(["run", "--processed-dir", proc, "--method", arm,
                              "--fwd", os.path.join(self.tmp, arm, "{subject}_fwd.nii.gz"),
                              "--inv", os.path.join(self.tmp, arm, "{subject}_inv.nii.gz"),
                              "--pairs", "all", "--out", os.path.join(self.tmp, "ev_" + arm)])
            self.assertEqual(rc, 0)
        rc, res, txt = self.compare(a="ev_fa", b="ev_fb")
        self.assertEqual(rc, 0, txt)
        self.assertEqual((res["n_subjects"], res["n_pairs"]), (4, 12))
        m = _by_metric(res)
        self.assertEqual([r["metric"] for r in res["primary"]], list(CT.PRIMARY))
        self.assertFalse(any(r.get("missing") for r in res["primary"]))
        self.assertEqual(m["pair_dice"]["n_pairs"], 12)
        self.assertEqual(len(res["per_label"]), 4)
        self.assertEqual(sorted(res["group"]), sorted(CT.GROUP_KEYS))
        self.assertIsNotNone(res["group"]["centrality_mean_mm"]["b"])
        self.assertLessEqual(m["template_dice"]["mean_diff"], 0.0)
        with open(os.path.join(self.tmp, "out", "compare.md")) as f:
            md = f.read()
        self.assertIn(CT.SCOPE, md)
        self.assertIn("## Primary endpoints", md)
        self.assertIn("## Group metrics", md)


if __name__ == "__main__":
    unittest.main()
