"""evaluation/ (method-independent groupwise metrics): numpy/scipy only, no TF.

Run alone: python -m unittest tests.test_evaluation -v
"""
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

import nibabel as nib
import numpy as np

from evaluation import fields as F
from evaluation import metrics as M
from evaluation.evaluate import main

SHAPE = (24, 28, 22)
AFF = np.array([[1.0, 0, 0, -12], [0, 1, 0, -14], [0, 0, 1, -11], [0, 0, 0, 1]])
# Subject i = atlas shifted by SHIFTS[i] voxels (sum 0 -> centred atlas); fwd_i = +s_i, inv_i = -s_i.
SHIFTS = np.array([[2, 1, -1], [-2, -1, 1], [1, -2, 2], [-1, 2, -2]], np.float32)


def _atlas():
    x, y, z = np.indices(SHAPE).astype(np.float32)
    cx, cy, cz = (np.asarray(SHAPE) - 1) / 2.0
    r = np.sqrt(((x - cx) / 7.0) ** 2 + ((y - cy) / 8.0) ** 2 + ((z - cz) / 6.0) ** 2)
    brain = r < 1.0
    img = (0.4 + 0.3 * (x - cx) / 7 + 0.2 * (y - cy) / 8 - 0.1 * (z - cz) / 6 + 0.3 * np.cos(r * 4)) * brain
    img = np.clip(img, 0.05, None) * brain
    lab = np.zeros(SHAPE, np.uint16)
    lab[brain & (x < cx)] = 2
    lab[brain & (x >= cx)] = 41
    lab[brain & (r > 0.7) & (y < cy)] = 1002
    lab[brain & (r > 0.7) & (y >= cy) & (z < cz)] = 2035
    return img.astype(np.float32), lab


def _save(path, data, dtype=np.float32):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data).astype(dtype), AFF), path)
    return path


def _encode(canon_vox, components="zyx", signs="+-+", spacing=(1.0, 1.0, 1.0)):
    """Canonical voxel field -> a method's file layout (inverse of Convention.to_canonical, mm units)."""
    raw = np.empty_like(canon_vox)
    for c, ax in enumerate(components):
        a = "xyz".index(ax)
        raw[..., c] = canon_vox[..., a] * spacing[a] * (-1 if signs[a] == "-" else 1)
    return raw


def _smooth_field(amp=2.5, seed=0):
    rng = np.random.RandomState(seed)
    x, y, z = np.indices(SHAPE).astype(np.float32)
    u = np.zeros(SHAPE + (3,), np.float32)
    for c in range(3):
        p = rng.uniform(0, 2 * np.pi, 3)
        u[..., c] = amp * np.sin(2 * np.pi * x / 21 + p[0]) * np.cos(2 * np.pi * y / 25 + p[1]) \
            * np.sin(2 * np.pi * z / 19 + p[2])
    return u


def _numeric_inverse(fwd, iters=60):
    inv = -fwd.copy()
    for _ in range(iters):
        inv = inv - F.compose(inv, fwd)
    return inv


def _run_quiet(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


class FieldOps(unittest.TestCase):
    def test_convention_round_trip(self):
        u = _smooth_field()
        for comp, signs, units in (("xyz", "+++", "mm"), ("zyx", "+-+", "mm"), ("yzx", "--+", "vox")):
            sp = (1.0, 1.0, 1.0) if units == "vox" else (1.0, 1.5, 0.8)
            raw = _encode(u, comp, signs, sp if units == "mm" else (1.0, 1.0, 1.0))
            got = F.Convention(comp, signs, units).to_canonical(raw, sp)
            np.testing.assert_allclose(got, u, atol=1e-5)

    def test_to_xyz3_layouts(self):
        u = _smooth_field()
        for a in (u, u[:, :, :, None, :], np.moveaxis(u, -1, 0), u[None]):
            np.testing.assert_array_equal(F.to_xyz3(a, SHAPE), u)
        with self.assertRaises(ValueError):
            F.to_xyz3(np.zeros((3, 3, 3, 3)), SHAPE)

    def test_warp_shift_and_labels(self):
        img, lab = _atlas()
        d = np.zeros(SHAPE + (3,), np.float32)
        d[..., 0] = 2
        w = F.warp(img, d, 1)
        np.testing.assert_allclose(w[:-2], img[2:], atol=1e-6)    # out(x) = img(x + 2)
        wl = F.warp(lab, d, 0)
        self.assertEqual(wl.dtype, lab.dtype)
        np.testing.assert_array_equal(wl[:-2], lab[2:])

    def test_compose_and_inverse(self):
        fwd = _smooth_field(1.2)                              # max |grad u| ~0.4: invertible, no folds
        self.assertGreater(float(F.jacobian_det(fwd).min()), 0.0)
        inv = _numeric_inverse(fwd)                           # solves compose(inv, fwd) = 0
        e_solved = F.norm_mm(F.compose(inv, fwd), (1, 1, 1))[3:-3, 3:-3, 3:-3]
        self.assertLess(float(e_solved.max()), 1e-3)
        # other order: only trilinear interpolation error of the sampled fields remains
        e = F.norm_mm(F.compose(fwd, inv), (1, 1, 1))[3:-3, 3:-3, 3:-3]
        self.assertLess(float(e.mean()), 0.03)
        self.assertLess(float(e.max()), 0.15)
        a = np.zeros(SHAPE + (3,), np.float32)
        a[..., 1] = 1.5
        np.testing.assert_allclose(F.compose(a, a)[..., 1], 3.0, atol=1e-6)

    def test_jacobian(self):
        x = np.indices(SHAPE).astype(np.float32)
        u = np.zeros(SHAPE + (3,), np.float32)
        u[..., 0] = 0.1 * x[0]
        u[..., 1] = -0.2 * x[1]
        np.testing.assert_allclose(F.jacobian_det(u), 1.1 * 0.8, atol=1e-5)
        u[..., 0] = -2.0 * x[0]                                   # folds everywhere
        self.assertEqual(M.folding_pct(F.jacobian_det(u), np.ones(SHAPE, bool)), 100.0)


class MetricFns(unittest.TestCase):
    def test_dice_and_split(self):
        a = np.array([0, 2, 2, 1002, 1002, 0])
        b = np.array([0, 2, 0, 1002, 41, 0])
        d = M.dice_per_label(a, b, [2, 41, 1002, 17])
        np.testing.assert_allclose(d[:3], [2 / 3, 0.0, 2 / 3])
        self.assertTrue(np.isnan(d[3]))
        s = M.dice_summary(d, [2, 41, 1002, 17], "dice")
        self.assertAlmostEqual(s["dice_cortical"], 2 / 3)
        self.assertAlmostEqual(s["dice_subcortical"], 1 / 3)

    def test_ncc_lncc_hd95(self):
        img, lab = _atlas()
        m = img > 0
        self.assertAlmostEqual(M.ncc(img, 2 * img + 1, m), 1.0, places=6)
        self.assertGreater(M.lncc(img, img, m), 0.9)
        self.assertLess(M.lncc(img, np.random.RandomState(0).rand(*SHAPE), m), 0.2)
        h = M.hd95_per_label(lab, lab, [2, 41], (1, 1, 1))
        np.testing.assert_allclose(h, 0.0)
        shifted = np.roll(lab, 2, axis=0)
        self.assertAlmostEqual(float(M.hd95_per_label(lab, shifted, [41], (1, 1, 1))[0]), 2.0, places=5)

    def test_majority_vote_loo(self):
        w = np.array([[[[1, 2]]], [[[1, 2]]], [[[3, 2]]]], np.uint16)  # [3, 1, 1, 2]
        mv = M.MajorityVote(w)
        np.testing.assert_array_equal(mv.full()[0, 0], [1, 2])
        np.testing.assert_array_equal(mv.loo(0)[0, 0], [1, 2])         # tie 1 vs 3 -> smaller label
        np.testing.assert_array_equal(mv.loo(2)[0, 0], [1, 2])
        np.testing.assert_array_equal(mv.full()[0, 0], [1, 2])         # counts restored


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgr_eval_")
        cls.proc = os.path.join(cls.tmp, "processed", "mindboggle")
        cls.fdir = os.path.join(cls.tmp, "baseline")
        atlas, alab = _atlas()
        cls.atlas, cls.alab = atlas, alab
        for i, s in enumerate(SHIFTS):
            name = "SUBJ-%d" % i
            inv = np.broadcast_to(-s, SHAPE + (3,)).astype(np.float32)
            fwd = np.broadcast_to(s, SHAPE + (3,)).astype(np.float32)
            img = F.warp(atlas, inv, 1)                           # I_i(y) = A(y - s_i)
            lab = F.warp(alab, inv, 0)
            d = os.path.join(cls.proc, name + "_T1w")
            _save(os.path.join(d, "image.nii.gz"), img)
            _save(os.path.join(d, "labels_dkt31aseg.nii.gz"), lab, np.uint16)
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump({"row": {"subject": name}}, f)
            _save(os.path.join(cls.fdir, name + "_fwd.nii.gz"), _encode(fwd))
            _save(os.path.join(cls.fdir, name + "_inv.nii.gz"), _encode(inv))
        _save(os.path.join(cls.fdir, "atlas.nii.gz"), atlas)
        _save(os.path.join(cls.fdir, "atlas_seg.nii.gz"), alab, np.uint16)
        cls.flags = ["--processed-dir", cls.proc, "--fwd", os.path.join(cls.fdir, "{subject}_fwd.nii.gz"),
                     "--inv", os.path.join(cls.fdir, "{subject}_inv.nii.gz")]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, out, *extra):
        rc, text = _run_quiet(["run", "--processed-dir", self.proc, "--out", os.path.join(self.tmp, out)]
                              + list(extra))
        self.assertEqual(rc, 0)
        with open(os.path.join(self.tmp, out, "summary.json")) as f:
            return json.load(f), text

    def test_method_exact_fields(self):
        s, _ = self._run("res_baseline", "--method", "baseline", "--fwd", self.flags[3], "--inv", self.flags[5],
                         "--components", "zyx", "--signs", "+-+", "--atlas",
                         os.path.join(self.fdir, "atlas.nii.gz"), "--atlas-seg",
                         os.path.join(self.fdir, "atlas_seg.nii.gz"), "--workers", "2", "--hd95")
        sm, pm, g = s["subject_metrics"], s["pair_metrics"], s["group"]
        self.assertEqual((s["n_subjects"], s["n_pairs"], s["n_labels"]), (4, 12, 4))
        for k in ("template_dice", "template_dice_loo", "transfer_dice", "template_dice_cortical",
                  "template_dice_subcortical"):
            self.assertGreater(sm[k]["min"], 0.999, k)
        self.assertGreater(pm["pair_dice"]["min"], 0.999)
        self.assertGreater(sm["template_ncc"]["min"], 0.999)
        self.assertGreater(pm["pair_ncc"]["min"], 0.999)
        self.assertAlmostEqual(sm["template_hd95"]["max"], 0.0, places=5)
        for k in ("fold_fwd_pct", "fold_inv_pct", "inv_err_fwd_mm", "inv_err_inv_mm", "logjac_std_fwd"):
            self.assertLess(sm[k]["max"], 1e-5, k)
        self.assertLess(pm["pair_fold_pct"]["max"], 1e-9)
        self.assertLess(g["centrality_max_mm"], 1e-5)
        self.assertLess(g["mean_logjac_abs"], 1e-5)
        self.assertLess(g["atlas_intensity_std"], 1e-5)
        self.assertGreater(sm["disp_fwd_mean_mm"]["min"], 2.4)
        with open(os.path.join(self.tmp, "res_baseline", "per_pair.csv")) as f:
            self.assertEqual(len(f.read().strip().splitlines()), 13)
        with open(os.path.join(self.tmp, "res_baseline", "labels.txt")) as f:
            self.assertEqual(f.read().split(), ["2", "41", "1002", "2035"])

    def test_identity_reference_and_wrong_convention(self):
        ident, _ = self._run("res_identity", "--method", "identity", "--identity", "--pairs", "5")
        self.assertEqual(ident["n_pairs"], 5)
        self.assertLess(ident["subject_metrics"]["template_dice"]["mean"], 0.95)
        self.assertEqual(ident["subject_metrics"]["fold_fwd_pct"]["max"], 0.0)
        self.assertEqual(ident["group"]["centrality_max_mm"], 0.0)
        wrong, _ = self._run("res_wrong", "--method", "wrong", "--fwd", self.flags[3], "--inv", self.flags[5])
        self.assertLess(wrong["subject_metrics"]["template_dice"]["mean"], 0.95)   # xyz/+++ is not the truth

    def test_majority_vote_atlas_when_none_given(self):
        s, _ = self._run("res_mv", "--method", "mv", "--fwd", self.flags[3], "--inv", self.flags[5],
                         "--components", "zyx", "--signs", "+-+", "--pairs", "none", "--no-preload-fwd")
        self.assertEqual(s["atlas_seg"], "majority vote of the warped labels")
        self.assertGreater(s["subject_metrics"]["template_dice"]["min"], 0.999)
        self.assertEqual(s["n_pairs"], 0)

    def test_check_convention_finds_layout_and_roles(self):
        """Nonlinear fields (translations cannot tell swapped roles from negated signs)."""
        d = os.path.join(self.tmp, "conv")
        for i in range(4):
            name = "SUBJ-%d" % i
            img = nib.load(os.path.join(self.proc, name + "_T1w", "image.nii.gz")).get_fdata().astype(np.float32)
            fwd = _smooth_field(1.5, seed=i)
            inv = _numeric_inverse(fwd)
            _save(os.path.join(d, name + "_moved.nii.gz"), F.warp(img, fwd, 1))
            # stored swapped: the method's "fwd" file holds our inverse; components yxz, signs -++
            _save(os.path.join(d, name + "_fwd.nii.gz"), _encode(inv, "yxz", "-++"))
            _save(os.path.join(d, name + "_inv.nii.gz"), _encode(fwd, "yxz", "-++"))
        rc, text = _run_quiet(["check-convention", "--processed-dir", self.proc,
                               "--fwd", os.path.join(d, "{subject}_fwd.nii.gz"),
                               "--inv", os.path.join(d, "{subject}_inv.nii.gz"),
                               "--moved", os.path.join(d, "{subject}_moved.nii.gz"), "--n-subjects", "3",
                               "--out", os.path.join(d, "conv.json")])
        self.assertEqual(rc, 0, text)
        self.assertIn("RESULT: use  --components yxz --signs -++ --units mm --swap-roles", text)


if __name__ == "__main__":
    unittest.main()
