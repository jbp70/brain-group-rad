"""Validation PNGs (runtime.val_png and the Trainer hook): panel arrays, figure layout, file
names and sizes, the identity / perfect-model rows, and no output when the PNGs are off.
Milestone 6: the deformable figure (identity rows, label colours, padded Jacobian, file size)."""
import os
import shutil
import struct
import tempfile
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.runtime_helpers import free_memory, narrow_config, synthetic_group
import numpy as np

from data_loading import params as P
from models.group_net import GroupRegLevelModel
from runtime import fixed_data as fd
from runtime import optimizers as O
from runtime import val_png as V
from runtime.run import Trainer, TrainConfig
from runtime.schedulers import ScheduleConfig

N = 4
SPACING = P.LEVEL_SPACING[4]


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", head[16:24])


class TestPanels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        g = synthetic_group(4, N, "rigid", seed=3)
        cls.vs = fd.ValidationSet.from_group(g["images"], g["T"], g["T_target"], "rigid", 4)

    def tearDown(self):
        free_memory()

    def test_select_indices_fixed(self):
        a = V.select_indices(39, 8, seed=0)
        self.assertEqual(len(a), 8)
        self.assertEqual(len(set(a.tolist())), 8)
        np.testing.assert_array_equal(a, V.select_indices(39, 8, seed=0))
        np.testing.assert_array_equal(V.select_indices(4, 8), np.arange(4))   # K = N if smaller

    def test_axial_display_convention(self):
        v = np.zeros((5, 7, 6), np.float32)
        v[4, 6, 3] = 1.0          # max X (right), max Y (anterior), middle Z
        d = V.axial_display(v)
        self.assertEqual(d.shape, (7, 5))
        self.assertEqual(d[0, 0], 1.0)              # anterior at the top, R on the left

    def test_identity_and_perfect_rows(self):
        grp = self.vs.groups[0]
        imgs, a_star = grp["images"], grp["A_star"]
        eye = np.tile(np.eye(4), (N, 1, 1))
        p = V.build_panels(imgs, eye, a_star, SPACING)
        self.assertEqual((len(p), len(p[0])), (3, N + 1))
        for r1, r2 in zip(p[0], p[1]):                              # identity model: row 1 == row 2
            np.testing.assert_allclose(r1, r2, atol=1e-6)
        self.assertGreater(np.abs(p[0][0] - p[0][1]).max(), 0.05)  # inputs differ
        p = V.build_panels(imgs, a_star, a_star, SPACING, mean_col=False)
        self.assertEqual(len(p[0]), N)
        for r2, r3 in zip(p[1], p[2]):                              # perfect model: row 2 == row 3
            np.testing.assert_array_equal(r2, r3)
        np.testing.assert_allclose(V.build_panels(imgs, a_star, a_star, SPACING)[2][-1],
                                   np.mean(np.stack(p[2]), axis=0), atol=1e-6)   # mean column
        p = V.build_panels(imgs, eye, a_star, SPACING, gt_row=False)
        self.assertEqual(len(p), 2)

    def test_figure_layout(self):
        grp = self.vs.groups[0]
        for gt, mean in ((True, True), (False, False)):
            p = V.build_panels(grp["images"], grp["A_star"], grp["A_star"], SPACING, gt, mean)
            fig = V.make_figure(p, "t", np.zeros(N), mean_col=mean)
            rows, cols = (3 if gt else 2), N + (1 if mean else 0)
            self.assertEqual(len(fig.axes), rows * cols)
            self.assertEqual(fig.axes[1 * cols].get_xlabel(), "0.00 mm")      # row-2 error label


class TestDeformPanels(unittest.TestCase):
    """Milestone 6: the deformable figure (4 rows: input, warped, labels, det J)."""

    @classmethod
    def setUpClass(cls):
        g = fd.synthetic_deform_group(4, N, seed=0)
        cls.images = g["images"]
        cls.seg = g["labels"][..., 0].astype(np.int64)
        shape = P.LEVEL_SHAPE[4]
        cls.jac = np.ones((N,) + tuple(s - 2 for s in shape), np.float32)

    def tearDown(self):
        free_memory()

    def test_identity_panels(self):
        template = self.seg[0]
        p = V.build_deform_panels(self.images, self.images, self.seg, template, self.jac)
        self.assertEqual((len(p), len(p[0])), (4, N + 1))
        Y, X = P.LEVEL_SHAPE[4][1], P.LEVEL_SHAPE[4][0]
        self.assertEqual(p[0][0].shape, (Y, X))
        self.assertEqual(p[2][0].shape, (Y, X, 3))                  # RGB label panels
        for r1, r2 in zip(p[0], p[1]):                              # identity: warped row == input row
            np.testing.assert_array_equal(r1, r2)
        np.testing.assert_allclose(p[0][-1], np.mean(np.stack(p[0][:N]), axis=0), atol=1e-6)
        np.testing.assert_array_equal(p[2][-1], V.label_display(template))   # template column
        np.testing.assert_allclose(p[3][-1], 1.0)                   # mean J of identity maps
        bg = V.axial_display(self.seg[0]) == 0
        self.assertTrue(bg.any())
        np.testing.assert_array_equal(p[2][0][bg], 0.0)             # background is black
        self.assertGreater(len({tuple(c) for c in p[2][0].reshape(-1, 3)}), 3)
        p = V.build_deform_panels(self.images, self.images, self.seg, template, self.jac, mean_col=False)
        self.assertEqual(len(p[0]), N)

    def test_jacobian_padding_and_file(self):
        jac = self.jac.copy()
        jac[0, 5, 5, P.LEVEL_SHAPE[4][2] // 2 - 1] = -1.0           # interior index -> full index + 1
        full = V.pad_jacobian(jac)
        self.assertEqual(full.shape, (N,) + tuple(P.LEVEL_SHAPE[4]))
        self.assertEqual(full[0, 6, 6, P.LEVEL_SHAPE[4][2] // 2], -1.0)
        self.assertEqual(full[0, 0, 0, 0], 1.0)
        tmp = tempfile.mkdtemp(prefix="bgr_m6_png_")
        try:
            path = os.path.join(tmp, "deform_step0000007.png")
            title = V.deform_figure_title(7, 0.8, 0.7, 0.01, P.LEVEL_SHAPE[4][2] // 2)
            self.assertIn("deform step 7", title)
            panels, fig = V.write_deform_png(path, self.images, self.images, self.seg, self.seg[0], jac, title,
                                             np.zeros(N), labels=["a", "b", "c", "d"])
            self.assertEqual(png_size(path), V.figure_size_px(4, N + 1))
            self.assertEqual(len(fig.axes), 4 * (N + 1))
            self.assertEqual(fig.axes[3 * (N + 1)].get_xlabel(), "fold 0.000 %")
            self.assertEqual(fig.axes[0].get_title(), "a")
            path2 = os.path.join(tmp, "small.png")
            V.write_deform_png(path2, self.images[:2], self.images[:2], self.seg[:2], self.seg[0], jac[:2], title,
                               np.zeros(2), mean_col=False)
            self.assertEqual(png_size(path2), V.figure_size_px(4, 2))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestTrainerPng(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_m4_png_")
        g = synthetic_group(4, N, "rigid", seed=0)
        self.g = g
        self.vs = fd.ValidationSet.from_group(g["images"], g["T"], g["T_target"], "rigid", 4)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        free_memory()

    def make(self, **kw):
        d = dict(level=4, group_size=N, steps=100, val_every=2, log_every=1, task_ratio={"rigid": 1.0},
                 schedule=ScheduleConfig("constant", base_lr=1e-4, total_steps=100, warmup_steps=0))
        d.update(kw)
        model = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1)).build_model(N)
        return Trainer(model, O.make_optimizer("adam", lr=1e-4),
                       {"rigid": fd.FixedGroupSource(self.g["batch"])}, self.g["S"], TrainConfig(**d),
                       val_sets={"rigid": self.vs})

    def test_written_at_validation_and_final_step(self):
        out = os.path.join(self.tmp, "val_png")
        tr = self.make(val_png_dir=out, val_png_k=8)
        tr.fit(3)
        self.assertEqual(sorted(os.listdir(out)), ["rigid_step0000002.png", "rigid_step0000003.png"])
        self.assertEqual([v["step"] for v in tr.val_history], [2, 3])   # final-step validation
        self.assertEqual(png_size(os.path.join(out, "rigid_step0000003.png")), V.figure_size_px(3, N + 1))
        tr2 = self.make(val_png_dir=os.path.join(self.tmp, "small"), val_png_k=2,
                        val_png_gt_row=False, val_png_mean_col=False)
        tr2.validate()
        self.assertEqual(png_size(os.path.join(self.tmp, "small", "rigid_step0000000.png")),
                         V.figure_size_px(2, 2))

    def test_off_writes_nothing_and_failure_does_not_raise(self):
        tr = self.make()                       # val_png_dir None (--val-png false)
        tr.fit(3)
        self.assertEqual(os.listdir(self.tmp), [])
        self.assertEqual([v["step"] for v in tr.val_history], [2])   # no extra final validation
        bad = self.make(val_png_dir=os.path.join(self.tmp, "x"), val_png_group=0)
        self.vs.groups[0] = dict(self.vs.groups[0], index=None)       # breaks the label step
        m = bad.validate()                                            # warns, no exception
        self.assertIn("val_rigid/center_rms_mm", m)
        self.assertEqual(bad.val_png_paths, [])


if __name__ == "__main__":
    unittest.main()
