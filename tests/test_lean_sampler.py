"""Lean trilinear sampler (`models.layers.sample_voxel_coords(lean=True)`, milestone 9 design 3.3).

Forward: max |diff| 0 against the default sampler, compared EAGERLY (no tf.function, so no
Grappler rewrite can move values), and once more inside tf.function with Grappler off
(`tests.test_cascade._GrapplerOff`) and op determinism on (`tests.test_trainer._OpDeterminism`),
where it must also be 0. Gradients w.r.t. coords and the volume within 1e-4 relative (L2 norm of
the difference over the norm; float64 1e-12). ``lean_images_grad=False``: no images gradient and
the same coords gradient. `exp_and_inverse(lean=True)` against the default.

`test_graph_mode_memory` (local GPU, Grappler at its defaults, level-2-size field 80x112x96,
N = 2, float32): peak minus inputs of a second call, against the per-voxel model of design 2.1
computed here from the same formulas. Case A: one-hot warp (C = 19) gradient w.r.t. u, lean
<= 1.5 x its forward moment. Case B: the integrator block under tf.recompute_grad (T = 7), lean
<= 1.5 x model and at least 0.85 GiB below the default sampler. Case C: the lean backward alone
(`_lean_linear_backward`), C1 one-hot zero mode without the images gradient, C2 field border mode
with it: the corner-sequencing check. Case B without the dy dependency (`layers._DY_DEP = False`)
is printed only. The numbers are printed for the report.
"""
import sys
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.field_helpers import smooth_field
from tests.test_cascade import _GrapplerOff
from tests.test_trainer import _OpDeterminism
import numpy as np
import tensorflow as tf

from models import diffeo as D
from models import layers as L

GIB = float(2 ** 30)


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def dense(g):
    return None if g is None else tf.convert_to_tensor(g).numpy()


def make_case(rng, c, dtype, shape=(9, 10, 8), n=2, spread=3.0):
    """Random volume [n, *shape, c] and sampling coords (index grid + smooth field + noise; some
    positions fall outside the volume)."""
    img = rng.uniform(-1.0, 1.0, size=(n,) + shape + (c,))
    u = smooth_field(shape, rng, n=n, sigma=1.5, max_vox=spread) + rng.normal(0, 0.2, (n,) + shape + (3,))
    grid = np.stack(np.meshgrid(*[np.arange(s, dtype=np.float64) for s in shape], indexing="ij"), -1)
    return tf.constant(img.astype(dtype)), tf.constant((grid[None] + u).astype(dtype))


COMBOS = [(b, f) for b in ("zero", "border") for f in (0.0, 0.3)]


class TestLeanForward(unittest.TestCase):
    def test_eager_bit_identical(self):
        rng = np.random.default_rng(0)
        for dtype in (np.float32, np.float64):
            for c in (1, 3, 19):
                img, crd = make_case(rng, c, dtype)
                for boundary, fill in COMBOS:
                    a = L.sample_voxel_coords(img, crd, fill_value=fill, boundary=boundary).numpy()
                    b = L.sample_voxel_coords(img, crd, fill_value=fill, boundary=boundary, lean=True).numpy()
                    self.assertEqual(a.dtype, b.dtype)
                    self.assertTrue(np.array_equal(a, b), (dtype, c, boundary, fill, np.abs(a - b).max()))

    def test_graph_bit_identical(self):
        rng = np.random.default_rng(1)
        with _GrapplerOff(), _OpDeterminism():
            for c in (1, 19):
                img, crd = make_case(rng, c, np.float32)
                for boundary, fill in COMBOS:
                    f_def = tf.function(lambda: L.sample_voxel_coords(img, crd, fill_value=fill, boundary=boundary))
                    f_lean = tf.function(lambda: L.sample_voxel_coords(img, crd, fill_value=fill,
                                                                       boundary=boundary, lean=True))
                    a, b = f_def().numpy(), f_lean().numpy()
                    self.assertTrue(np.array_equal(a, b), (c, boundary, fill, np.abs(a - b).max()))

    def test_nearest_ignores_lean(self):
        rng = np.random.default_rng(2)
        img, crd = make_case(rng, 3, np.float32)
        a = L.sample_voxel_coords(img, crd, interpolation="nearest").numpy()
        b = L.sample_voxel_coords(img, crd, interpolation="nearest", lean=True).numpy()
        self.assertTrue(np.array_equal(a, b))

    def test_float16_volume(self):
        rng = np.random.default_rng(3)
        img, crd = make_case(rng, 3, np.float32)
        img16 = tf.cast(img, tf.float16)
        a = L.sample_voxel_coords(img16, crd)
        b = L.sample_voxel_coords(img16, crd, lean=True)
        self.assertEqual((a.dtype, b.dtype), (tf.float16, tf.float16))
        self.assertTrue(np.array_equal(a.numpy(), b.numpy()))


class TestLeanGradients(unittest.TestCase):
    def grads(self, img, crd, w, lean, images_grad=True, **kw):
        with tf.GradientTape() as tape:
            tape.watch([img, crd])
            out = L.sample_voxel_coords(img, crd, lean=lean, lean_images_grad=images_grad, **kw)
            loss = tf.reduce_sum(tf.cast(out, w.dtype) * w)
        gi, gc = tape.gradient(loss, [img, crd])
        return dense(gi), dense(gc)

    def test_gradients_match(self):
        rng = np.random.default_rng(4)
        worst = {}
        for dtype, tol in ((np.float32, 1e-4), (np.float64, 1e-12)):
            for c in (1, 3, 19):
                img, crd = make_case(rng, c, dtype)
                w = tf.constant(rng.normal(size=img.shape[:4].as_list() + [c]).astype(dtype))
                for boundary, fill in COMBOS:
                    di, dc = self.grads(img, crd, w, False, boundary=boundary, fill_value=fill)
                    li, lc = self.grads(img, crd, w, True, boundary=boundary, fill_value=fill)
                    for name, a, b in (("images", li, di), ("coords", lc, dc)):
                        e = rel(a, b)
                        key = (np.dtype(dtype).name, name)
                        worst[key] = max(worst.get(key, 0.0), e)
                        self.assertLessEqual(e, tol, (dtype, c, boundary, fill, name, e))
        print("\n[lean grads] worst rel. error: %s" % ", ".join(
            "%s/%s %.2e" % (k[0], k[1], v) for k, v in sorted(worst.items())), file=sys.stderr)

    def test_images_grad_false(self):
        rng = np.random.default_rng(5)
        for boundary, fill in COMBOS:
            img, crd = make_case(rng, 19, np.float32)
            w = tf.constant(rng.normal(size=img.shape.as_list()).astype(np.float32))
            ti, tc = self.grads(img, crd, w, True, True, boundary=boundary, fill_value=fill)
            fi, fc = self.grads(img, crd, w, True, False, boundary=boundary, fill_value=fill)
            self.assertIsNotNone(ti)
            self.assertIsNone(fi)
            self.assertTrue(np.array_equal(tc, fc), (boundary, fill))

    def test_exp_and_inverse(self):
        rng = np.random.default_rng(6)
        shape = (16, 18, 14)
        v = tf.constant(smooth_field(shape, rng, n=2, sigma=2.0, max_vox=3.0).astype(np.float32))
        w = tf.constant(rng.normal(size=(2,) + shape + (3,)).astype(np.float32))
        res = {}
        for lean in (False, True):
            with tf.GradientTape() as tape:
                tape.watch(v)
                u, ui = D.exp_and_inverse(v, 7, lean=lean)
                loss = tf.reduce_sum(u * w) + tf.reduce_sum(ui * w * 0.5)
            res[lean] = (u.numpy(), ui.numpy(), tape.gradient(loss, v).numpy())
        self.assertTrue(np.array_equal(res[False][0], res[True][0]))
        self.assertTrue(np.array_equal(res[False][1], res[True][1]))
        e = rel(res[True][2], res[False][2])
        print("\n[lean exp_and_inverse] grad rel. error %.2e" % e, file=sys.stderr)
        self.assertLessEqual(e, 1e-4)


# ============================================================ graph-mode memory (design 3.3)
SHAPE2 = (80, 112, 96)
N_MEM = 2
B_PER_GIB = N_MEM * int(np.prod(SHAPE2)) / GIB      # 1 B per voxel per image, in GiB


def t_fwd(c):
    return 84 + 32 * c                               # float tensors of one lean forward call


def fwd_moment(c):
    return t_fwd(c) + 168 + 4 * c                    # + int64 indices + output / running sum


def t_bwd(c, images_grad):
    return 88 + 8 * c + (16 * c if images_grad else 0)


def peak_gib(fn):
    """Peak minus inputs of the second call of `fn` (a tf.function returning a scalar), GiB."""
    float(fn())                                      # trace and warm-up
    tf.config.experimental.reset_memory_stats("GPU:0")
    base = tf.config.experimental.get_memory_info("GPU:0")["current"]
    float(fn())
    return (tf.config.experimental.get_memory_info("GPU:0")["peak"] - base) / GIB


class TestGraphModeMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not tf.config.list_physical_devices("GPU"):
            raise unittest.SkipTest("needs a GPU")
        rng = np.random.default_rng(0)
        shape = (N_MEM,) + SHAPE2
        labels = rng.integers(0, 19, size=shape)
        cls.onehot = tf.one_hot(tf.constant(labels, tf.int32), 19, dtype=tf.float32)
        cls.u = tf.Variable(smooth_field(SHAPE2, np.random.default_rng(0), n=N_MEM, max_vox=2.0).astype(np.float32))
        cls.w19 = tf.constant(rng.normal(size=shape + (19,)).astype(np.float32))
        cls.w3 = tf.constant(rng.normal(size=shape + (3,)).astype(np.float32))
        grid = L.index_grid(SHAPE2)[None]
        cls.coords = tf.constant((grid + cls.u).numpy())
        cls.field = tf.constant(cls.u.numpy())

    @classmethod
    def tearDownClass(cls):
        for k in ("onehot", "u", "w19", "w3", "coords", "field"):
            if hasattr(cls, k):
                delattr(cls, k)

    def case_a(self, lean):
        onehot, u, w = self.onehot, self.u, self.w19

        @tf.function
        def f():
            with tf.GradientTape() as tape:
                out = D.warp(onehot, u, lean=lean, lean_images_grad=False)
                loss = tf.reduce_sum(out * w)
            return tf.reduce_sum(tape.gradient(loss, u))
        return peak_gib(f)

    def case_b(self, lean):
        v, w = self.u, self.w3

        @tf.function
        def f():
            with tf.GradientTape() as tape:
                out = tf.recompute_grad(lambda x: D.integrate_svf(x, 7, lean=lean))(tf.convert_to_tensor(v))
                loss = tf.reduce_sum(out * w)
            return tf.reduce_sum(tape.gradient(loss, v))
        return peak_gib(f)

    def case_c(self, images, dy, boundary, images_grad):
        coords = self.coords

        @tf.function
        def f():
            d_img, d_crd = L._lean_linear_backward(images, coords, dy, 0.0, boundary, images_grad)
            s = tf.reduce_sum(d_crd)
            return s if d_img is None else s + tf.reduce_sum(d_img)
        return peak_gib(f)

    def test_graph_mode_memory(self):
        res = {}
        # Case A: one-hot warp, C = 19, zero mode, gradient w.r.t. u only.
        m_a = max(fwd_moment(19), 12 + 4 * 19 + t_bwd(19, False))
        res["A_lean"], res["A_default"] = self.case_a(True), self.case_a(False)
        # Case B: integrator block under tf.recompute_grad, T = 7.
        m_b_lean = 6 * 24 + 12 + fwd_moment(3)
        m_b_def = 7 * 180 + 12 + 12 + 12 + 250
        res["B_lean"], res["B_default"] = self.case_b(True), self.case_b(False)
        was = L._DY_DEP
        try:
            L._DY_DEP = False
            res["B_lean_no_dy_dep"] = self.case_b(True)
        finally:
            L._DY_DEP = was
        # Case C: the backward alone.
        m_c1 = t_bwd(19, False)
        m_c2 = t_bwd(3, True) + 12
        dy19 = self.w19
        res["C1"] = self.case_c(self.onehot, dy19, "zero", False)
        res["C2"] = self.case_c(self.field, self.w3, "border", True)
        model = {"A": m_a, "B_lean": m_b_lean, "B_default": m_b_def, "C1": m_c1, "C2": m_c2}
        lines = ["[graph-mode memory] level-2 field %s, N %d (1 B/voxel/image = %.6f GiB); peak minus inputs, GiB"
                 % ("x".join(map(str, SHAPE2)), N_MEM, B_PER_GIB)]
        lines.append("  A  one-hot warp C19: lean %.3f (model %d B = %.3f, bound %.3f), default %.3f"
                     % (res["A_lean"], m_a, m_a * B_PER_GIB, 1.5 * m_a * B_PER_GIB, res["A_default"]))
        lines.append("  B  integrator T7:   lean %.3f (model %d B = %.3f, bound %.3f), default %.3f (model %d B = %.3f),"
                     " saving %.3f (predicted %.3f, floor 0.85); lean without dy dep %.3f"
                     % (res["B_lean"], m_b_lean, m_b_lean * B_PER_GIB, 1.5 * m_b_lean * B_PER_GIB,
                        res["B_default"], m_b_def, m_b_def * B_PER_GIB, res["B_default"] - res["B_lean"],
                        (m_b_def - m_b_lean) * B_PER_GIB, res["B_lean_no_dy_dep"]))
        lines.append("  C1 backward C19 zero, no images grad: %.3f (model %d B = %.3f, bound 0.58)"
                     % (res["C1"], m_c1, m_c1 * B_PER_GIB))
        lines.append("  C2 backward C3 border, images grad:   %.3f (model %d B = %.3f, bound 0.41)"
                     % (res["C2"], m_c2, m_c2 * B_PER_GIB))
        print("\n" + "\n".join(lines), file=sys.stderr)
        self.assertLessEqual(res["A_lean"], 2.25)
        self.assertLessEqual(res["B_lean"], 1.24)
        self.assertGreaterEqual(res["B_default"] - res["B_lean"], 0.85)
        self.assertLessEqual(res["C1"], 0.58)
        self.assertLessEqual(res["C2"], 0.41)
        self.assertEqual(model["A"], 936)
        self.assertEqual((model["B_lean"], model["B_default"], model["C1"], model["C2"]), (516, 1546, 240, 172))


if __name__ == "__main__":
    unittest.main()
