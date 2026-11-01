"""Streaming prediction engine (predict-script design 4).

FEAT-002: `inference.streaming` (TestMaxElements: per-image element bound and chunk cap against
hand counts; TestStreamedLevel: streamed == the model's own call on random-weight TINY models at
levels 4 and 2, chunks 1, 2 and N, float32 (S_stream) and mixed_float16 (S_fp16), variants;
TestStreamingLevel1: level 1 on the CPU). Measured maxima are printed.

FEAT-001: guard of the bit-identical GroupConvBlock refactor (design 4.2).
`GroupConvBlock._split_call` now delegates to `pre` and `conv_with_group_mean`; its outputs must
equal a verbatim copy of the pre-refactor body bitwise (float32 and mixed_float16, fp16_act_clip 0
and 100 with the clip active, strides 1 and 2). `conv_with_group_mean(pre(x), mean(pre(x)))`
reproduces the block's own call: bitwise with split on, within float rounding with split off (the
concat conv is different arithmetic); the stem-like block (pre_activation=False) has an identity
`pre`.
"""
import logging
import os
import resource
import sys
import tempfile
import time
import unittest
from unittest import mock

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory limit)
from tests.runtime_helpers import free_memory, tiny_config
from tests.test_cascade import randomize
import numpy as np
import tensorflow as tf

from data_loading import params as P
from inference import streaming as ST
from inference.store import DiskStore, RamStore
from models import diffeo
from models import layers as L
from models.config import ModelConfig
from models.group_net import GroupRegLevelModel
from models.layers import clip_fp16
from models.precision import policy_scope


def _legacy_split_call(block, x, training):
    """Verbatim copy of `GroupConvBlock._split_call` at commit 95e58c8 (models/layers.py
    691-704), with ``self`` -> ``block``; reads only block._pre, block.conv (kernel, bias),
    block.strides and block.fp16_act_clip."""
    x = block._pre(x, training, concat=False)
    channels = int(x.shape[-1])
    mean = tf.reduce_mean(x, axis=0, keepdims=True)
    k = tf.cast(block.conv.kernel, x.dtype)
    strides = [1, block.strides, block.strides, block.strides, 1]
    c = block.fp16_act_clip
    y = (clip_fp16(tf.nn.conv3d(x, k[..., :channels, :], strides, "SAME"), c)
         + clip_fp16(tf.nn.conv3d(mean, k[..., channels:, :], strides, "SAME"), c))
    return clip_fp16(tf.nn.bias_add(y, tf.cast(block.conv.bias, x.dtype)), c)


SHAPE = (3, 8, 8, 8, 4)
SCALE = 100.0  # input scale: under float16 some pre-clip values exceed 100 (asserted)


def _input(seed=0):
    rng = np.random.default_rng(seed)
    return (SCALE * rng.normal(size=SHAPE)).astype(np.float32)


def _block(policy, strides, clip, pre_activation=True, filters=6):
    """Built GroupConvBlock (group=True, slope 0.01, seed 0) under ``policy``; the first call
    takes the standard concat path and builds the conv."""
    with policy_scope(policy):
        blk = L.GroupConvBlock(filters, strides=strides, group=True, negative_slope=0.01,
                               pre_activation=pre_activation, seed=0)
    x = tf.cast(tf.constant(_input()), blk.compute_dtype)
    blk(x, training=False)
    blk.fp16_act_clip = float(clip)
    return blk, x


def _pre_clip_max(block, x):
    """Largest |value| among the unclipped conv terms and their bias-added sum."""
    p = block.pre(x, False)
    channels = int(p.shape[-1])
    mean = tf.reduce_mean(p, axis=0, keepdims=True)
    k = tf.cast(block.conv.kernel, p.dtype)
    s = [1, block.strides, block.strides, block.strides, 1]
    a = tf.nn.conv3d(p, k[..., :channels, :], s, "SAME")
    b = tf.nn.conv3d(mean, k[..., channels:, :], s, "SAME")
    y = tf.nn.bias_add(a + b, tf.cast(block.conv.bias, p.dtype))
    return max(float(tf.reduce_max(tf.abs(tf.cast(t, tf.float32)))) for t in (a, b, y))


class TestRefactorGuard(unittest.TestCase):
    """Legacy vs refactored `_split_call`: bitwise equal in all 8 combinations."""

    def tearDown(self):
        free_memory()

    def test_bitwise_all_combinations(self):
        n = 0
        for policy in ("float32", "mixed_float16"):
            for clip in (0.0, 100.0):
                for strides in (1, 2):
                    blk, x = _block(policy, strides, clip)
                    blk.split_group = True
                    if policy == "mixed_float16" and clip > 0:
                        m = _pre_clip_max(blk, x)
                        print("\n[refactor guard] fp16 stride %d max |pre-clip| %.1f"
                              % (strides, m), file=sys.stderr)
                        self.assertGreater(m, clip)  # the clip is active
                    ref = _legacy_split_call(blk, x, False)
                    got = blk._split_call(x, False)
                    self.assertEqual(got.dtype, ref.dtype)
                    self.assertEqual(got.dtype, tf.as_dtype(blk.compute_dtype))
                    np.testing.assert_array_equal(got.numpy(), ref.numpy(),
                                                  err_msg="%s clip %g stride %d"
                                                  % (policy, clip, strides))
                    if clip > 0 and policy == "mixed_float16":
                        self.assertLessEqual(float(tf.reduce_max(tf.abs(got))), clip)
                    n += 1
        self.assertEqual(n, 8)
        print("\n[refactor guard] %d combinations bitwise equal" % n, file=sys.stderr)


class TestConvWithGroupMean(unittest.TestCase):
    """`conv_with_group_mean(pre(x), mean(pre(x)), inner_clip=split_group)` vs `block(x)`."""

    def tearDown(self):
        free_memory()

    def _engine(self, blk, x):
        p = blk.pre(x, False)
        mean = tf.reduce_mean(p, axis=0, keepdims=True)
        return blk.conv_with_group_mean(p, mean, inner_clip=blk.split_group)

    def test_split_on_bitwise(self):
        for policy in ("float32", "mixed_float16"):
            for clip in (0.0, 100.0):
                for strides in (1, 2):
                    blk, x = _block(policy, strides, clip)
                    blk.split_group = True
                    np.testing.assert_array_equal(self._engine(blk, x).numpy(),
                                                  blk(x, training=False).numpy(),
                                                  err_msg="%s clip %g stride %d"
                                                  % (policy, clip, strides))

    def test_split_off_close_float32(self):
        worst = 0.0
        for strides in (1, 2):
            blk, x = _block("float32", strides, 0.0)
            self.assertFalse(blk.split_group)
            got = self._engine(blk, x).numpy()
            ref = blk(x, training=False).numpy()
            worst = max(worst, float(np.max(np.abs(got - ref)) / np.max(np.abs(ref))))
            np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5 * float(np.max(np.abs(ref))))
        print("\n[conv_with_group_mean] split off float32 max |diff| / max |ref| %.2e" % worst,
              file=sys.stderr)

    def test_stem_pre_identity(self):
        for strides in (1, 2):
            blk, x = _block("float32", strides, 0.0, pre_activation=False, filters=4)
            self.assertIsNone(blk.norm)
            self.assertIsNone(blk.act)
            np.testing.assert_array_equal(blk.pre(x, False).numpy(), x.numpy())
            got = self._engine(blk, x).numpy()
            ref = blk(x, training=False).numpy()
            np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5 * float(np.max(np.abs(ref))))


# ============================================================================ FEAT-002
class TestMaxElements(unittest.TestCase):
    """Hand counts at the level shapes, default widths (32, 64, 128, 192, 256, 384) and decoder
    (128, 96, 64, 32, 16). Voxels: L1 6881280, L2 860160, L4 107520."""

    def check(self, level, tasks, split, expected):
        self.assertEqual(ST.max_elements_per_image(level, ModelConfig(), tasks, split), expected)

    def test_level1(self):
        self.check(1, ("deform",), True, (440401920, "d1 skip concat"))     # 64 ch x 6881280
        self.check(1, ("rigid",), True, (220200960, "stem output"))         # 32 ch x 6881280
        self.check(1, ("affine",), False, (220200960, "stem output"))
        self.check(1, ("deform",), False, (880803840, "d1 group concat"))   # 2C = 128 ch
        self.assertEqual(ST.chunk_cap(1, ModelConfig(), ("deform",)), 4)
        self.assertEqual(ST.chunk_cap(1, ModelConfig(), ("rigid",)), 9)
        self.assertEqual(ST.chunk_cap(1, ModelConfig(), "deform"), 4)
        e_off, _ = ST.max_elements_per_image(1, ModelConfig(), ("deform",), False)
        self.assertEqual((2 ** 31 - 1) // e_off, 2)   # split off: larger bound, lower cap

    def test_level4(self):
        self.check(4, ("deform",), True, (24084480, "d3 skip concat"))      # (96 + 128) x 107520
        self.check(4, ("deform",), False, (48168960, "d3 group concat"))
        self.check(4, ("rigid",), True, (13762560, "l3 output"))            # 128 x 107520
        self.check(4, ("rigid",), False, (27525120, "l4 group concat"))     # 2 x 128 x 107520
        self.assertEqual(ST.chunk_cap(4, ModelConfig(), ("deform",)), 89)

    def test_level2(self):
        self.check(2, ("deform",), True, (110100480, "d2 skip concat"))     # (64 + 64) x 860160
        self.check(2, ("deform",), False, (220200960, "d2 group concat"))
        self.check(2, ("rigid",), True, (55050240, "l2 output"))            # 64 x 860160
        self.check(2, ("rigid",), False, (55050240, "l2 group concat"))     # 2 x 32 x 860160
        self.assertEqual(ST.chunk_cap(2, ModelConfig(), ("deform",)), 19)

    def test_config_b_and_bad_task(self):
        cfg = ModelConfig(group_config="B")
        self.assertEqual(ST.max_elements_per_image(1, cfg, ("deform",), False),
                         (880803840, "d1 group concat"))   # the decoder is grouped in B too
        self.assertEqual(ST.max_elements_per_image(4, cfg, ("rigid",), False),
                         (13762560, "l3 output"))           # no encoder group concat in B
        with self.assertRaises(ValueError):
            ST.max_elements_per_image(1, cfg, ("pairwise",), True)


N = 3
B_STREAM_FIELD, B_STREAM_A = 1e-4, 1e-5          # S_stream (design 1.3 AC 0)
B_FP16_MAX, B_FP16_MEAN = 2e-2, 1e-3              # S_fp16 (fields)
B_FP16_A = 1e-2                                   # fp16 linear: A entries (translation mm)


def _images(level, n=N, seed=0, scale=1.0):
    """Blocky random images (4^3 blocks + noise) at the level shape, float32 [n, X, Y, Z, 1]."""
    rng = np.random.default_rng(seed)
    shape = tuple(P.LEVEL_SHAPE[level])
    coarse = rng.uniform(0.0, 1.0, (n,) + tuple(s // 4 for s in shape) + (1,))
    x = coarse.repeat(4, 1).repeat(4, 2).repeat(4, 3)
    return (scale * (x + 0.05 * rng.normal(size=x.shape))).astype(np.float32)


# Level 1 on the CPU: TF 2.10's CPU Conv3D segfaults (rc 139, probed 2026-11-01) on a
# [3, 160, 224, 192, 16] float32 input ([3, ..., 8] and [2, ..., 16] run), which the TINY model's
# d1 conv gets at N 3 (d2 8 + stem 8 channels). TINY with stem 4 and d2 / d1 4 keeps every
# full-resolution conv input at <= 8 channels.
TINY_L1 = dict(filters=(4, 16, 16, 16, 16, 16), decoder_widths=(8, 8, 8, 4, 4))


def _model(level, policy="float32", split=True, group_config="A", clip=0.0, dd="policy",
           seed=1, block="single", build=True, **cfg_kw):
    cfg = tiny_config(deform_decoder=True, seed=1, integration_steps=3, group_config=group_config,
                      block=block, **cfg_kw)
    kw = dict(split_group_conv=split, lean_sampler=True, deform_decoder_dtype=dd)
    if clip:
        kw.update(fp16_act_clip=clip, fp16_act_clip_scope="all")
    with policy_scope(policy):
        m = GroupRegLevelModel(level, cfg, **kw)
        if build:
            m.build_model(1)
    if build:
        randomize(m, seed, 0.02)
    return m


def _ref_deform(m, x, prev=None):
    out = m(tf.constant(x), task="deform", training=False)
    u, u_inv = out["u"], out["u_inv"]
    if prev is not None:
        u, u_inv = diffeo.compose_cascade(tf.constant(prev[0]), tf.constant(prev[1]), u, u_inv,
                                          lean=True)
    return {"u": u.numpy(), "u_inv": u_inv.numpy(), "v_raw": out["v_raw"].numpy()}


def _stream_deform(eng, x, prev=None):
    n = x.shape[0]
    got = {}

    def sink(i0, i1, d):
        for k in ("u", "u_inv", "v_raw"):
            a = d[k].numpy()
            got.setdefault(k, np.zeros((n,) + a.shape[1:], a.dtype))[i0:i1] = a

    prev_fn = None if prev is None else (lambda i0, i1: (prev[0][i0:i1], prev[1][i0:i1]))
    res = eng.deform(n, lambda i0, i1: x[i0:i1], prev_fn=prev_fn, sink=sink)
    return got, res


def _diff(got, ref):
    d = np.abs(np.asarray(got, np.float64) - np.asarray(ref, np.float64))
    return float(d.max()), float(d.mean())


class TestStreamedLevel(unittest.TestCase):
    """Streamed `StreamedLevel` == `model(x, task, training=False)` (design 4.3, AC 1)."""

    def tearDown(self):
        free_memory()

    def _check_model(self, label, m, x, chunks, tasks, fp16, store_factory=RamStore):
        """Every chunk size and task; returns {key: worst max diff}."""
        worst = {}
        ref_d = _ref_deform(m, x) if "deform" in tasks else None
        ref_a = {t: m(tf.constant(x), task=t, training=False)["A_raw"].numpy()
                 for t in tasks if t != "deform"}
        n = x.shape[0]
        for chunk in chunks:
            st = store_factory()
            eng = ST.StreamedLevel(m, st, chunk)
            k = min(chunk, n)
            expect_bounds = [(i0, min(i0 + k, n)) for i0 in range(0, n, k)]
            for task in tasks:
                if task == "deform":
                    got, res = _stream_deform(eng, x)
                    stats = res["stats"]
                    self.assertIsNone(res["u"])
                    for key in ("u", "u_inv", "v_raw"):
                        mx, mean = _diff(got[key], ref_d[key])
                        worst[key] = max(worst.get(key, 0.0), mx)
                        worst[key + "_mean"] = max(worst.get(key + "_mean", 0.0), mean)
                        msg = "%s chunk %d %s max %.3g mean %.3g" % (label, chunk, key, mx, mean)
                        if fp16:
                            self.assertLessEqual(mx, B_FP16_MAX, msg)
                            self.assertLessEqual(mean, B_FP16_MEAN, msg)
                        else:
                            self.assertLessEqual(mx, B_STREAM_FIELD, msg)
                    passes = set("P%d" % p for p in range(13))
                    if not m.encoder.l2.group and not fp16:
                        passes.discard("P1")   # config B float32: nothing to accumulate in P1
                    self.assertEqual(set(stats["chunks"]), passes)
                    self.assertGreater(stats["max_abs_v"], 0.0)
                else:
                    a = eng.linear(n, lambda i0, i1: x[i0:i1], task).numpy()
                    stats = eng.last_stats
                    mx, _ = _diff(a, ref_a[task])
                    worst["A_" + task] = max(worst.get("A_" + task, 0.0), mx)
                    self.assertLessEqual(mx, B_FP16_A if fp16 else B_STREAM_A,
                                         "%s chunk %d A_raw %s max %.3g" % (label, chunk, task, mx))
                    self.assertNotIn("P7", stats["chunks"])
                self.assertEqual(stats["chunks"]["P0"], expect_bounds)
                self.assertEqual(stats["chunk"], k)
                self.assertEqual(st.nbytes(), 0, "%s: stores left %s" % (label, st.names()))
                if fp16:
                    self.assertTrue(stats["fp16_max_abs"])
                    for v in stats["fp16_max_abs"].values():
                        self.assertLess(v["ratio"], 1.0)
                else:
                    self.assertEqual(stats["fp16_max_abs"], {})
            self.assertEqual(eng.capped, {})
            st.close()
        print("\n[streamed] %s: %s" % (label, ", ".join(
            "%s %.2e" % (k, v) for k, v in sorted(worst.items()))), file=sys.stderr)
        return worst

    def test_float32_levels_4_2(self):
        for level in (4, 2):
            m = _model(level)
            self._check_model("L%d float32" % level, m, _images(level), (1, 2, N),
                              ("deform", "rigid", "affine"), fp16=False)
            free_memory()

    def test_fp16_levels_4_2(self):
        for level in (4, 2):
            m = _model(level, policy="mixed_float16")
            self._check_model("L%d mixed_float16" % level, m, _images(level), (1, 2, N),
                              ("deform", "rigid", "affine"), fp16=True)
            free_memory()

    def test_variants_level2(self):
        variants = (("split off", dict(split=False), False, 1.0),
                    ("config B", dict(group_config="B"), False, 1.0),
                    ("config B split off", dict(group_config="B", split=False), False, 1.0),
                    ("fp16 clip 100", dict(policy="mixed_float16", clip=100.0), True, 100.0),
                    ("fp16 float32_fine", dict(policy="mixed_float16", dd="float32_fine"), True,
                     1.0))
        for label, kw, fp16, scale in variants:
            m = _model(2, **kw)
            x = _images(2, scale=scale)
            if scale != 1.0:
                # Inputs x100 put every fp16 block at the clip (asserted below); the SVF head is
                # scaled by 1/100 so the fields keep the scale-1 size (|v| ~ 0.1-0.2 vox) and the
                # absolute S_fp16 bound measures the engine, not fp16 rounding of 10-vox fields.
                head = m.deform_decoder.svf_head
                head.kernel.assign(head.kernel / scale)
                head.bias.assign(head.bias / scale)
            self._check_model("L2 " + label, m, x, (2,), ("deform", "rigid"), fp16=fp16)
            if "clip" in kw:
                eng = ST.StreamedLevel(m, RamStore(), 2)
                stats = _stream_deform(eng, x)[1]["stats"]
                clipped = [b for b, v in stats["fp16_max_abs"].items()
                           if abs(v["max_abs"] - 100.0) < 1e-3]
                print("\n[streamed] fp16 clip 100: blocks at the clip %s" % clipped,
                      file=sys.stderr)
                self.assertTrue(clipped, "the clip is not active: %s" % stats["fp16_max_abs"])
            if kw.get("dd") == "float32_fine":
                self.assertEqual(m.deform_decoder.d1.compute_dtype, "float32")
                self.assertEqual(m.deform_decoder.d3.compute_dtype, "float16")
            free_memory()

    def test_disk_store_fp16_level4(self):
        m = _model(4, policy="mixed_float16")
        with tempfile.TemporaryDirectory() as tmp:
            parent = os.path.join(tmp, "_stream_tmp")
            self._check_model("L4 mixed_float16 disk", m, _images(4), (2,), ("deform", "affine"),
                              fp16=True, store_factory=lambda: DiskStore(parent))
            self.assertFalse(os.path.exists(parent))

    def test_cascade_store_mode_and_cap(self):
        m = _model(2)
        x = _images(2)
        rng = np.random.default_rng(5)
        coarse = rng.normal(0.0, 0.5, (N,) + tuple(P.LEVEL_SHAPE[4]) + (3,)).astype(np.float32)
        u_prev = diffeo.upsample_displacement(tf.constant(coarse)).numpy()
        u_inv_prev = (-0.9 * u_prev).astype(np.float32)
        prev = (u_prev, u_inv_prev)
        ref = _ref_deform(m, x, prev)
        st = RamStore()
        eng = ST.StreamedLevel(m, st, 2)
        got, _ = _stream_deform(eng, x, prev)
        worst = 0.0
        for key in ("u", "u_inv"):
            mx, _ = _diff(got[key], ref[key])
            worst = max(worst, mx)
            self.assertLessEqual(mx, B_STREAM_FIELD, key)
        # store mode: the same chunks give the same values; the names are prefixed per level
        res = eng.deform(N, lambda i0, i1: x[i0:i1],
                         prev_fn=lambda i0, i1: (u_prev[i0:i1], u_inv_prev[i0:i1]))
        self.assertEqual((res["u"], res["u_inv"]), ("L2_u", "L2_u_inv"))
        self.assertEqual(sorted(st.names()), ["L2_u", "L2_u_inv"])
        np.testing.assert_array_equal(st.get("L2_u", 0, N), got["u"])
        np.testing.assert_array_equal(st.get("L2_u_inv", 0, N), got["u_inv"])
        st.free("L2_u")
        st.free("L2_u_inv")
        # chunk cap: a limit that admits one image per chunk caps chunk 2 to 1 (WARNING)
        e_max, tensor = ST.max_elements_per_image(2, m.config, ("deform",), True)
        with mock.patch.object(ST, "INT32_LIMIT", 2 * e_max):
            eng2 = ST.StreamedLevel(m, st, 2)
            with self.assertLogs(ST.logger, level=logging.WARNING):
                got2, res2 = _stream_deform(eng2, x)
        self.assertEqual(eng2.capped["deform"],
                         {"requested": 2, "used": 1, "e_max": e_max, "tensor": tensor})
        self.assertEqual(res2["stats"]["chunks"]["P5"], [(0, 1), (1, 2), (2, 3)])
        ref0 = _ref_deform(m, x)
        mx, _ = _diff(got2["u"], ref0["u"])
        self.assertLessEqual(mx, B_STREAM_FIELD)
        print("\n[streamed] L2 cascade compose max %.2e, capped run u max %.2e" % (worst, mx),
              file=sys.stderr)

    def test_lean_bitwise(self):
        m = _model(2)
        x = _images(2)
        a, _ = _stream_deform(ST.StreamedLevel(m, RamStore(), 2, lean=True), x)
        b, _ = _stream_deform(ST.StreamedLevel(m, RamStore(), 2, lean=False), x)
        for key in ("u", "u_inv", "v_raw"):
            np.testing.assert_array_equal(a[key], b[key], key)

    def test_residual_rejected(self):
        m = _model(4, block="residual", build=False)
        with self.assertRaises(ValueError):
            ST.StreamedLevel(m, RamStore(), 2)


class TestStreamingLevel1(unittest.TestCase):
    """Level 1 (pool2, up_d1, fused stem / d1 passes) on the CPU inside the GPU process: TINY
    with the `TINY_L1` full-resolution widths, float32, N 3, chunks 1 and 2, deform. Never
    skipped for memory."""

    def tearDown(self):
        free_memory()

    def test_level1_cpu(self):
        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("tf_gpu env requires a visible GPU (the level-1 case runs on /CPU:0)")
        t0 = time.time()
        with tf.device("/CPU:0"):
            m = _model(1, **TINY_L1)
            x = _images(1)
            ref = _ref_deform(m, x)
            t_ref = time.time() - t0
            worst = {}
            for chunk in (1, 2):
                st = RamStore()
                eng = ST.StreamedLevel(m, st, chunk)
                got, res = _stream_deform(eng, x)
                for key in ("u", "u_inv", "v_raw"):
                    mx, _ = _diff(got[key], ref[key])
                    worst[key] = max(worst.get(key, 0.0), mx)
                    self.assertLessEqual(mx, B_STREAM_FIELD, "chunk %d %s" % (chunk, key))
                k = chunk
                self.assertEqual(res["stats"]["chunks"]["P11"],
                                 [(i0, min(i0 + k, N)) for i0 in range(0, N, k)])
                self.assertEqual(st.nbytes(), 0)
                self.assertGreater(st.peak_nbytes(), 0)
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print("\n[streamed] L1 float32 CPU: %s; reference %.0f s, total %.0f s, peak RSS %.1f GB"
              % (", ".join("%s %.2e" % kv for kv in sorted(worst.items())), t_ref,
                 time.time() - t0, rss_gb), file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
