"""Fine-grid integration of frozen cascade stages (`.agents/tasks/fine-grid-integration/plan.md`
D1-D4, D6, items 1-3): `diffeo.exp_compose_upsampled`, `cascade.integrate_frozen_at`, the
``integration_grid`` of `FrozenStage` / `FrozenChain` ('native' = today's formulas, bitwise;
'current' = the frozen SVFs integrated on the grid of the level that uses them), the
--frozen-integration-grid chain rule and main.py's config.json records. TINY models under the
6000 MB GTX 1080 cap; level-1 chain forwards at N 2 (no level-1 training step).

TestInferenceFine (plan D5, item 4): predict.py --integration-grid fine / auto in the streamed and
unchunked runners (default plan, --levels plans, level-4 checkpoints, TTA, affine+deform, the
level-1 chain on /CPU:0), the loader rule, LoadedModel checks and the CLI flag.
"""
import argparse
import contextlib
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import unittest

import tests.tf_setup  # noqa: F401  (log level, TF32 off, GPU memory cap)
import numpy as np
import tensorflow as tf

import main as MAIN
from inference import cli as CLI
from inference import levels as LVS
from inference import loader as LD
from inference import predict as PR
from inference import tta as TTA
from inference.loader import LoadedModel
from inference.runner import StreamedRunner, UnchunkedRunner, _pool_to
from inference.store import RamStore
from models import cascade
from models import diffeo
from models.group_net import GroupRegLevelModel
from models.precision import policy_scope
from runtime import args as A
from runtime import checkpoint as C
from runtime import fixed_data as fd
from tests import test_cascade as TC
from tests import test_predict_runner as TPR
from tests.predict_helpers import save_cascade_run, save_level4_run, write_subjects
from tests.runtime_helpers import TINY_ARGS, free_memory, tiny_config
from tests.test_predict_streaming import TINY_L1, _images
from tests.test_trainer import _OpDeterminism

WHILE_OPS = ("While", "StatelessWhile")
T_STEPS = TC.STAGE_KW["integration_steps"]


def _while_ops(fn, shape):
    g = tf.function(fn).get_concrete_function(tf.TensorSpec(shape, tf.float32)).graph
    return [op.name for op in g.get_operations() if op.type in WHILE_OPS]


def _max_diff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _np(d, keys=None):
    """{k: numpy} of a dict of tensors (host copies: level-1 fields stay off the 6000 MB GPU cap)."""
    return {k: np.asarray(d[k]) for k in (keys or d)}


def _traced_grads_none(fn, shape):
    """{key: gradient wrt the input is None} for every output of `fn`, from a TRACED tape (the
    function is never executed: level-1 eager tapes would keep every intermediate field)."""
    res = {}

    def f(t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            out = fn(t)
        for k in sorted(out):
            res[k] = tape.gradient(tf.reduce_sum(out[k]), t) is None
        return tf.constant(0.0)

    tf.function(f).get_concrete_function(tf.TensorSpec(shape, tf.float32))
    return res


class TestExpComposeUpsampled(unittest.TestCase):
    """Item 1: per-image integration on the common grid == the batched reference."""

    def setUp(self):
        rng = np.random.default_rng(5)
        self.v_coarse = tf.constant(np.stack([fd.random_svf((4, 6, 5), rng, 1.0, 1.5) for _ in range(2)]))
        self.v_mid = tf.constant(np.stack([fd.random_svf((8, 12, 10), rng, 1.5, 2.0) for _ in range(2)]))

    def tearDown(self):
        free_memory()

    def test_single_svf(self):
        up = diffeo.upsample_displacement
        for k, v in ((0, self.v_mid), (1, self.v_mid), (2, self.v_coarse)):
            ref = v
            for _ in range(k):
                ref = up(ref)
            want = diffeo.exp_and_inverse(ref, T_STEPS)
            got = diffeo.exp_compose_upsampled([v], [k], T_STEPS)
            self.assertEqual(len(got), 2)
            for g, w in zip(got, want):
                self.assertEqual(tuple(g.shape), tuple(w.shape))
                self.assertEqual(g.dtype, tf.float32)
                np.testing.assert_allclose(g.numpy(), w.numpy(), rtol=0, atol=1e-5)
            self.assertGreater(float(tf.reduce_max(tf.abs(got[0]))), 0.1)

    def manual_two(self):
        up = diffeo.upsample_displacement
        a, ai = diffeo.exp_and_inverse(up(up(self.v_coarse)), T_STEPS)
        b, bi = diffeo.exp_and_inverse(up(self.v_mid), T_STEPS)
        return diffeo.compose(a, b), diffeo.compose(bi, ai), a, ai

    def test_two_svfs_compose_order_and_batched(self):
        u, ui, a, ai = self.manual_two()
        got = diffeo.exp_compose_upsampled([self.v_coarse, self.v_mid], [2, 1], T_STEPS)
        diffs = [_max_diff(got[0], u), _max_diff(got[1], ui)]
        print("\n[FGI-1] two SVFs map_fn vs batched max |diff| u %.3g, u_inv %.3g" % tuple(diffs), file=sys.stderr)
        np.testing.assert_allclose(got[0].numpy(), u.numpy(), rtol=0, atol=1e-6)
        np.testing.assert_allclose(got[1].numpy(), ui.numpy(), rtol=0, atol=1e-6)
        # The other order (phi_mid o phi_coarse) is a different map: the check is not vacuous.
        b = diffeo.exp_and_inverse(diffeo.upsample_displacement(self.v_mid), T_STEPS)[0]
        self.assertGreater(_max_diff(got[0], diffeo.compose(b, a)), 1e-3)
        # Graph mode (tf.function, one While loop) gives the same values.
        # Graph mode (tf.function, one While loop): same values up to Grappler's float32 rewrites
        # (measured max |diff| 1.4e-6 on 2 of 46080 elements, GTX 1080).
        fn = tf.function(lambda p, q: diffeo.exp_compose_upsampled([p, q], [2, 1], T_STEPS))
        g2 = fn(self.v_coarse, self.v_mid)
        np.testing.assert_allclose(g2[0].numpy(), u.numpy(), rtol=0, atol=1e-5)
        np.testing.assert_allclose(g2[1].numpy(), ui.numpy(), rtol=0, atol=1e-5)

    def test_return_first(self):
        u, ui, a, ai = self.manual_two()
        got = diffeo.exp_compose_upsampled([self.v_coarse, self.v_mid], [2, 1], T_STEPS, return_first=True)
        self.assertEqual(len(got), 4)
        for g, w in zip(got, (u, ui, a, ai)):
            np.testing.assert_allclose(g.numpy(), w.numpy(), rtol=0, atol=1e-6)
        one = diffeo.exp_compose_upsampled([self.v_mid], [1], T_STEPS, return_first=True)
        np.testing.assert_array_equal(one[0].numpy(), one[2].numpy())
        np.testing.assert_array_equal(one[1].numpy(), one[3].numpy())

    def test_errors(self):
        with self.assertRaises(ValueError):
            diffeo.exp_compose_upsampled([self.v_mid, self.v_mid], [1, 2], T_STEPS)      # shapes differ
        with self.assertRaises(ValueError):
            diffeo.exp_compose_upsampled([self.v_mid], [1, 1], T_STEPS)                  # lengths differ
        with self.assertRaises(ValueError):
            diffeo.exp_compose_upsampled([], [], T_STEPS)
        with self.assertRaises(ValueError):
            diffeo.exp_compose_upsampled([self.v_mid], [-1], T_STEPS)
        with self.assertRaises(ValueError):
            cascade.integrate_frozen_at([(2, self.v_mid)], 4, T_STEPS)                    # level_k < level
        with self.assertRaises(ValueError):
            cascade.integrate_frozen_at([], 1, T_STEPS)
        # integrate_frozen_at: ups = log2(level_k / level).
        got = cascade.integrate_frozen_at([(4, self.v_coarse), (2, self.v_mid)], 1, T_STEPS)
        ref = diffeo.exp_compose_upsampled([self.v_coarse, self.v_mid], [2, 1], T_STEPS)
        for g, w in zip(got, ref):
            np.testing.assert_array_equal(g.numpy(), w.numpy())


def chain_reference(chain, x1, skip4=False, skip2=False):
    """Manual 'current' chain fields: v4 on pool(pool(x1)), exp at level 2 of up(v4) pre-warps
    model2's input, v2, then both integrated at level 1 and composed (phi4 o phi2)."""
    steps = chain.stage4.model.config.integration_steps
    up = diffeo.upsample_displacement
    x2 = cascade.pool_to_coarser(x1)
    v4 = chain.stage4.model(cascade.pool_to_coarser(x2), task="deform", training=False)["v"]
    if skip4:
        v4 = tf.zeros_like(v4)
    u4_2 = diffeo.exp_and_inverse(up(v4), steps)[0]
    v2 = chain.model2(cascade.prewarp_deform(x2, u4_2), task="deform", training=False)["v"]
    if skip2:
        v2 = tf.zeros_like(v2)
    out = {"v4": v4.numpy(), "v2": v2.numpy(), "V": up(v2).numpy(), "V4": up(up(v4)).numpy()}
    a, ai = diffeo.exp_and_inverse(up(up(v4)), steps)
    out.update({"U4": a.numpy(), "U4_inv": ai.numpy()})
    b, bi = diffeo.exp_and_inverse(up(v2), steps)
    out["U"] = diffeo.compose(a, b).numpy()
    del a
    out["U_inv"] = diffeo.compose(bi, ai).numpy()
    return out                                     # numpy (host) arrays


class TestFrozenCurrent(TC.ChainCase):
    """Item 2: FrozenStage / FrozenChain in 'current' mode == manual references; native unchanged."""

    def stage_models(self, seed=9):
        cfg = tiny_config(**TC.STAGE_KW)
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        TC.randomize(m4, seed, 0.02)
        return cfg, m4

    def test_stage_level2(self):
        _, m4 = self.stage_models()
        nat = cascade.FrozenStage.from_model(m4)
        cur = cascade.FrozenStage.from_model(m4, integration_grid="current")
        self.assertEqual((nat.integration_grid, cur.integration_grid), ("native", "current"))
        with self.assertRaises(AttributeError):
            cur.integration_grid = "native"
        x = tf.constant(fd.synthetic_deform_group(2, 2)["images"])
        up = diffeo.upsample_displacement
        with _OpDeterminism():
            out = nat.model(cascade.pool_to_coarser(x), task="deform", training=False)
            pn = nat.deform(x)
            for k, s in (("U", "u"), ("U_inv", "u_inv"), ("V", "v")):         # native: bitwise
                np.testing.assert_array_equal(pn[k].numpy(), up(out[s]).numpy(), err_msg=k)
            pc = cur.deform(x)
            v = cur.svf(x)
        np.testing.assert_array_equal(v.numpy(), out["v"].numpy())
        # No gradient to the input (traced tape: an eager tape over the per-image loop keeps every
        # sampler intermediate and exceeds the 6000 MB cap at level 2).
        shape = (2,) + TC.SHAPE2 + (1,)
        got = _traced_grads_none(lambda t: dict(cur.deform(t), v=cur.svf(t)), shape)
        self.assertEqual(sorted(got), ["U", "U_inv", "V", "v"])
        self.assertTrue(all(got.values()), got)
        want_u, want_ui = diffeo.exp_and_inverse(up(out["v"]), T_STEPS)
        d = {"U": _max_diff(pc["U"], want_u), "U_inv": _max_diff(pc["U_inv"], want_ui),
             "native vs current U": _max_diff(pc["U"], pn["U"])}
        print("\n[FGI-2] level-2 current vs manual: %s" % ", ".join("%s %.3g" % kv for kv in d.items()),
              file=sys.stderr)
        np.testing.assert_allclose(pc["U"].numpy(), want_u.numpy(), rtol=0, atol=1e-5)
        np.testing.assert_allclose(pc["U_inv"].numpy(), want_ui.numpy(), rtol=0, atol=1e-5)
        np.testing.assert_array_equal(pc["V"].numpy(), pn["V"].numpy())
        self.assertEqual(tuple(pc["U"].shape), (2,) + TC.SHAPE2 + (3,))
        self.assertGreater(d["native vs current U"], 1e-5)                      # the grids differ
        # Graph: native has no While op, current has the per-image map_fn loop.
        self.assertEqual(_while_ops(lambda t: nat.deform(t)["U"], shape), [])
        self.assertTrue(_while_ops(lambda t: cur.deform(t)["U"], shape))

    def test_bad_grid_values(self):
        cfg, m4 = self.stage_models()
        for bad in ("fine", None, "Native"):
            with self.assertRaises(ValueError):
                cascade.FrozenStage.from_model(m4, integration_grid=bad)
        with self.assertRaises(ValueError):
            cascade.FrozenStage.from_config(cfg, integration_grid="fine")
        with self.assertRaises(ValueError):
            cascade.FrozenChain.from_config(cfg, integration_grid="fine")
        self.assertEqual(cascade.INTEGRATION_GRIDS, ("native", "current"))
        self.assertEqual(A.FROZEN_INTEGRATION_GRIDS, cascade.INTEGRATION_GRIDS)
        m2 = cascade._build_frozen_model(cfg, level=2)
        cur4 = cascade.FrozenStage.from_model(m4, integration_grid="current")
        self.assertEqual(cascade.FrozenChain(cur4, m2).integration_grid, "current")
        self.assertEqual(cascade.FrozenChain(cur4, m2, "current").integration_grid, "current")
        with self.assertRaises(ValueError):
            cascade.FrozenChain(cur4, m2, "native")
        with self.assertRaises(ValueError):
            cascade.FrozenChain(cascade.FrozenStage(m4), m2, "current")
        self.assertEqual(cascade.FrozenChain.from_config(cfg).integration_grid, "native")
        ch = cascade.FrozenChain.from_config(cfg, integration_grid="current")
        self.assertEqual((ch.integration_grid, ch.stage4.integration_grid), ("current", "current"))
        with self.assertRaises(AttributeError):
            ch.integration_grid = "native"

    def test_chain_level1(self):
        _, m4, m2 = self.frozen_models()
        nat = cascade.FrozenChain.from_models(m4, m2)
        cur = cascade.FrozenChain.from_models(m4, m2, integration_grid="current")
        self.assertEqual((nat.integration_grid, cur.integration_grid), ("native", "current"))
        self.assertEqual(cur.stage4.integration_grid, "current")
        x = tf.constant(self.dgroup1["images"])
        up = diffeo.upsample_displacement
        keys = ("U", "U_inv", "V", "U4", "U4_inv", "V4")
        with _OpDeterminism():
            # Native: the pre-change formulas, bitwise (host copies, compared one by one).
            x2 = cascade.pool_to_coarser(x)
            p4 = nat.stage4.deform(x2)
            o2 = nat.model2(cascade.prewarp_deform(x2, p4["U"]), task="deform", training=False)
            v2_native = o2["v"].numpy()
            u2, u2i = diffeo.compose_cascade(p4["U"], p4["U_inv"], o2["u"], o2["u_inv"])
            old = {"U": up(u2).numpy(), "U_inv": up(u2i).numpy(), "V": up(o2["v"]).numpy(),
                   "U4": up(p4["U"]).numpy(), "U4_inv": up(p4["U_inv"]).numpy(), "V4": up(p4["V"]).numpy()}
            del p4, o2, u2, u2i
            dn = _np(nat.deform_stages(x))
            self.assertEqual(sorted(dn), sorted(keys))
            for k in keys:
                np.testing.assert_array_equal(dn[k], old[k], err_msg="native " + k)
            np.testing.assert_array_equal(_np(nat.deform(x))["U"], old["U"])
            del old
            np.testing.assert_array_equal(nat.svfs(x)[2].numpy(), v2_native)   # native svfs: native pre-warp
        ref = chain_reference(cur, x)
        dc = _np(cur.deform(x))
        self.assertEqual(sorted(dc), ["U", "U_inv", "V"])
        diffs = {"deform/" + k: _max_diff(dc[k], ref[k]) for k in dc}
        u_cur = dc["U"]
        del dc
        dsc = _np(cur.deform_stages(x))
        self.assertEqual(sorted(dsc), sorted(keys))
        diffs.update({"deform_stages/" + k: _max_diff(dsc[k], ref[k]) for k in keys})
        np.testing.assert_array_equal(dsc["V4"], dn["V4"])                      # V4 = up(up(v4)) both modes
        del dsc
        print("\n[FGI-2] level-1 chain current vs manual max |diff|: %s" % ", ".join(
            "%s %.3g" % kv for kv in sorted(diffs.items())), file=sys.stderr)
        for k, v in diffs.items():
            self.assertLessEqual(v, 1e-5, k)
        sc = cur.svfs(x)
        np.testing.assert_array_equal(sc[4].numpy(), ref["v4"])
        self.assertLessEqual(_max_diff(sc[2], ref["v2"]), 1e-5)
        self.assertGreater(_max_diff(sc[2], v2_native), 1e-6)                   # the pre-warp differs
        self.assertEqual(u_cur.shape, (2,) + TC.SHAPE1 + (3,))
        self.assertGreater(_max_diff(u_cur, dn["U"]), 1e-5)
        self.assertGreater(float(np.abs(u_cur).max()), 1e-3)
        del ref, dn, u_cur, sc
        # No gradient to the input (traced tapes), native graphs without While ops.
        shape = (2,) + TC.SHAPE1 + (1,)
        for name, fn in (("deform", cur.deform), ("deform_stages", cur.deform_stages),
                         ("deform_skip", lambda t: cur.deform_skip(t, tf.constant(True), tf.constant(False)))):
            got = _traced_grads_none(fn, shape)
            self.assertTrue(got and all(got.values()), (name, got))
        self.assertEqual(_while_ops(lambda t: nat.deform(t)["U"], shape), [])
        self.assertEqual(_while_ops(lambda t: nat.deform_stages(t)["U"], shape), [])
        self.assertTrue(_while_ops(lambda t: cur.deform(t)["U"], shape))

    def test_chain_skips(self):
        _, m4, m2 = self.frozen_models()
        cur = cascade.FrozenChain.from_models(m4, m2, integration_grid="current")
        x = tf.constant(self.dgroup1["images"])
        yes, no = tf.constant(True), tf.constant(False)
        full = _np(cur.deform(x), ("U", "U_inv"))
        diffs = {}
        for name, s4, s2 in (("skip4", yes, None), ("skip2", None, yes), ("both", yes, yes),
                             ("none", no, no)):
            got = _np(cur.deform_skip(x, skip4=s4, skip2=s2))
            self.assertEqual(sorted(got), ["U", "U_inv"])
            ref = chain_reference(cur, x, skip4=s4 is yes, skip2=s2 is yes)
            for k in ("U", "U_inv"):
                diffs["%s/%s" % (name, k)] = _max_diff(got[k], ref[k])
            if name == "skip2":                     # = the level-4 factor alone (deform_stages U4)
                np.testing.assert_allclose(ref["U"], ref["U4"], rtol=0, atol=1e-6)
            if name == "both":
                self.assertEqual(float(np.abs(got["U"]).max()), 0.0)
                self.assertEqual(float(np.abs(got["U_inv"]).max()), 0.0)
            if name == "none":
                for k in ("U", "U_inv"):
                    np.testing.assert_allclose(got[k], full[k], rtol=0, atol=1e-6)
            del got, ref
        print("\n[FGI-2] level-1 chain current deform_skip vs manual max |diff|: %s" % ", ".join(
            "%s %.3g" % kv for kv in sorted(diffs.items())), file=sys.stderr)
        for k, v in diffs.items():
            self.assertLessEqual(v, 1e-5, k)


class TestChainRule(unittest.TestCase):
    """Item 3: `A.resolve_frozen_integration_grid` (plan D4) and the parse check (TF-free)."""

    R = staticmethod(A.resolve_frozen_integration_grid)

    def test_fresh_level2(self):
        self.assertEqual(self.R(None, source_level=4), "native")
        self.assertEqual(self.R("native", source_level=4), "native")
        self.assertEqual(self.R("current", source_level=4), "current")
        self.assertEqual(self.R(None), "native")
        with self.assertRaises(ValueError):
            self.R("fine", source_level=4)

    def test_fresh_level1(self):
        cur = {"frozen_integration_grid": "current"}
        self.assertEqual(self.R(None, source_level=2, source_cascade=cur), "current")
        self.assertEqual(self.R("current", source_level=2, source_cascade=cur), "current")
        self.assertEqual(self.R(None, source_level=2, source_cascade={}), "native")        # old source
        self.assertEqual(self.R(None, source_level=2, source_cascade=None), "native")
        self.assertEqual(self.R("native", source_level=2, source_cascade={"skip_prob": 0.0}), "native")
        for req, src in (("native", cur), ("current", {})):
            with self.assertRaises(ValueError) as cm:
                self.R(req, source_level=2, source_cascade=src)
            msg = str(cm.exception)
            self.assertIn("--frozen-integration-grid %s" % req, msg)
            self.assertIn("trained with --frozen-integration-grid %s" % src.get("frozen_integration_grid", "native"),
                          msg)

    def test_resume(self):
        cur = {"frozen_integration_grid": "current", "source_level": 4}
        self.assertEqual(self.R(None, saved_cascade=cur), "current")
        self.assertEqual(self.R("current", saved_cascade=cur), "current")
        self.assertEqual(self.R(None, saved_cascade={"source_level": 4}), "native")     # record before the key
        # The saved record wins over the source rule (resume never reads the source).
        self.assertEqual(self.R(None, source_level=2, source_cascade={}, saved_cascade=cur), "current")
        for req, saved in (("native", cur), ("current", {})):
            with self.assertRaises(ValueError) as cm:
                self.R(req, saved_cascade=saved)
            self.assertIn("--frozen-integration-grid %s differs" % req, str(cm.exception))
            self.assertIn(saved.get("frozen_integration_grid", "native"), str(cm.exception))

    def test_parse_args(self):
        base = ["--overfit-one-group", "--overfit-synthetic"]
        self.assertIsNone(A.parse_args(base).frozen_integration_grid)
        self.assertEqual(A.parse_args(base + ["--frozen-integration-grid", "native"]).frozen_integration_grid,
                         "native")
        lvl2 = base + ["--level", "2", "--cascade-from", "/nonexistent/ckpt/last"]
        self.assertEqual(A.parse_args(lvl2 + ["--frozen-integration-grid", "current"]).frozen_integration_grid,
                         "current")
        self.assertIsNone(A.parse_args(lvl2).frozen_integration_grid)
        for bad in (base + ["--frozen-integration-grid", "current"],
                    base + ["--frozen-integration-grid", "fine"]):
            with self.assertRaises(SystemExit):
                A.parse_args(bad)


class TestMainCascadeGrid(TC.CascadeCase):
    """Item 3: main.py resolves the grid, records it in config.json (args + cascade) and keeps it
    on a resume; mismatches raise (plan D4). Rigid-only TINY runs (TestCascadeResume pattern)."""

    def read_config(self, d):
        with open(os.path.join(d, "config.json")) as f:
            return json.load(f)

    def grids(self, d):
        cfg = self.read_config(d)
        return cfg["args"]["frozen_integration_grid"], cfg["cascade"]["frozen_integration_grid"]

    def test_main_records_and_rule(self):
        res = os.path.join(self.tmp, "results")
        common = (["--overfit-one-group", "--overfit-synthetic", "--group-size", "4", "--task-ratio", "rigid:1",
                   "--deform-decoder", "true", "--log-every", "1", "--val-every", "0", "--val-png", "false",
                   "--schedule", "constant", "--warmup-steps", "0", "--results", res] + TINY_ARGS + TC.MAIN_COMMON)
        self.assertIn(MAIN.main(common + ["--level", "4", "--run-name", "src", "--overfit-max-steps", "2"]), (0, 1))
        free_memory()
        self.assertIsNone(self.read_config(os.path.join(res, "src"))["args"]["frozen_integration_grid"])
        src = os.path.join(res, "src", "ckpt", "last")
        lvl2 = common + ["--level", "2", "--cascade-from", src, "--run-name", "c"]
        self.assertIn(MAIN.main(lvl2 + ["--frozen-integration-grid", "current", "--overfit-max-steps", "2"]), (0, 1))
        free_memory()
        d = os.path.join(res, "c")
        self.assertEqual(self.grids(d), ("current", "current"))
        rec0 = self.read_config(d)["cascade"]
        # Resume without the flag inherits the saved value; the cascade record is carried over.
        self.assertIn(MAIN.main(lvl2 + ["--overfit-max-steps", "3"]), (0, 1))
        free_memory()
        self.assertEqual(self.grids(d), ("current", "current"))
        self.assertEqual(self.read_config(d)["cascade"], rec0)
        # Resume with a different explicit value -> RuntimeError naming both, before any model.
        with self.assertRaises(RuntimeError) as cm:
            MAIN.main(lvl2 + ["--frozen-integration-grid", "native", "--overfit-max-steps", "4"])
        self.assertIn("--frozen-integration-grid native differs", str(cm.exception))
        self.assertIn("current", str(cm.exception))
        free_memory()
        # A default level-2 run records native in both places.
        self.assertIn(MAIN.main(common + ["--level", "2", "--cascade-from", src, "--run-name", "n",
                                          "--overfit-max-steps", "1"]), (0, 1))
        free_memory()
        self.assertEqual(self.grids(os.path.join(res, "n")), ("native", "native"))
        # Level-1 rule: the level-2 source's record reaches the resolver via upstream_cascade.
        c_ckpt = os.path.join(d, "ckpt", "last")
        _, _, info = C.load_cascade_source(c_ckpt, 1)
        self.assertEqual(info["upstream_cascade"]["frozen_integration_grid"], "current")
        self.assertEqual(A.resolve_frozen_integration_grid(None, source_level=2,
                                                           source_cascade=info["upstream_cascade"]), "current")
        # An explicit mismatch at level 1 exits before any model is built.
        lvl1 = common + ["--level", "1", "--cascade-from", c_ckpt, "--run-name", "l1"]
        with self.assertRaises(SystemExit) as cm:
            MAIN.main(lvl1 + ["--frozen-integration-grid", "native", "--overfit-max-steps", "1"])
        msg = str(cm.exception)
        self.assertIn("main.py: --cascade-from %s" % c_ckpt, msg)
        self.assertIn("trained with --frozen-integration-grid current", msg)
        self.assertFalse(os.path.exists(os.path.join(res, "l1", "config.json")))
        # _cascade_lines prints the grid.
        self.assertIn("frozen integration grid: current", MAIN._cascade_lines(rec0, False, None))


class TestTrainerCurrent(TC.CascadeTrainerCase):
    """Item 3: a level-2 Trainer with a 'current' FrozenStage (runtime/run.py unchanged): the deform
    step trains, the frozen weights never change, validation and BN-recal inputs use the
    current-grid frozen fields."""

    def stages(self, seed=3, gradient_checkpointing=TC.STAGE_CKPT):
        cfg = tiny_config(**TC.STAGE_KW)
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        TC.randomize(m4, seed, 0.02)
        m2 = GroupRegLevelModel(2, cfg, gradient_checkpointing=gradient_checkpointing).build_model(2)
        m2.copy_weights_from(m4)
        return m2, cascade.FrozenStage.from_model(m4, integration_grid="current")

    def test_deform_step_validation_and_recal_input(self):
        tr = self.make(cfg=TC.cascade_config(task_ratio={"deform": 1.0}))
        frozen = tr.cascade
        self.assertEqual(frozen.integration_grid, "current")
        fz_before = TC._vals(frozen.model.weights)
        fp_before = frozen.fingerprint
        tv_before = TC._vals(tr.model.trainable_variables)
        res = tr.train_step("deform", self.batches["deform"])
        self.assertTrue(res["finite"])
        self.assertTrue(np.isfinite(res["loss"]))
        self.assertTrue(TC._all_equal(fz_before, frozen.model.weights))
        self.assertEqual(frozen.fingerprint, fp_before)
        self.assertFalse(TC._all_equal(tv_before, tr.model.trainable_variables))
        # Eval-mode validation forward: its frozen fields are frozen.deform's (traced vs eager).
        x = tf.constant(self.dgroup["images"])
        val = tr._cascade_val_deform_fn(int(x.shape[0]))(x)
        want = frozen.deform(x)
        d = {k: _max_diff(val[k], want[k]) for k in ("U", "U_inv", "V")}
        print("\n[FGI-3] validation frozen fields vs frozen.deform max |diff|: %s"
              % ", ".join("%s %.3g" % kv for kv in d.items()), file=sys.stderr)
        for k, v in d.items():
            self.assertLessEqual(v, 1e-5, k)
        self.assertTrue(np.all(np.isfinite(val["u"].numpy())))
        # BN-recal input (Trainer._recal_input, post-process stubbed): pre-warp with the current U.
        tr._pp_call = lambda task, inputs, synth_override=None: {"images": x}
        x_in, ok = tr._recal_input("deform", None)
        self.assertTrue(ok)
        np.testing.assert_allclose(x_in.numpy(), cascade.prewarp_deform(x, want["U"]).numpy(), rtol=0, atol=1e-6)


# ============================================================================ inference (item 4)
B_FINE_REF = 1e-5          # runner (unchunked, float32) vs eager training-time references


def _fine_level2(seed=3, cfg=None):
    """`TPR._level2` with a 'current' FrozenStage -> LoadedModel(integration_grid='fine')."""
    cfg = cfg or tiny_config(**TC.STAGE_KW)
    m4 = GroupRegLevelModel(4, cfg).build_model(2)
    TC.randomize(m4, seed, 0.02)
    frozen = cascade.FrozenStage.from_model(m4, integration_grid="current")
    with policy_scope("float32"):
        m2 = GroupRegLevelModel(2, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
    m2.copy_weights_from(m4)
    TC.randomize(m2, seed + 1, 0.01)
    return LoadedModel(2, m2, frozen, integration_grid="fine")


def _set_grid(prefix, cascade_value=None, args_value=None):
    """Edit the run's config.json: cascade / args frozen_integration_grid (None = key removed)."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(prefix))), "config.json")
    with open(path) as f:
        cfg = json.load(f)
    for block, value in (("cascade", cascade_value), ("args", args_value)):
        d = cfg.setdefault(block, {})
        d.pop("frozen_integration_grid", None)
        if value is not None:
            d["frozen_integration_grid"] = value
    with open(path, "w") as f:
        json.dump(cfg, f)


class TestInferenceFine(TPR.RunnerCase):
    """Item 4: --integration-grid fine in both runners == the training-time ('current') formulas,
    streamed == unchunked; the loader / LoadedModel / CLI rules (plan D5)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_fgi_inf_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        super(TestInferenceFine, self).tearDown()

    def check_refs(self, label, pairs, atol=B_FINE_REF):
        diffs = {k: _max_diff(g, w) for k, g, w in pairs}
        print("\n[FGI-4] %s vs references max |diff|: %s" % (label, ", ".join(
            "%s %.3g" % kv for kv in sorted(diffs.items()))), file=sys.stderr)
        for k, v in diffs.items():
            self.assertLessEqual(v, atol, "%s %s" % (label, k))

    def test_level2_default_plan(self):
        loaded = _fine_level2()
        self.assertEqual(loaded.knobs["integration_grid"], {"checkpoint": "current", "used": "fine"})
        steps = loaded.model_cfg.integration_steps
        x1 = TPR._full_res(_images(2), 2)
        worst = {}
        out = self.compare_modes("L2 fine", loaded, x1, (1, 2, TPR.N), ("deform", "affine+deform"), False,
                                 worst, fields=TPR.FIELD_KEYS + ("v",), keep_ref=True)
        TPR._print("L2 fine streamed vs unchunked", worst)
        for br in out:
            self.assertEqual(out[br]["summary"]["integration_grid"], "fine", br)
        ref = out["deform"]["ref"]
        # The trainer's level-2 formulas with a 'current' FrozenStage; U1 = both SVFs at level 1.
        fz, model = loaded.frozen, loaded.model
        x2 = _pool_to(x1, 2)
        p = fz.deform(x2)
        o = model(cascade.prewarp_deform(x2, p["U"]), task="deform", training=False)
        u, ui = diffeo.compose_cascade(p["U"], p["U_inv"], o["u"], o["u_inv"])
        U1, U1i = cascade.integrate_frozen_at([(4, fz.svf(x2)), (2, o["v"])], 1, steps)
        self.check_refs("L2 fine unchunked", (("u_ckpt", ref["u_ckpt"], u), ("u_inv_ckpt", ref["u_inv_ckpt"], ui),
                                              ("v", ref["v"], o["v"]), ("U1", ref["U1"], U1),
                                              ("U1_inv", ref["U1_inv"], U1i)))
        # Not the native read-out (upsampled level-2 fields), and not identity.
        self.assertGreater(_max_diff(ref["U1"], diffeo.upsample_displacement(u)), 1e-5)
        self.assertGreater(float(np.abs(ref["U1"]).max()), 1e-3)

    def test_level_plans(self):
        loaded = _fine_level2()
        steps = loaded.model_cfg.integration_steps
        x1 = TPR._full_res(_images(2), 2)
        x2 = _pool_to(x1, 2)
        fz, model = loaded.frozen, loaded.model
        worst = {}
        # --levels 4: truncation, U1 = exp at level 1 of up(up(v4)); no checkpoint fields.
        p4 = LVS.plan(loaded.levels, (4,))
        out = self.compare_modes("L2 fine --levels 4", loaded, x1, (1, TPR.N), ("deform", "affine+deform"),
                                 False, worst, fields=("U1", "U1_inv"), plan=p4, keep_ref=True)
        ref = out["deform"]["ref"]
        U1, U1i = cascade.integrate_frozen_at([(4, fz.svf(x2))], 1, steps)
        self.check_refs("--levels 4", (("U1", ref["U1"], U1), ("U1_inv", ref["U1_inv"], U1i)))
        # --levels 2 (skip 4, untrained skip allowed): the trainer's skip input warp(x2, 0).
        p2 = LVS.plan(loaded.levels, (2,))
        out = self.compare_modes("L2 fine --levels 2", loaded, x1, (1, TPR.N), ("deform",), False, worst,
                                 fields=TPR.FIELD_KEYS + ("v",), plan=p2, keep_ref=True)
        ref = out["deform"]["ref"]
        z = tf.zeros(ref["u_ckpt"].shape, tf.float32)
        o = model(cascade.prewarp_deform(x2, z), task="deform", training=False)
        u, ui = diffeo.compose_cascade(z, z, o["u"], o["u_inv"])
        U1, U1i = cascade.integrate_frozen_at([(2, o["v"])], 1, steps)
        self.check_refs("--levels 2", (("u_ckpt", ref["u_ckpt"], u), ("u_inv_ckpt", ref["u_inv_ckpt"], ui),
                                       ("v", ref["v"], o["v"]), ("U1", ref["U1"], U1),
                                       ("U1_inv", ref["U1_inv"], U1i)))
        TPR._print("L2 fine plans streamed vs unchunked", worst)

    def test_level4_checkpoint(self):
        cfg = tiny_config(**TC.STAGE_KW)
        with policy_scope("float32"):
            m = GroupRegLevelModel(4, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
        TC.randomize(m, 3, 0.02)
        loaded = LoadedModel(4, m, integration_grid="fine")
        self.assertEqual(loaded.knobs["integration_grid"], {"checkpoint": "native", "used": "fine"})
        x1 = TPR._full_res(_images(2), 2)
        worst = {}
        out = self.compare_modes("L4 fine", loaded, x1, (1, TPR.N), ("deform", "affine+deform"), False, worst,
                                 fields=TPR.FIELD_KEYS + ("v",), keep_ref=True)
        TPR._print("L4 fine streamed vs unchunked", worst)
        ref = out["deform"]["ref"]
        o = m(_pool_to(x1, 4), task="deform", training=False)
        up = diffeo.upsample_displacement
        a, ai = diffeo.exp_and_inverse(up(up(o["v"])), cfg.integration_steps)
        self.check_refs("L4 fine", (("U1", ref["U1"], a), ("U1_inv", ref["U1_inv"], ai),
                                    ("u_ckpt", ref["u_ckpt"], o["u"]), ("v", ref["v"], o["v"])))
        self.assertGreater(_max_diff(ref["U1"], up(up(o["u"]))), 1e-5)            # not the native read-out

    def test_tta(self):
        loaded = _fine_level2()
        x1 = TPR._full_res(_images(2), 2)
        chunks = {"l1": 2, "l2": 2, "l4": 2}
        keys = ("A",) + TPR.FIELD_KEYS + ("v",)

        def run(tta, br):
            st = RamStore()
            try:
                got, summ = TPR._collect(StreamedRunner(loaded, st, chunks=chunks, tta=tta), x1, br, keys)
                self.assertEqual(st.nbytes(), 0, st.names())
            finally:
                st.close()
            return got, summ
        base, s0 = run(None, "deform")
        self.assertEqual(s0["integration_grid"], "fine")
        self.assertIsNone(s0["tta"])
        idt, s1 = run(TTA.TTAConfig("flip8", loaded.levels, flips=TTA.IDENTITY_ONLY), "deform")
        self.assertEqual(sorted(idt), sorted(base))
        for k in base:
            np.testing.assert_array_equal(idt[k], base[k], err_msg=k)
        f8, s8 = run(TTA.TTAConfig("flip8", loaded.levels), "affine+deform")
        self.assertEqual(sorted(f8), sorted(keys))
        for k, a in f8.items():
            self.assertTrue(np.all(np.isfinite(a)), k)
        self.assertEqual(s8["integration_grid"], "fine")
        self.assertTrue(s8["tta"]["rows"])
        self.assertIn("l2/deform", s8["tta"]["nonfinite_flips"])
        self.assertFalse(any(v for v in s8["tta"]["nonfinite_flips"].values()), s8["tta"]["nonfinite_flips"])
        self.assertGreater(float(np.abs(f8["U1"]).max()), 1e-3)

    def test_level1_chain_cpu(self):
        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("tf_gpu env requires a visible GPU (the level-1 case runs on /CPU:0)")
        t0 = time.time()
        worst = {}
        with tf.device("/CPU:0"):
            cfg = tiny_config(**dict(TC.STAGE_KW, **TINY_L1))
            m4 = GroupRegLevelModel(4, cfg).build_model(2)
            TC.randomize(m4, 3, 0.02)
            m2 = GroupRegLevelModel(2, cfg).build_model(2)
            m2.copy_weights_from(m4)
            TC.randomize(m2, 4, 0.01)
            chain = cascade.FrozenChain.from_models(m4, m2, integration_grid="current")
            with policy_scope("float32"):
                m1 = GroupRegLevelModel(1, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
            m1.copy_weights_from(chain.model2)
            TC.randomize(m1, 5, 0.01)
            loaded = LoadedModel(1, m1, chain, integration_grid="fine")
            x1 = _images(1, n=2)
            out = self.compare_modes("L1 fine CPU", loaded, x1, (1,), ("deform",), False, worst,
                                     fields=("U1", "U1_inv"), keep_ref=True)
            ref = out["deform"]["ref"]
            self.assertEqual(out["deform"]["summary"]["integration_grid"], "fine")
            # Level-1 model input and prior of a streamed run (one chunk) == the training pre-warp.
            sr = StreamedRunner(loaded, RamStore(), chunks={"l1": 2, "l2": 2, "l4": 2})
            cap = {}
            orig = sr._deform_stage

            def spy(lv, eng, n, input_fn, prev_fn, sink):
                if lv == 1:
                    cap["x_in"] = np.asarray(input_fn(0, n))
                    cap["U"] = np.asarray(prev_fn(0, n)[0])
                return orig(lv, eng, n, input_fn, prev_fn, sink)
            sr._deform_stage = spy
            got, _ = TPR._collect(sr, x1, "deform", ("U1", "U1_inv"))
            sr.store.close()
            xt = tf.constant(x1)
            U = chain.deform(xt)["U"]
            want_x = cascade.prewarp_deform(xt, U)
            self.check_refs("L1 fine streamed", (("model input", cap["x_in"], want_x), ("prior U", cap["U"], U),
                                                 ("U1 chunk 2 vs unchunked", got["U1"], ref["U1"]),
                                                 ("U1_inv chunk 2 vs unchunked", got["U1_inv"], ref["U1_inv"])),
                            atol=TPR.B_STREAM_FIELD)
        TPR._print("L1 fine CPU streamed vs unchunked", worst, "; total %.0f s" % (time.time() - t0))

    def save_level2(self, name, cfg=None):
        cfg = cfg or tiny_config(**TC.STAGE_KW)
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        TC.randomize(m4, 3, 0.02)
        frozen = cascade.FrozenStage.from_model(m4)
        m2 = GroupRegLevelModel(2, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
        m2.copy_weights_from(m4)
        TC.randomize(m2, 4, 0.01)
        return save_cascade_run(self.tmp, name, m2, frozen), cfg

    def assert_loaded(self, lm, used, ckpt, frozen=None):
        self.assertEqual(lm.integration_grid, used)
        self.assertEqual(lm.knobs["integration_grid"], {"checkpoint": ckpt, "used": used})
        if frozen is not None:
            self.assertEqual(lm.frozen.integration_grid, frozen)

    def test_loader_rule(self):
        prefix, cfg = self.save_level2("l2")
        self.assert_loaded(LD.load_model(prefix, "deform"), "native", "native", "native")   # no key: native
        _set_grid(prefix, cascade_value="current")
        lm = LD.load_model(prefix, "deform")
        self.assert_loaded(lm, "fine", "current", "current")
        self.assert_loaded(LD.load_model(prefix, "deform", integration_grid="fine"), "fine", "current", "current")
        with self.assertLogs(LD.logger, logging.WARNING) as logs:
            lm_nat = LD.load_model(prefix, "deform", integration_grid="native")
        self.assertTrue(any("--integration-grid native overrides" in m for m in logs.output), logs.output)
        self.assert_loaded(lm_nat, "native", "current", "native")
        # args fallback (no cascade key)
        _set_grid(prefix, cascade_value=None, args_value="current")
        self.assert_loaded(LD.load_model(prefix, "affine+deform"), "fine", "current", "current")
        # invalid values
        _set_grid(prefix, cascade_value="fine")
        with self.assertRaisesRegex(LD.LoaderError, "frozen_integration_grid"):
            LD.load_model(prefix, "deform")
        _set_grid(prefix, cascade_value="current")
        for bad in ("current", "Fine", None):
            with self.assertRaises(LD.LoaderError):
                LD.load_model(prefix, "deform", integration_grid=bad)
        # LoadedModel consistency checks
        with self.assertRaisesRegex(ValueError, "frozen stages"):
            LoadedModel(2, lm_nat.model, lm_nat.frozen, integration_grid="fine")
        with self.assertRaisesRegex(ValueError, "frozen stages"):
            LoadedModel(2, lm.model, lm.frozen)
        with self.assertRaises(ValueError):
            LoadedModel(2, lm.model, lm.frozen, integration_grid="current")
        self.assertEqual(LoadedModel(2, lm_nat.model, lm_nat.frozen).knobs["integration_grid"],
                         {"checkpoint": "native", "used": "native"})
        # predict._load passes the flag through.
        ns = argparse.Namespace(checkpoint=prefix, branches="deform", precision="float32", device="gpu",
                                integration_grid="auto")
        self.assert_loaded(PR._load(ns), "fine", "current", "current")
        ns.integration_grid = "native"
        with self.assertLogs(LD.logger, logging.WARNING):
            self.assert_loaded(PR._load(ns), "native", "current", "native")
        del ns.integration_grid                                     # linear-eval: no flag -> auto
        self.assert_loaded(PR._load(ns), "fine", "current", "current")
        # Level 4: auto = native; explicit fine warns (a level-4 run has no frozen grid).
        prefix4, _ = save_level4_run(self.tmp, "l4", cfg, seed=5)
        self.assert_loaded(LD.load_model(prefix4, "deform"), "native", "native")
        with self.assertLogs(LD.logger, logging.WARNING):
            lm4 = LD.load_model(prefix4, "deform", integration_grid="fine")
        self.assert_loaded(lm4, "fine", "native")
        self.assertIsNone(lm4.frozen)

    def test_predict_run_records_knobs(self):
        prefix, _ = self.save_level2("l2run")
        _set_grid(prefix, cascade_value="current")
        proc = os.path.join(self.tmp, "proc")
        write_subjects(proc, 3, textured=True, seed=2)
        out = os.path.join(self.tmp, "out")
        args = CLI.parse_args(["run", "--checkpoint", prefix, "--processed-dir", proc, "--out", out,
                               "--branches", "deform", "--check-internal", "1"])
        code = PR.run_predict(args, configure_device=False)
        with open(os.path.join(out, "predict.json")) as f:
            rec = json.load(f)
        self.assertEqual(code, 0, rec.get("message"))
        self.assertEqual(rec["knobs"]["integration_grid"], {"checkpoint": "current", "used": "fine"})
        self.assertTrue(rec["internal_check"]["passed"])

    def test_cli_flag(self):
        ck = os.path.join(self.tmp, "run", "ckpt", "last")
        os.makedirs(ck)
        proc = os.path.join(self.tmp, "proc")
        os.makedirs(proc)
        common = ["--checkpoint", ck, "--processed-dir", proc]
        run = ["run"] + common + ["--out", os.path.join(self.tmp, "o")]
        chk = ["check-streaming"] + common + ["--out", os.path.join(self.tmp, "c.json")]
        lin = ["linear-eval"] + common + ["--task", "rigid", "--out", os.path.join(self.tmp, "le")]
        self.assertEqual(CLI.parse_args(run).integration_grid, "auto")
        self.assertEqual(CLI.parse_args(chk).integration_grid, "auto")
        for v in LD.INTEGRATION_GRIDS_CLI:
            self.assertEqual(CLI.parse_args(run + ["--integration-grid", v]).integration_grid, v)
            self.assertEqual(CLI.parse_args(chk + ["--integration-grid", v]).integration_grid, v)
        self.assertFalse(hasattr(CLI.parse_args(lin), "integration_grid"))
        for argv in (run + ["--integration-grid", "current"], lin + ["--integration-grid", "fine"]):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                CLI.parse_args(argv)
            self.assertEqual(cm.exception.code, 2, err.getvalue())


if __name__ == "__main__":
    unittest.main()
