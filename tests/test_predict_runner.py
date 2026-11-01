"""Tests of `inference.runner` (predict-script FEAT-003; design 3.4, 4.4, 4.5, AC 1, AC 2).

(a) TestStreamedVsUnchunked: `StreamedRunner` == `UnchunkedRunner` on random-weight TINY cascades,
level 4 (single stage) and level 2 (FrozenStage + trainable level 2), branches rigid, affine,
deform, affine+deform, chunks 1, 2 and N, float32 (S_stream) and a mixed_float16 trainable stage
(S_fp16); chunk boundaries per level. TestStreamingLevel1: FrozenChain + float32 level-1 stage
on /CPU:0 (TINY_L1 widths, N 3, chunks 1 and 2, deform and affine+deform).
(b) TestTrainerEquivalence: the runners' internal checkpoint-level fields and A vs the trainer's
validation forwards (`Trainer._val_deform_fn` / `_val_fn`, cascade dispatch) under op
determinism: unchunked within S_exact (fallback rule of AC 0), streamed chunk 1 and N within
S_stream; Dice via `Trainer._deform_errors` on both sides; the trainer's warp vs the runner's.
Level 2 on the GPU, level 1 on /CPU:0. Measured maxima, runtime and peak RSS are printed.
"""
import resource
import sys
import time
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory limit)
from tests.runtime_helpers import free_memory, synthetic_group, tiny_config
from tests.test_cascade import N_CLASSES, STAGE_KW, cascade_config, chain_config, randomize
from tests.test_predict_streaming import TINY_L1, _images
from tests.test_trainer import _OpDeterminism
import numpy as np
import tensorflow as tf

from data_loading import params as P
from inference.loader import LoadedModel
from inference.runner import BRANCHES, StreamedRunner, UnchunkedRunner, _pool_to
from inference.store import RamStore
from models import cascade as CS
from models import diffeo
from models.group_net import GroupRegLevelModel
from models.precision import policy_scope
from runtime import fixed_data as fd
from runtime import optimizers as O
from runtime.run import Trainer

N = 3
B_STREAM_FIELD, B_STREAM_A, B_STREAM_DICE, B_STREAM_WARP = 1e-4, 1e-5, 1e-4, 1e-4   # S_stream
B_FP16_MAX, B_FP16_MEAN, B_FP16_A = 2e-2, 1e-3, 1e-2                                # S_fp16
B_EXACT, B_EXACT_FALLBACK, B_EXACT_WARP, B_DICE_FALLBACK = 1e-6, 1e-5, 1e-5, 1e-6   # S_exact
FIELD_KEYS = ("U1", "U1_inv", "u_ckpt", "u_inv_ckpt")
CKPT_KEYS = ("u_ckpt", "u_inv_ckpt", "v", "v_raw")


def _full_res(x, level):
    """Level-`level` images -> level 1 by np.repeat (2x per axis and pooling step)."""
    k = 2 ** {1: 0, 2: 1, 4: 2}[level]
    return x.repeat(k, 1).repeat(k, 2).repeat(k, 3) if k > 1 else x


def _quantize(x):
    """Multiples of 2^-10: block sums of 8 equal values are exact, so pooling returns x."""
    return (np.round(np.asarray(x, np.float64) * 1024.0) / 1024.0).astype(np.float32)


def _collect(runner, x1, branches, keys=("A",) + FIELD_KEYS):
    n = x1.shape[0]
    got = {}

    def sink(i0, i1, res):
        for k in keys:
            if res.get(k) is None:
                continue
            a = res[k].numpy() if hasattr(res[k], "numpy") else np.asarray(res[k])
            got.setdefault(k, np.zeros((n,) + a.shape[1:], a.dtype))[i0:i1] = a
    summary = runner.run(x1, branches, sink)
    return got, summary


def _diff(a, b):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return float(d.max()), float(d.mean())


def _note(worst, key, value):
    worst[key] = max(worst.get(key, 0.0), float(value))


def _print(label, worst, extra=""):
    print("\n[runner] %s: %s%s" % (label, ", ".join("%s %.2e" % kv for kv in sorted(worst.items())),
                                   extra), file=sys.stderr)


def _level4(policy, seed=3):
    cfg = tiny_config(**STAGE_KW)
    with policy_scope(policy):
        m = GroupRegLevelModel(4, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
    randomize(m, seed, 0.02)
    return LoadedModel(4, m)


def _level2(policy, seed=3, cfg=None):
    cfg = cfg or tiny_config(**STAGE_KW)
    m4 = GroupRegLevelModel(4, cfg).build_model(2)
    randomize(m4, seed, 0.02)
    frozen = CS.FrozenStage.from_model(m4)
    with policy_scope(policy):
        m2 = GroupRegLevelModel(2, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
    m2.copy_weights_from(m4)
    randomize(m2, seed + 1, 0.01)
    return LoadedModel(2, m2, frozen)


def _level1(seed=3):
    """FrozenChain of TINY_L1 level-4 / level-2 models + a float32 level-1 copy (call inside the
    CPU scope)."""
    cfg = tiny_config(**dict(STAGE_KW, **TINY_L1))
    m4 = GroupRegLevelModel(4, cfg).build_model(2)
    randomize(m4, seed, 0.02)
    m2 = GroupRegLevelModel(2, cfg).build_model(2)
    m2.copy_weights_from(m4)
    randomize(m2, seed + 1, 0.01)
    chain = CS.FrozenChain.from_models(m4, m2)
    with policy_scope("float32"):
        m1 = GroupRegLevelModel(1, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
    m1.copy_weights_from(chain.model2)
    randomize(m1, seed + 2, 0.01)
    return LoadedModel(1, m1, chain)


def _expected_bounds(n, chunk):
    k = n if chunk <= 0 else min(chunk, n)
    return [(i0, min(i0 + k, n)) for i0 in range(0, n, k)]


class RunnerCase(unittest.TestCase):
    def tearDown(self):
        tf.keras.mixed_precision.set_global_policy("float32")
        free_memory()

    def compare_modes(self, label, loaded, x1, chunks, branches, fp16, worst, fields=FIELD_KEYS):
        """Streamed (every chunk size) vs unchunked for every branch; asserts and fills `worst`.
        One branch's reference is held at a time (level-1 host memory). Returns per branch the
        unchunked A, max |U1| and summary."""
        un = UnchunkedRunner(loaded)
        n = x1.shape[0]
        runners = {c: StreamedRunner(loaded, RamStore(), chunks={"l1": c, "l2": c, "l4": c})
                   for c in chunks}
        keys = ("A",) + tuple(fields)
        out = {}
        for br in branches:
            ref, ref_sum = _collect(un, x1, br, keys)
            out[br] = {"A": ref.get("A"), "summary": ref_sum,
                       "U1_max": float(np.abs(ref["U1"]).max()) if "U1" in ref else None}
            for chunk in chunks:
                sr = runners[chunk]
                st = sr.store
                got, summ = _collect(sr, x1, br, keys)
                lin, dfm = BRANCHES[br]
                self.assertEqual(sorted(got), sorted(ref), "%s %s" % (label, br))
                msg = "%s %s chunk %d" % (label, br, chunk)
                if lin is not None:
                    mx, _ = _diff(got["A"], ref["A"])
                    _note(worst, "A", mx)
                    self.assertLessEqual(mx, B_FP16_A if fp16 else B_STREAM_A, msg + " A")
                    np.testing.assert_array_equal(summ["A"], got["A"])
                if dfm is not None:
                    for key in fields:
                        mx, mean = _diff(got[key], ref[key])
                        _note(worst, key, mx)
                        _note(worst, key + "_mean", mean)
                        if fp16:
                            self.assertLessEqual(mx, B_FP16_MAX, "%s %s max" % (msg, key))
                            self.assertLessEqual(mean, B_FP16_MEAN, "%s %s mean" % (msg, key))
                        else:
                            self.assertLessEqual(mx, B_STREAM_FIELD, "%s %s" % (msg, key))
                    self.assertEqual(got["U1"].shape, (n,) + tuple(P.LEVEL_SHAPE[1]) + (3,))
                    if "u_ckpt" in got:
                        self.assertEqual(got["u_ckpt"].shape[1:4], tuple(P.LEVEL_SHAPE[loaded.level]))
                    self.assertGreater(summ["stats"]["max_abs_v"]["l%d" % loaded.level], 0.0)
                expect = _expected_bounds(n, chunk)
                for lv in loaded.levels:
                    for task in (t for t in (lin, dfm) if t is not None):
                        self.assertEqual(summ["chunks"]["l%d" % lv][task], expect,
                                         "%s level %d %s" % (msg, lv, task))
                self.assertEqual(summ["capped"], {})
                if fp16:
                    self.assertIn("l%d" % loaded.level, summ["fp16_max_abs"])
                else:
                    self.assertEqual(summ["fp16_max_abs"], {})
                self.assertEqual(st.nbytes(), 0, "%s: stores left %s" % (msg, st.names()))
                self.assertTrue(summ["timing_s"])
                if chunk == 2 and n == 3:
                    print("[runner] %s chunks %s" % (msg, summ["chunks"]), file=sys.stderr)
                del got
            del ref
        for sr in runners.values():
            sr.store.close()
        return out


class TestStreamedVsUnchunked(RunnerCase):
    """AC 1 (a) at levels 4 and 2: all four branches, chunks 1, 2, N."""

    def _run(self, level, policy):
        fp16 = policy == "mixed_float16"
        loaded = _level4(policy) if level == 4 else _level2(policy)
        x1 = _full_res(_images(2), 2)
        worst = {}
        out = self.compare_modes("L%d %s" % (level, policy), loaded, x1, (1, 2, N), tuple(BRANCHES),
                                 fp16, worst)
        # sanity: the cascade does something (non-identity A, non-zero fields)
        self.assertGreater(float(np.abs(out["affine"]["A"] - np.eye(4)[None]).max()), 1e-3)
        self.assertGreater(out["deform"]["U1_max"], 1e-3)
        un_sum = out["affine+deform"]["summary"]
        _print("L%d %s streamed vs unchunked" % (level, policy), worst,
               "; unchunked timing %s, gpu_peak_mb %s" % (
                   {k: round(v, 2) for k, v in un_sum["timing_s"].items()}, un_sum["gpu_peak_mb"]))

    def test_level4_float32(self):
        self._run(4, "float32")

    def test_level4_fp16(self):
        self._run(4, "mixed_float16")

    def test_level2_float32(self):
        self._run(2, "float32")

    def test_level2_fp16(self):
        self._run(2, "mixed_float16")


class TestStreamingLevel1(RunnerCase):
    """AC 1 level 1: FrozenChain + float32 level-1 stage on /CPU:0, N 3, chunks 1 and 2, deform
    and affine+deform. Never skipped for memory."""

    def test_level1_cpu(self):
        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("tf_gpu env requires a visible GPU (the level-1 case runs on /CPU:0)")
        t0 = time.time()
        worst = {}
        with tf.device("/CPU:0"):
            loaded = _level1()
            x1 = _images(1)
            # At level 1 u_ckpt / u_inv_ckpt are U1 / U1_inv (same tensors): compared once.
            out = self.compare_modes("L1 float32 CPU", loaded, x1, (1, 2),
                                     ("deform", "affine+deform"), False, worst,
                                     fields=("U1", "U1_inv"))
            self.assertGreater(out["deform"]["U1_max"], 1e-3)
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        _print("L1 float32 CPU streamed vs unchunked", worst,
               "; total %.0f s, peak RSS %.1f GB" % (time.time() - t0, rss_gb))


class TestTrainerEquivalence(RunnerCase):
    """AC 2 internal part: runner fields / A vs the trainer's validation forwards."""

    def _compare(self, label, tr, loaded, x_lvl, labels, xl_lvl, x1, xl1, chunks):
        """x_lvl / xl_lvl: deform / linear inputs at the checkpoint level (trainer side);
        x1 / xl1: the same at level 1 (runner side). Returns the worst diffs per mode."""
        n = x1.shape[0]
        report = {}
        with _OpDeterminism():
            ref_d = {k: v.numpy() for k, v in tr._val_deform_fn(n)(tf.constant(x_lvl)).items()}
            ref_a = {t: tr._val_fn(t, n)(tf.constant(xl_lvl))["A"].numpy() for t in ("rigid", "affine")}
            ref_err = tr._deform_errors(labels, ref_d["u"], ref_d["u_inv"], ref_d["v"], ref_d["v_raw"],
                                        seed=0)[0]
            ref_warp = diffeo.warp(tf.constant(x_lvl), tf.constant(ref_d["u"])).numpy()
            modes = [("unchunked", lambda: UnchunkedRunner(loaded))]
            for c in chunks:
                modes.append(("streamed chunk %d" % c, lambda c=c: StreamedRunner(
                    loaded, RamStore(), chunks={"l1": c, "l2": c, "l4": c})))
            for mode, make in modes:
                runner = make()
                exact = mode == "unchunked"
                w = {}
                got, _ = _collect(runner, x1, "deform", keys=CKPT_KEYS)
                for key, rkey in (("u_ckpt", "u"), ("u_inv_ckpt", "u_inv"), ("v", "v"), ("v_raw", "v_raw")):
                    _note(w, rkey, _diff(got[key], ref_d[rkey])[0])
                for t in ("rigid", "affine"):
                    _note(w, "A_" + t, _diff(runner.run(xl1, t)["A"], ref_a[t])[0])
                err = tr._deform_errors(labels, got["u_ckpt"], got["u_inv_ckpt"], got["v"], got["v_raw"],
                                        seed=0)[0]
                dice_same = True
                for key in ("template_dice", "pair_dice"):
                    dice_same = dice_same and np.array_equal(err[key], ref_err[key])
                    _note(w, key, _diff(err[key], ref_err[key])[0])
                warp = diffeo.warp(tf.constant(x_lvl), tf.constant(got["u_ckpt"])).numpy()
                _note(w, "warp", _diff(warp, ref_warp)[0])
                fields = max(w["u"], w["u_inv"], w["A_rigid"], w["A_affine"])
                if exact:
                    tag = "exact" if fields <= B_EXACT else "FALLBACK 1e-5"
                    self.assertLessEqual(fields, B_EXACT_FALLBACK, "%s %s %s" % (label, mode, w))
                    self.assertLessEqual(w["warp"], B_EXACT_WARP, "%s %s warp" % (label, mode))
                    if not dice_same:
                        tag += ", Dice FALLBACK 1e-6"
                        self.assertLessEqual(max(w["template_dice"], w["pair_dice"]), B_DICE_FALLBACK)
                else:
                    tag = "S_stream"
                    self.assertLessEqual(max(w["u"], w["u_inv"]), B_STREAM_FIELD, "%s %s" % (label, mode))
                    self.assertLessEqual(max(w["A_rigid"], w["A_affine"]), B_STREAM_A, "%s %s" % (label, mode))
                    self.assertLessEqual(max(w["template_dice"], w["pair_dice"]), B_STREAM_DICE)
                    self.assertLessEqual(w["warp"], B_STREAM_WARP, "%s %s warp" % (label, mode))
                _print("%s trainer vs %s (%s, Dice identical %s)" % (label, mode, tag, dice_same), w)
                report[mode] = w
                del got, err, warp, runner
        print("[runner] %s trainer Dice: template %s, pair mean %.4f" % (
            label, np.round(ref_err["template_dice"], 4), float(np.mean(ref_err["pair_dice"]))),
            file=sys.stderr)
        return report

    def test_level2(self):
        loaded = _level2("float32")
        dgroup = fd.synthetic_deform_group(2, N, seed=0)
        groups = {t: synthetic_group(2, N, t, seed=i) for i, t in enumerate(("rigid", "affine"))}
        sources = {t: fd.FixedGroupSource(groups[t]["batch"]) for t in ("rigid", "affine")}
        sources["deform"] = fd.FixedGroupSource(dgroup["batch"])
        cfg = cascade_config(group_size=N, deform_group_size=N, lean_sampler=True, n_classes=N_CLASSES)
        tr = Trainer(loaded.model, O.make_optimizer("adam", lr=1e-3), sources, groups["rigid"]["S"], cfg,
                     cascade=loaded.frozen)
        x2 = _quantize(dgroup["images"])
        xl2 = _quantize(groups["rigid"]["images"])
        x1, xl1 = _full_res(x2, 2), _full_res(xl2, 2)
        np.testing.assert_array_equal(_pool_to(tf.constant(x1), 2).numpy(), x2)
        np.testing.assert_array_equal(_pool_to(tf.constant(xl1), 2).numpy(), xl2)
        self._compare("L2", tr, loaded, x2, dgroup["labels"], xl2, x1, xl1, (1, N))

    def test_level1_cpu(self):
        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("tf_gpu env requires a visible GPU (the level-1 case runs on /CPU:0)")
        t0 = time.time()
        with tf.device("/CPU:0"):
            loaded = _level1()
            dgroup = fd.synthetic_deform_group(1, N, seed=0)
            for k in ("u_true", "v_true"):              # level-1 host memory: not needed
                dgroup.pop(k)
            lgroup = synthetic_group(1, N, "rigid", seed=0)
            sources = {"rigid": fd.FixedGroupSource(lgroup["batch"]),
                       "deform": fd.FixedGroupSource(dgroup["batch"])}
            cfg = chain_config(amp=False, val_pair_chunk=1, n_classes=N_CLASSES, group_size=N,
                               deform_group_size=N)
            tr = Trainer(loaded.model, O.make_optimizer("adam", lr=1e-3, amp=False), sources,
                         lgroup["S"], cfg, cascade=loaded.frozen)
            x1, xl1 = dgroup["images"], lgroup["images"]
            self._compare("L1 CPU", tr, loaded, x1, dgroup["labels"], xl1, x1, xl1, (1, N))
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print("[runner] L1 trainer equivalence on CPU: total %.0f s, peak RSS %.1f GB"
              % (time.time() - t0, rss_gb), file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
