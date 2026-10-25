"""Unit tests of tests/gpu/check_mixed_run.py (stdlib only, no TF) on fabricated run directories:
rigid+affine (milestone-5 behaviour unchanged), deform-only and rigid+affine+deform (milestone 6).
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "check_mixed_run.py")
_spec = importlib.util.spec_from_file_location("check_mixed_run", _PATH)
cmr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmr)

STEPS = 20


def linear_val(task, pair):
    d = {"val_%s/center_rms_mm" % task: 1.0, "val_%s/pair_rms_mm" % task: pair}
    if task == "affine":
        d.update({"val_affine/" + k: 0.5 for k in cmr.AFFINE_KEYS})
    return d


def deform_val():
    return {"val_deform/template_dice": 0.8, "val_deform/pair_dice": 0.7, "val_deform/neg_jac_pct": 0.0,
            "val_deform/inv_consistency_mm": 0.05, "val_deform/centrality_mm": 1e-7,
            "val_deform/pair_dice_loss": 0.3}


def cascade_config():
    """config.json blocks of a milestone-8 cascade run (design §2.2 / §9)."""
    return {"init_weights": {"mode": "full", "via": "cascade", "restored_parts": {"encoder": 30},
                             "fresh_parts": {}, "config_diff": {}},
            "cascade": {"source_checkpoint": "/x/ckpt/best/best-11000", "source_config": "/x/config.json",
                        "source_level": 4, "source_step": 11000,
                        "fingerprint": {"n_vars": 120, "abs_sum": 1.5, "sq_sum": 0.5},
                        "S_rel_diff_vs_level4": None, "skip_prob": 0.0}}


def cascade_val(tasks, drop=()):
    """Final-step frozen-stage baselines (composed values: linear_val / deform_val)."""
    d = {}
    for t in tasks:
        if t == "deform":
            d.update({"val_deform_frozen/template_dice": 0.75, "val_deform_frozen/pair_dice": 0.72})
        else:
            d.update({"val_%s_frozen/center_rms_mm" % t: 2.0, "val_%s_frozen/pair_rms_mm" % t: 5.0})
    for k in [k for k in d if k.split("/")[0] in drop]:
        del d[k]
    return d


BN_EVENT = {"event": "bn_recalibration", "num_batches": 8, "planned": {"rigid": 2, "affine": 2, "deform": 4},
            "batches": {"rigid": 2, "affine": 2, "deform": 4}, "skipped_nonfinite": {"rigid": 0, "affine": 0, "deform": 0},
            "n_layers_updated": 20, "layers_without_updates": [], "elapsed_s": 3.0, "freeze": True,
            "gap": {"gap_pre/rigid/center_rms_mm": 0.35, "gap_post/rigid/center_rms_mm": 0.001,
                    "gap_pre/deform/template_dice": 0.01, "gap_post/deform/template_dice": 0.002}}


def make_run(root, tasks, val_overrides=None, drop_pngs=(), best_dirs=None, trace=None, init_weights=None,
             enc_grad=False, enc_grad_bad=None, grad_norm_keys=False, steps=STEPS, every=5, eff=None,
             config=None, events=(), train_extra=None):
    """Fabricated run dir: metrics.jsonl with train/val records, fit_summary.json, PNGs, ckpts;
    optional config.json init_weights record and train_<task>/diag_enc_grad_norm_* keys
    (``enc_grad_bad``: a task whose _eff values are NaN); ``grad_norm_keys``: the v3.10
    --task-grad-norm keys train_<task>/grad_norm_scale 0.5 and grad_norm_gated_frac 0.
    Milestone 8: train records at every..steps (window ends), ``eff(task, step)`` overrides the
    _eff value, ``config`` extra config.json blocks, ``events`` records logged at step 0,
    ``train_extra`` extra keys of every train record."""
    os.makedirs(os.path.join(root, "logs"))
    os.makedirs(os.path.join(root, "val_png"))
    recs = [{"step": 0, "data": dict(ev)} for ev in events]
    for step in range(every, steps + 1, every):
        data = {"train_%s/loss" % t: 1.0 / step for t in tasks}
        if enc_grad:
            for t in tasks:
                e = float("nan") if t == enc_grad_bad else (2.0 if t == "deform" else 1.0)
                if eff is not None:
                    e = eff(t, step)
                data.update({"train_%s/diag_enc_grad_norm_raw" % t: 10.0, "train_%s/diag_enc_grad_norm_eff" % t: e,
                             "train_%s/grad_norm" % t: 5.0})
        if grad_norm_keys:
            for t in tasks:
                data.update({"train_%s/grad_norm_scale" % t: 0.5, "train_%s/grad_norm_gated_frac" % t: 0.0})
        data.update(train_extra or {})
        recs.append({"step": step, "data": data})
    if init_weights is not None or config is not None:
        cfg = dict(config or {})
        if init_weights is not None:
            cfg["init_weights"] = init_weights
        with open(os.path.join(root, "config.json"), "w") as f:
            json.dump(cfg, f)
    val = {}
    pairs = []
    for t in tasks:
        if t == "deform":
            val.update(deform_val())
        else:
            p = {"rigid": 6.0, "affine": 7.0}[t]
            pairs.append(p)
            val.update(linear_val(t, p))
    if len(pairs) >= 2:
        val["val/mean_pair_rms_mm"] = sum(pairs) / len(pairs)
    val.update(val_overrides or {})
    recs.append({"step": steps, "data": val})
    with open(os.path.join(root, "logs", "metrics.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(root, "fit_summary.json"), "w") as f:
        json.dump({"trace_counts": trace or {t: 1 for t in tasks}, "nonfinite_total": 0}, f)
    for t in tasks:
        if t not in drop_pngs:
            open(os.path.join(root, "val_png", "%s_step%07d.png" % (t, steps)), "wb").close()
    if best_dirs is None:
        best_dirs = {"best": "best"}
        if "deform" in tasks and any(t != "deform" for t in tasks):
            best_dirs["best_deform"] = "best_deform"
    for sub, prefix in best_dirs.items():
        d = os.path.join(root, "ckpt", sub)
        os.makedirs(d)
        open(os.path.join(d, "%s-2.index" % prefix), "w").close()
    return root


class CheckMixedRunCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_check_mixed_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_dir(self, name, tasks, **kw):
        return make_run(os.path.join(self.tmp, name), tasks, **kw)

    def failed(self, res):
        return [n for n, p, _ in res if not p]


class TestLinear(CheckMixedRunCase):
    def test_rigid_affine_pass(self):
        ok, res = cmr.check(self.run_dir("ra", ["rigid", "affine"]), ["rigid", "affine"], STEPS)
        self.assertTrue(ok, self.failed(res))
        self.assertEqual([n for n, _, _ in res],
                         ["final_step", "train_rigid", "train_affine", "val_rigid", "val_affine",
                          "val_mean_pair_rms_mm", "trace_once", "finite_steps", "val_png", "best_checkpoint"])

    def test_rigid_affine_wrong_mean_fails(self):
        d = self.run_dir("ra", ["rigid", "affine"], val_overrides={"val/mean_pair_rms_mm": 1.0})
        ok, res = cmr.check(d, ["rigid", "affine"], STEPS)
        self.assertFalse(ok)
        self.assertEqual(self.failed(res), ["val_mean_pair_rms_mm"])

    def test_missing_affine_metric_fails(self):
        d = self.run_dir("ra", ["rigid", "affine"], val_overrides={"val_affine/pair_shear_pct": float("nan")})
        ok, res = cmr.check(d, ["rigid", "affine"], STEPS)
        self.assertEqual(self.failed(res), ["val_affine"])

    def test_main_exit_code(self):
        d = self.run_dir("ra", ["rigid", "affine"])
        self.assertEqual(cmr.main([d, "--tasks", "rigid,affine", "--steps", str(STEPS)]), 0)
        self.assertEqual(cmr.main([d, "--tasks", "rigid,affine", "--steps", str(STEPS + 10)]), 1)


class TestDeform(CheckMixedRunCase):
    def test_deform_only_pass(self):
        ok, res = cmr.check(self.run_dir("d", ["deform"]), ["deform"], STEPS)
        self.assertTrue(ok, self.failed(res))
        names = [n for n, _, _ in res]
        self.assertIn("val_deform", names)
        self.assertNotIn("best_deform_checkpoint", names)    # deform-only: ckpt/best only

    def test_mixed3_pass(self):
        tasks = ["rigid", "affine", "deform"]
        ok, res = cmr.check(self.run_dir("m3", tasks), tasks, STEPS)
        self.assertTrue(ok, self.failed(res))
        self.assertIn("best_deform_checkpoint", [n for n, _, _ in res])
        self.assertEqual(cmr.main([os.path.join(self.tmp, "m3"), "--tasks", ",".join(tasks)]), 0)

    def test_mixed3_missing_best_deform_fails(self):
        tasks = ["rigid", "affine", "deform"]
        d = self.run_dir("m3", tasks, best_dirs={"best": "best"})
        ok, res = cmr.check(d, tasks, STEPS)
        self.assertFalse(ok)
        self.assertEqual(self.failed(res), ["best_deform_checkpoint"])

    def test_deform_nonfinite_val_and_missing_png_fail(self):
        d = self.run_dir("d", ["deform"], val_overrides={"val_deform/neg_jac_pct": float("nan")},
                         drop_pngs=("deform",))
        ok, res = cmr.check(d, ["deform"], STEPS)
        self.assertFalse(ok)
        self.assertEqual(self.failed(res), ["val_deform", "val_png"])

    def test_rigid_deform_skips_mean(self):
        tasks = ["rigid", "deform"]
        ok, res = cmr.check(self.run_dir("rd", tasks), tasks, STEPS)
        self.assertTrue(ok, self.failed(res))
        detail = dict((n, d) for n, _, d in res)["val_mean_pair_rms_mm"]
        self.assertIn("skipped", detail)

    def test_init_weights_and_enc_grad(self):
        # Milestone 7: optional warm-start and encoder grad-norm checks (default off = old behaviour).
        tasks = ["rigid", "affine", "deform"]
        partial = {"mode": "partial", "restored_parts": {"encoder": 30, "linear_decoder": 20, "centering": 0},
                   "fresh_parts": {"deform_decoder": 40}, "config_diff": {"deform_decoder": [False, True]}}
        d = self.run_dir("m7", tasks, init_weights=partial, enc_grad=True)
        ok, res = cmr.check(d, tasks, STEPS, expect_init_weights="partial", expect_enc_grad=True)
        self.assertTrue(ok, self.failed(res))
        names = [n for n, _, _ in res]
        self.assertEqual(names[-5:], ["init_weights", "enc_grad_rigid", "enc_grad_affine", "enc_grad_deform",
                                      "enc_grad_ratio"])
        self.assertIn("rigid 0.5", dict((n, x) for n, _, x in res)["enc_grad_ratio"])
        self.assertEqual(cmr.main([d, "--tasks", ",".join(tasks), "--expect-init-weights", "partial",
                                   "--expect-enc-grad", "true"]), 0)
        self.assertEqual(cmr.main([d, "--tasks", ",".join(tasks), "--expect-init-weights", "full"]), 1)
        # defaults: neither check runs
        self.assertNotIn("init_weights", [n for n, _, _ in cmr.check(d, tasks, STEPS)[1]])
        # no record / wrong fresh parts / missing or NaN diagnostics fail
        d2 = self.run_dir("m7_none", tasks)
        ok, res = cmr.check(d2, tasks, STEPS, expect_init_weights="partial", expect_enc_grad=True)
        self.assertEqual(self.failed(res), ["init_weights", "enc_grad_rigid", "enc_grad_affine", "enc_grad_deform"])
        d3 = self.run_dir("m7_bad", tasks, init_weights=dict(partial, fresh_parts={}), enc_grad=True,
                          enc_grad_bad="affine")
        ok, res = cmr.check(d3, tasks, STEPS, expect_init_weights="partial", expect_enc_grad=True)
        self.assertEqual(self.failed(res), ["init_weights", "enc_grad_affine"])
        # linear-only partial warm start (deform source): fresh parts must be empty
        lin = ["rigid", "affine"]
        d4 = self.run_dir("lin", lin, init_weights=dict(partial, fresh_parts={}), enc_grad=True)
        ok, res = cmr.check(d4, lin, STEPS, expect_init_weights="partial", expect_enc_grad=True)
        self.assertTrue(ok, self.failed(res))
        self.assertNotIn("enc_grad_ratio", [n for n, _, _ in res])
        self.assertNotIn("grad_norm_scale", dict((n, x) for n, _, x in res)["enc_grad_rigid"])
        # v3.10 --task-grad-norm keys: info appended to the detail, same result names
        d5 = self.run_dir("gn", tasks, init_weights=partial, enc_grad=True, grad_norm_keys=True)
        ok, res5 = cmr.check(d5, tasks, STEPS, expect_init_weights="partial", expect_enc_grad=True)
        self.assertTrue(ok, self.failed(res5))
        self.assertEqual([n for n, _, _ in res5], names)
        for t in tasks:
            self.assertIn("grad_norm_scale median 0.5, gated frac 0", dict((n, x) for n, _, x in res5)["enc_grad_" + t])

    def test_retrace_fails(self):
        tasks = ["rigid", "affine", "deform"]
        d = self.run_dir("m3", tasks, trace={"rigid": 1, "affine": 1, "deform": 2})
        ok, res = cmr.check(d, tasks, STEPS)
        self.assertEqual(self.failed(res), ["trace_once"])


TASKS3 = ["rigid", "affine", "deform"]
TIMING = {"train_%s/%s" % (t, k): v for t in TASKS3 for k, v in (("data_ms", 10.0), ("compute_ms", 90.0))}


class TestCascade(CheckMixedRunCase):
    """T12 (milestone 8, design §13): cascade / bn-recal / skip / grad-balance options."""

    def cascade_run(self, name, drop=(), events=(BN_EVENT,), **kw):
        kw.setdefault("val_overrides", cascade_val(TASKS3, drop))
        kw.setdefault("train_extra", TIMING)
        return self.run_dir(name, TASKS3, config=cascade_config(), events=events, **kw)

    def details(self, res):
        return dict((n, x) for n, _, x in res)

    def test_cascade_pass(self):
        d = self.cascade_run("c", enc_grad=True)
        ok, res = cmr.check(d, TASKS3, STEPS, expect_init_weights="full", expect_enc_grad=True,
                            expect_cascade=True, expect_bn_recal=True, grad_balance_rule=True,
                            grad_balance_from_step=5)
        self.assertTrue(ok, self.failed(res))
        x = self.details(res)
        for n in ("cascade_config", "val_rigid_frozen", "val_affine_frozen", "val_deform_frozen", "bn_recalibration"):
            self.assertIn(n, x)
        # composed rigid center 1.0 < frozen 2.0 (better), deform template 0.8 > 0.75 (better), pair 0.7 < 0.72
        self.assertIn("center_rms_mm composed 1 vs frozen 2 (better)", x["cascade_gain_rigid"])
        self.assertIn("pair_rms_mm composed 6 vs frozen 5 (worse)", x["cascade_gain_rigid"])
        self.assertIn("template_dice composed 0.8 vs frozen 0.75 (better)", x["cascade_gain_deform"])
        self.assertIn("pair_dice composed 0.7 vs frozen 0.72 (worse)", x["cascade_gain_deform"])
        self.assertTrue(x["cascade_gain_deform"].startswith("INFO cascade_gain deform"))
        self.assertIn("fraction 0.100", x["data_wait_rigid"])
        self.assertIn("no rigid_g validation set", x["g_gap_rigid"])
        self.assertIn("center_rms_mm pre 0.35 post 0.001", x["bn_gap_rigid"])
        self.assertIn("template_dice pre 0.01 post 0.002", x["bn_gap_deform"])
        # default eff: linear 1.0, deform 2.0 -> ratios 0.5 -> KEEP
        self.assertIn("grad_balance KEEP task-grad-norm", x["grad_balance"])
        self.assertEqual(cmr.main([d, "--tasks", ",".join(TASKS3), "--expect-init-weights", "full",
                                   "--expect-cascade", "true", "--expect-bn-recal", "true",
                                   "--grad-balance-rule", "true", "--grad-balance-from-step", "5"]), 0)
        # defaults off: no milestone-8 result names (level-4 behaviour unchanged)
        names = [n for n, _, _ in cmr.check(d, TASKS3, STEPS)[1]]
        self.assertFalse([n for n in names if n.startswith(("cascade", "val_rigid_frozen", "bn_", "grad_balance",
                                                             "g_gap", "data_wait"))])

    def test_cascade_wrong_provenance_fails(self):
        d = self.cascade_run("c")
        cfg = cascade_config()
        cfg["cascade"]["source_level"] = 2
        cfg["init_weights"].pop("via")
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f)
        ok, res = cmr.check(d, TASKS3, STEPS, expect_cascade=True)
        self.assertEqual(self.failed(res), ["cascade_config"])
        self.assertIn("source_level 2", self.details(res)["cascade_config"])

    def test_missing_val_rigid_frozen_fails(self):
        d = self.cascade_run("c", drop=("val_rigid_frozen",))
        ok, res = cmr.check(d, TASKS3, STEPS, expect_cascade=True)
        self.assertFalse(ok)
        self.assertEqual(self.failed(res), ["val_rigid_frozen"])
        self.assertNotIn("cascade_gain_rigid", self.details(res))

    def test_g_gap_lines(self):
        g = {"val_rigid_g/center_rms_mm": 1.5, "val_rigid_g/pair_rms_mm": 7.0, "val_rigid_g/center_rot_deg": 2.0,
             "val_rigid/center_rot_deg": 1.0, "val_rigid_g_frozen/center_rms_mm": 3.0,
             "val_rigid_g_frozen/pair_rms_mm": 8.0, "val_affine_g/pair_shear_pct": 5.3,
             "val_affine/pair_shear_pct": 4.0}
        over = dict(cascade_val(TASKS3), **g)
        d = self.cascade_run("c", val_overrides=over)
        ok, res = cmr.check(d, TASKS3, STEPS, expect_cascade=True)
        self.assertTrue(ok, self.failed(res))
        x = self.details(res)
        self.assertIn("center_rms_mm 1 -> 1.5", x["g_gap_rigid"])
        self.assertIn("center_rot_deg 1 -> 2", x["g_gap_rigid"])
        self.assertIn("center_rms_mm 2 -> 3", x["g_gap_rigid_frozen"])
        self.assertIn("pair_shear_pct 4 -> 5.3", x["g_gap_affine"])
        self.assertIn("no affine_g validation set", x["g_gap_affine_frozen"])
        self.assertIn("cascade_gain_rigid_g", x)

    def test_bn_recal_event_missing_fails(self):
        d = self.cascade_run("c", events=())
        ok, res = cmr.check(d, TASKS3, STEPS, expect_bn_recal=True)
        self.assertEqual(self.failed(res), ["bn_recalibration"])
        stale = dict(BN_EVENT, layers_without_updates=["encoder/l2/bn"])
        d2 = self.cascade_run("c2", events=(stale,))
        self.assertEqual(self.failed(cmr.check(d2, TASKS3, STEPS, expect_bn_recal=True)[1]), ["bn_recalibration"])

    def test_bn_recal_empty_gap_passes(self):
        d = self.cascade_run("c", events=(dict(BN_EVENT, gap={}),))
        ok, res = cmr.check(d, TASKS3, STEPS, expect_bn_recal=True)
        self.assertTrue(ok, self.failed(res))
        self.assertFalse([n for n, _, _ in res if n.startswith("bn_gap")])

    def test_skip(self):
        d = self.cascade_run("c", train_extra=dict(TIMING, **{cmr.SKIP_KEY: 0.25}))
        ok, res = cmr.check(d, TASKS3, STEPS, expect_skip=True)
        self.assertTrue(ok, self.failed(res))
        d2 = self.cascade_run("c2")
        self.assertEqual(self.failed(cmr.check(d2, TASKS3, STEPS, expect_skip=True)[1]), ["cascade_skip"])

    def verdict(self, eff, steps=STEPS, every=5, from_step=None):
        d = self.cascade_run("gb_%d" % len(os.listdir(self.tmp)), enc_grad=True, eff=eff, steps=steps,
                             every=every)
        kw = {} if from_step is None else {"grad_balance_from_step": from_step}
        ok, res = cmr.check(d, TASKS3, steps, grad_balance_rule=True, **kw)
        self.assertTrue(ok, self.failed(res))                 # never a failure
        self.assertIn("enc_grad_rigid", self.details(res))   # implies the enc-grad checks
        return self.details(res)["grad_balance"]

    def test_grad_balance_keep_fallback_undecided(self):
        ratios = {"rigid": 1.3, "affine": 0.7, "deform": 1.0}
        x = self.verdict(lambda t, s: ratios[t], steps=75, every=25)
        self.assertIn("grad_balance KEEP task-grad-norm", x)
        self.assertIn("rigid 1.3, affine 0.7", x)
        x = self.verdict(lambda t, s: {"rigid": 40.0, "affine": 0.7, "deform": 1.0}[t], steps=75, every=25)
        self.assertIn("grad_balance FALLBACK deform:64", x)
        self.assertIn("rigid 40", x)
        # no record at or after the cut -> values missing -> UNDECIDED (and still no failure)
        x = self.verdict(lambda t, s: ratios[t], steps=75, every=25, from_step=100)
        self.assertIn("grad_balance UNDECIDED (treat as FALLBACK and inspect the run)", x)
        self.assertIn("['deform', 'rigid', 'affine']", x)

    def test_grad_balance_step_cut(self):
        # Extreme value only at the window ending at step 25: excluded (>= 50); step 50 included.
        eff = lambda t, s: 1000.0 if (t == "rigid" and s == 25) else {"rigid": 1.3, "affine": 0.7, "deform": 1.0}[t]
        x = self.verdict(eff, steps=75, every=25)
        self.assertIn("grad_balance KEEP task-grad-norm", x)
        self.assertIn("rigid 1.3", x)
        # step 50 included: an extreme value there flips the median of (50, 75) records
        eff50 = lambda t, s: 1000.0 if (t == "rigid" and s == 50) else {"rigid": 1.3, "affine": 0.7, "deform": 1.0}[t]
        self.assertIn("grad_balance FALLBACK deform:64", self.verdict(eff50, steps=75, every=25))
        # --grad-balance-from-step 25 includes the step-25 window
        x = self.verdict(eff, steps=75, every=25, from_step=25)
        self.assertIn("rigid 1.3", x)                          # median of (1000, 1.3, 1.3)
        x = self.verdict(eff, steps=50, every=25, from_step=25)
        self.assertIn("grad_balance FALLBACK deform:64", x)    # median of (1000, 1.3) = 500.65

    def test_grad_balance_function(self):
        recs = [{"step": 50, "data": {"train_rigid/diag_enc_grad_norm_eff": 2.0}}]
        self.assertEqual(cmr.grad_balance(recs, ["rigid"])[0], "UNDECIDED")      # no deform task
        recs.append({"step": 50, "data": {"train_deform/diag_enc_grad_norm_eff": 1.0}})
        verdict, ratios, _ = cmr.grad_balance(recs, ["rigid", "deform"])
        self.assertEqual((verdict, ratios), ("KEEP", {"rigid": 2.0}))
        self.assertEqual(cmr.grad_balance(recs, ["rigid", "deform"], from_step=51)[0], "UNDECIDED")

    def test_grad_balance_deform64_label(self):
        # Clip 12 + deform:64 run (config.json args.task_grad_norm 0): KEEP keeps the current balance.
        cfg = dict(cascade_config(), args={"task_grad_norm": 0.0})
        d = self.run_dir("d64", TASKS3, config=cfg, events=(BN_EVENT,), enc_grad=True,
                         val_overrides=cascade_val(TASKS3), train_extra=TIMING)
        ok, res = cmr.check(d, TASKS3, STEPS, grad_balance_rule=True, grad_balance_from_step=5)
        self.assertTrue(ok, self.failed(res))
        self.assertIn("grad_balance KEEP current balance (deform64)", self.details(res)["grad_balance"])
        cfg["args"]["task_grad_norm"] = 12.0
        d2 = self.run_dir("gn12", TASKS3, config=cfg, enc_grad=True, val_overrides=cascade_val(TASKS3))
        x = self.details(cmr.check(d2, TASKS3, STEPS, grad_balance_rule=True, grad_balance_from_step=5)[1])
        self.assertIn("grad_balance KEEP task-grad-norm", x["grad_balance"])


# Milestone 9 phase 2 (design §9): level-1 3-stage cascade fixtures.
TASKS_RD = ["rigid", "deform"]


def chain_config(min_pairs="n", n=4, n_deform=2):
    cfg = cascade_config()
    cfg["init_weights"]["cross_level"] = [2, 1]
    cfg["cascade"].update({"source_level": 2, "source_step": 6000, "S_rel_diff_vs_level2": 0.01,
                           "skip_stage": 2, "stages": [
                               {"level": 4, "checkpoint_key": "frozen_stage", "origin_step": 11000,
                                "fingerprint": {"n_vars": 120, "abs_sum": 1.0, "sq_sum": 1.0}},
                               {"level": 2, "checkpoint_key": "model", "origin_step": 6000,
                                "fingerprint": {"n_vars": 120, "abs_sum": 2.0, "sq_sum": 2.0}}]})
    cfg["args"] = {"min_pairs": min_pairs, "group_size": n, "deform_group_size": n_deform, "p_pair": 0.1,
                   "p_pdice": 0.1, "task_ratio_parsed": {"rigid": 1, "deform": 1}, "val_every": 10,
                   "task_grad_norm": 12.0}
    return cfg


def chain_val(tasks, drop=()):
    d = cascade_val(tasks)
    for t in tasks:
        if t == "deform":
            d.update({"val_deform_frozen4/template_dice": 0.7, "val_deform_frozen4/pair_dice": 0.6,
                      "val_deform/ncc_template": 0.5, "val_deform/ncc_template_identity": 0.3})
        else:
            d.update({"val_%s_frozen4/center_rms_mm" % t: 3.0, "val_%s_frozen4/pair_rms_mm" % t: 4.0})
    for k in [k for k in d if k.split("/")[0] in drop]:
        del d[k]
    d[cmr.VAL_WALL_KEY] = 30.0
    return d


class TestLevel1Chain(CheckMixedRunCase):
    """Milestone 9 phase 2: --expect-cascade-stages 2, --expect-min-pairs, --memory-budget-mb,
    --project-steps."""

    def chain_run(self, name, cfg=None, drop=(), n_pairs=None, **kw):
        extra = dict(TIMING)
        extra.update({"train_rigid/n_pairs": 4.0 if n_pairs is None else n_pairs,      # N 4 -> 4

                      "train_deform/n_pairs": 2.0, "gpu_peak_mb": 30000.0})
        kw.setdefault("val_overrides", chain_val(TASKS_RD, drop))
        return self.run_dir(name, TASKS_RD, config=cfg or chain_config(), events=(BN_EVENT,),
                            train_extra=extra, **kw)

    def details(self, res):
        return dict((n, x) for n, _, x in res)

    def test_chain_pass_and_info_lines(self):
        d = self.chain_run("c")
        ok, res = cmr.check(d, TASKS_RD, STEPS, expect_cascade=True, expect_cascade_stages=2,
                            expect_min_pairs="n", memory_budget_mb=36864, project_steps=8000)
        self.assertTrue(ok, self.failed(res))
        x = self.details(res)
        for n in ("cascade_stages", "val_rigid_frozen4", "val_deform_frozen4", "min_pairs_config",
                  "n_pairs_rigid", "n_pairs_deform", "deform_extra", "gpu_peak", "step_time"):
            self.assertIn(n, x)
        self.assertIn("vs level 2 0.01", x["cascade_config"])
        self.assertIn("center_rms_mm composed 1 vs frozen 2 (better) vs frozen4 3 (better)", x["cascade_gain_rigid"])
        self.assertIn("pair_dice composed 0.7 vs frozen 0.72 (worse) vs frozen4 0.6 (better)", x["cascade_gain_deform"])
        self.assertIn("ncc_template 0.5 / n/a / n/a / 0.3", x["deform_extra"])
        self.assertIn("neg_jac_pct 0 / n/a / n/a / n/a", x["deform_extra"])
        self.assertIn("29.30 GiB) vs A100 budget 36 GiB: fits", x["gpu_peak"])
        # rigid / deform 0.1 s each (TIMING 10 + 90 ms), one val/wall_s of 30 s, val_every 10,
        # recal 3 s: 1.3 x (8000 x 0.1 + 800 x 30 + 3) = 32243.9 s
        self.assertIn("ratio-weighted mean step 0.1 s", x["step_time"])
        self.assertIn("= 32244 s", x["step_time"])
        self.assertEqual(cmr.main([d, "--tasks", "rigid,deform", "--expect-cascade", "true",
                                   "--expect-cascade-stages", "2", "--expect-min-pairs", "n",
                                   "--memory-budget-mb", "36864", "--project-steps", "8000"]), 0)
        # gpu_peak over budget: still INFO (passes)
        x = self.details(cmr.check(d, TASKS_RD, STEPS, memory_budget_mb=20000)[1])
        self.assertIn("exceeds -> keep rtx8000 or --lean-losses true", x["gpu_peak"])
        # defaults: no milestone-9 result names
        names = [n for n, _, _ in cmr.check(d, TASKS_RD, STEPS, expect_cascade=True)[1]]
        self.assertFalse([n for n in names if n in ("cascade_stages", "val_rigid_frozen4", "deform_extra",
                                                    "min_pairs_config", "gpu_peak", "step_time")])

    def test_missing_frozen4_fails(self):
        d = self.chain_run("c", drop=("val_rigid_frozen4",))
        ok, res = cmr.check(d, TASKS_RD, STEPS, expect_cascade_stages=2)
        self.assertFalse(ok)
        self.assertEqual(self.failed(res), ["val_rigid_frozen4"])

    def test_wrong_stages_fails(self):
        cfg = chain_config()
        cfg["cascade"]["stages"] = cfg["cascade"]["stages"][::-1]
        ok, res = cmr.check(self.chain_run("c", cfg=cfg), TASKS_RD, STEPS, expect_cascade_stages=2)
        self.assertEqual(self.failed(res), ["cascade_stages"])
        self.assertIn("stages levels [2, 4]", self.details(res)["cascade_stages"])
        cfg = chain_config()
        cfg["init_weights"]["cross_level"] = [4, 2]
        cfg["cascade"]["stages"][0]["origin_step"] = None
        ok, res = cmr.check(self.chain_run("c2", cfg=cfg), TASKS_RD, STEPS, expect_cascade_stages=2)
        self.assertEqual(self.failed(res), ["cascade_stages"])
        # a milestone-8 (level-2) record checked as a chain: source level 4 != 2, no stages
        ok, res = cmr.check(self.chain_run("c3", cfg=dict(cascade_config(), args=chain_config()["args"])),
                            TASKS_RD, STEPS, expect_cascade_stages=2)
        self.assertEqual(self.failed(res), ["cascade_config", "cascade_stages"])

    def test_min_pairs(self):
        good = self.chain_run("c", n_pairs=4.0)              # expected_pairs(4, 4, 0.1) == 4
        ok, res = cmr.check(good, TASKS_RD, STEPS, expect_min_pairs="n")
        self.assertTrue(ok, self.failed(res))
        wrong = self.chain_run("c2", n_pairs=3.0)
        ok, res = cmr.check(wrong, TASKS_RD, STEPS, expect_min_pairs="n")
        self.assertEqual(self.failed(res), ["n_pairs_rigid"])
        self.assertIn("expected 4", self.details(res)["n_pairs_rigid"])
        ok, res = cmr.check(self.chain_run("c3", n_pairs=4.0), TASKS_RD, STEPS, expect_min_pairs="1")
        self.assertEqual(self.failed(res), ["min_pairs_config"])
        # resolved deform_group_size missing (null): falls back to group_size 4 -> 4 != logged 2
        cfg = chain_config()
        cfg["args"]["deform_group_size"] = None
        ok, res = cmr.check(self.chain_run("c4", cfg=cfg, n_pairs=4.0), TASKS_RD, STEPS, expect_min_pairs="n")
        self.assertEqual(self.failed(res), ["n_pairs_deform"])

    def test_project_val_every(self):
        # smoke-like cadence: val_every 100 in config.json; one val/wall_s of 30 s, mean step 0.1 s,
        # recal 3 s
        cfg = chain_config()
        cfg["args"]["val_every"] = 100
        d = self.chain_run("c", cfg=cfg)
        # default: the run's own cadence -> 8000 / 100 = 80 passes: 1.3 x (800 + 80 x 30 + 3) = 4163.9 s
        x = self.details(cmr.check(d, TASKS_RD, STEPS, project_steps=8000)[1])
        self.assertIn("80 val x median", x["step_time"])
        self.assertIn("= 4164 s", x["step_time"])
        # long-run cadence 500 -> 16 passes: 1.3 x (800 + 16 x 30 + 3) = 1667.9 s
        x = self.details(cmr.check(d, TASKS_RD, STEPS, project_steps=8000, project_val_every=500)[1])
        self.assertIn("16 val x median", x["step_time"])
        self.assertIn("= 1668 s", x["step_time"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = cmr.main([d, "--tasks", "rigid,deform", "--project-steps", "8000", "--project-val-every", "500"])
        self.assertEqual(rc, 0)
        self.assertIn("= 1668 s", out.getvalue())
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cmr.main([d, "--tasks", "rigid,deform", "--project-steps", "8000", "--project-val-every", "-1"])

    def test_info_lines_missing_keys(self):
        d = self.run_dir("plain", TASKS_RD)                  # no config.json, no timing / peak / wall keys
        ok, res = cmr.check(d, TASKS_RD, STEPS, memory_budget_mb=36864, project_steps=8000)
        self.assertTrue(ok, self.failed(res))
        x = self.details(res)
        self.assertIn("INFO gpu_peak n/a", x["gpu_peak"])
        self.assertIn("rigid n/a s", x["step_time"])
        self.assertTrue(x["step_time"].endswith("= n/a"), x["step_time"])

    def test_expected_pairs(self):
        for (n, want) in ((2, 2), (3, 3), (4, 4), (8, 8), (11, 11), (12, 13), (16, 24), (32, 99)):
            self.assertEqual(cmr.expected_pairs(n, n, 0.1), want, n)
        self.assertEqual(cmr.expected_pairs(1, 4, 0.1), 1)
        self.assertEqual(cmr.expected_pairs(1, 16, 0.1), 24)
        self.assertEqual(cmr.expected_pairs(4, 4, 1.0), 12)


if __name__ == "__main__":
    unittest.main()
