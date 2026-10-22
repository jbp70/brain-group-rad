#!/usr/bin/env python
"""Check a finished mixed-task main.py run directory (milestone 5 `dryrun_mixed`, milestone 6
`dryrun_deform`). numpy-free, TF-free (Python 3.8 stdlib only), so it runs inside or outside the
container.

    python tests/gpu/check_mixed_run.py <run_dir> [--tasks rigid,affine] [--steps 20]

Tasks: rigid, affine (linear) and deform. Passes (exit 0) when:
* every task trained: ``train_<task>/loss`` logged and finite, and the logged steps reach --steps;
* every task was validated at the final step: linear ``val_<task>/{center,pair}_rms_mm`` finite,
  the affine set also with its scale/shear/rotation errors; deform
  ``val_deform/{template_dice,pair_dice,neg_jac_pct,inv_consistency_mm,centrality_mm}`` finite;
  with >= 2 linear tasks ``val/mean_pair_rms_mm`` (the default best metric of mixed linear runs)
  equal to the mean of the linear tasks' pair RMS (skipped with fewer linear tasks);
* each step function was traced exactly once (``fit_summary.json`` or ``overfit_summary.json``
  ``trace_counts``) and no step was non-finite;
* ``val_png/<task>_step<final>.png`` exists for every task, and a 'best' checkpoint was written
  in ``ckpt/best`` (only checked when --expect-best, the default); with deform AND a linear task
  also ``ckpt/best_deform`` (the deform best file; ``ckpt/best`` keeps the linear criterion);
* milestone 7, optional: ``--expect-init-weights full|partial``: config.json ``init_weights``
  has that mode; partial: encoder + linear_decoder restored and ``fresh_parts`` exactly
  [deform_decoder] when deform is trained (else none). ``--expect-enc-grad true``: every task
  logged finite ``train_<task>/diag_enc_grad_norm_{raw,eff}``; prints the medians and the
  linear / deform median _eff ratio as an INFO line (never a failure); with --task-grad-norm
  (v3.10) each task's detail also gives the median grad_norm_scale and the mean gated fraction.
* milestone 8 (level cascade, `.agents/tasks/milestone-8/design.md` §13), all optional and off by
  default (level-4 directories check exactly as before):
  - ``--expect-cascade true``: config.json ``cascade`` has ``source_checkpoint``,
    ``source_level == 4``, an integer ``source_step`` and ``fingerprint.n_vars > 0``;
    ``init_weights.via == 'cascade'``; the final-step validation has finite
    ``val_<task>_frozen/{center,pair}_rms_mm`` per linear task and
    ``val_deform_frozen/{template_dice,pair_dice}``. INFO lines (never a failure):
    ``cascade_gain <set>`` (composed vs frozen-stage baseline per validation set, better / worse),
    the per-task data-wait fraction data_ms / (data_ms + compute_ms), and ``g_gap <task>``
    (``<task>`` vs ``<task>_g`` for the composed sets and the frozen baselines:
    {center,pair}_{rms_mm,rot_deg,trans_mm}, affine also {center,pair}_{scale_pct,shear_pct,
    qr_rot_deg}).
  - ``--expect-bn-recal true``: a ``bn_recalibration`` event with ``n_layers_updated > 0`` and
    ``layers_without_updates == []``; INFO lines ``bn_gap <set>`` (pre / post |eval - trainmode|)
    when its gap table is non-empty (an empty table is not a failure).
  - ``--expect-skip true``: ``train/cascade_skip_frac`` logged and finite.
  - ``--grad-balance-rule true`` (implies --expect-enc-grad): per linear task the ratio of the
    medians of ``train_<task>/diag_enc_grad_norm_eff`` and ``train_deform/..._eff`` over the log
    records whose ``step`` field (the window end) is >= --grad-balance-from-step; prints
    ``grad_balance KEEP task-grad-norm`` when every ratio is in [0.1, 10], else
    ``grad_balance FALLBACK deform:64``, or ``grad_balance UNDECIDED (treat as FALLBACK and
    inspect the run)`` when values are missing. Informational: never a failure.
INFO lines carry passed=True and the detail prefix 'INFO', so they never change the PASS/FAIL
tally; they are printed with the label INFO.
Prints one line per criterion and writes nothing.
"""
import argparse
import json
import math
import os
import sys

LINEAR_TASKS = ("rigid", "affine")
DEFORM_TASK = "deform"
AFFINE_KEYS = ("center_scale_pct", "center_shear_pct", "center_qr_rot_deg", "pair_scale_pct",
               "pair_shear_pct", "pair_qr_rot_deg")
DEFORM_VAL_KEYS = ("template_dice", "pair_dice", "neg_jac_pct", "inv_consistency_mm", "centrality_mm")
# Milestone 8 (cascade): frozen-stage baseline keys required at the final step, the keys compared
# by cascade_gain (lower is better for the linear ones, higher for the deform Dice) and g_gap.
LINEAR_FROZEN_KEYS = ("center_rms_mm", "pair_rms_mm")
DEFORM_FROZEN_KEYS = ("template_dice", "pair_dice")
G_GAP_KEYS = tuple("%s_%s" % (w, k) for w in ("center", "pair") for k in ("rms_mm", "rot_deg", "trans_mm"))
G_GAP_AFFINE_KEYS = tuple("%s_%s" % (w, k) for w in ("center", "pair")
                          for k in ("scale_pct", "shear_pct", "qr_rot_deg"))
CASCADE_SOURCE_LEVEL = 4
SKIP_KEY = "train/cascade_skip_frac"
GRAD_BALANCE_RANGE = (0.1, 10.0)
GRAD_BALANCE_FROM_STEP = 50


def _finite(v):
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _has_checkpoint(run_dir, sub, prefix):
    d = os.path.join(run_dir, "ckpt", sub)
    return os.path.isdir(d) and any(n.startswith(prefix + "-") for n in os.listdir(d))


def _median(values):
    v = sorted(float(x) for x in values)
    if not v:
        return float("nan")
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def _check_init_weights(run_dir, tasks, expect):
    """config.json init_weights record of a warm-started run (milestone 7)."""
    path = os.path.join(run_dir, "config.json")
    iw = None
    if os.path.exists(path):
        with open(path) as f:
            iw = json.load(f).get("init_weights")
    if not iw:
        return [("init_weights", False, "config.json has no init_weights record (expected %s)" % expect)]
    ok = iw.get("mode") == expect
    detail = "mode %s (expected %s)" % (iw.get("mode"), expect)
    if expect == "partial":
        restored = sorted(iw.get("restored_parts") or {})
        fresh = sorted(iw.get("fresh_parts") or {})
        want_fresh = [DEFORM_TASK + "_decoder"] if DEFORM_TASK in tasks else []
        ok = ok and "encoder" in restored and "linear_decoder" in restored and fresh == want_fresh
        detail += ", restored %s, fresh %s (expected %s), config diff %s" % (
            restored, fresh, want_fresh, iw.get("config_diff"))
    return [("init_weights", ok, detail)]


def _check_enc_grad(recs, tasks):
    """train_<task>/diag_enc_grad_norm_{raw,eff} logged and finite for every task; the median
    _eff ratio linear / deform is an info line (never a failure)."""
    res, med = [], {}
    for t in tasks:
        keys = ["train_%s/diag_enc_grad_norm_%s" % (t, k) for k in ("raw", "eff")]
        vals = {k: [r["data"][k] for r in recs if k in r["data"]] for k in keys}
        ok = all(vals[k] and all(_finite(v) for v in vals[k]) for k in keys)
        med[t] = {k.rsplit("_", 1)[1]: _median(vals[k]) for k in keys}
        gn = [r["data"]["train_%s/grad_norm" % t] for r in recs if "train_%s/grad_norm" % t in r["data"]]
        med[t]["grad_norm"] = _median(gn)
        detail = "%d/%d logged values, median raw %.4g, eff %.4g, task grad_norm %.4g" % (
            len(vals[keys[0]]), len(vals[keys[1]]), med[t]["raw"], med[t]["eff"], med[t]["grad_norm"])
        sk, gk = "train_%s/grad_norm_scale" % t, "train_%s/grad_norm_gated_frac" % t
        scales = [r["data"][sk] for r in recs if sk in r["data"]]
        if scales:                       # --task-grad-norm runs (v3.10); info only
            gated = [float(r["data"][gk]) for r in recs if gk in r["data"]]
            detail += ", grad_norm_scale median %.4g, gated frac %.3g" % (
                _median(scales), sum(gated) / len(gated) if gated else float("nan"))
        res.append(("enc_grad_%s" % t, ok, detail))
    if DEFORM_TASK in tasks:
        d = med[DEFORM_TASK]["eff"]
        ratios = {t: (med[t]["eff"] / d if d and math.isfinite(d) and d > 0 else float("nan"))
                  for t in tasks if t in LINEAR_TASKS}
        if ratios:
            res.append(("enc_grad_ratio", True, "INFO median eff ratio linear/deform %s (target [0.1, 10])"
                        % ", ".join("%s %.3g" % kv for kv in ratios.items())))
    return res


def _load_config(run_dir):
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _fmt(v):
    return "%.4g" % float(v) if _finite(v) else "n/a"


def _mean(values):
    v = [float(x) for x in values if _finite(x)]
    return sum(v) / len(v) if v else float("nan")


def _frozen_sets(val):
    """Validation set names with frozen-stage baselines at the final step (val_<set>_frozen/...)."""
    names = set()
    for k in val:
        head = k.split("/", 1)[0]
        if head.startswith("val_") and head.endswith("_frozen"):
            names.add(head[len("val_"):-len("_frozen")])
    return sorted(names)


def _check_cascade(run_dir, tasks, val, recs):
    """Milestone 8 --expect-cascade: provenance, final-step frozen baselines (failures) and the
    cascade_gain / data-wait / g_gap INFO lines (never failures)."""
    cfg = _load_config(run_dir)
    cas = cfg.get("cascade") or {}
    iw = cfg.get("init_weights") or {}
    fp = cas.get("fingerprint") or {}
    step, n_vars = cas.get("source_step"), fp.get("n_vars")
    problems = []
    if not cas:
        problems.append("config.json has no cascade record")
    if not cas.get("source_checkpoint"):
        problems.append("no source_checkpoint")
    if cas.get("source_level") != CASCADE_SOURCE_LEVEL:
        problems.append("source_level %r (expected %d)" % (cas.get("source_level"), CASCADE_SOURCE_LEVEL))
    if not isinstance(step, int) or isinstance(step, bool):
        problems.append("source_step %r is not an integer" % (step,))
    if not (isinstance(n_vars, int) and not isinstance(n_vars, bool) and n_vars > 0):
        problems.append("fingerprint.n_vars %r (expected > 0)" % (n_vars,))
    if iw.get("via") != "cascade":
        problems.append("init_weights.via %r (expected 'cascade')" % (iw.get("via"),))
    res = [("cascade_config", not problems, "source %s, level %s, step %s, fingerprint n_vars %s, S rel diff %s%s" % (
        cas.get("source_checkpoint"), cas.get("source_level"), step, n_vars, cas.get("S_rel_diff_vs_level4"),
        "" if not problems else "; " + "; ".join(problems)))]
    for t in tasks:
        if t == DEFORM_TASK:
            keys = ["val_deform_frozen/" + k for k in DEFORM_FROZEN_KEYS]
        else:
            keys = ["val_%s_frozen/%s" % (t, k) for k in LINEAR_FROZEN_KEYS]
        missing = [k for k in keys if not _finite(val.get(k))]
        res.append(("val_%s_frozen" % t, not missing, "final-step frozen-stage baseline %s" % (
            "complete" if not missing else "missing/non-finite: %s" % missing)))
    for name in _frozen_sets(val):
        deform = name.split("_", 1)[0] == DEFORM_TASK
        parts = []
        for k in (DEFORM_FROZEN_KEYS if deform else LINEAR_FROZEN_KEYS):
            c, z = val.get("val_%s/%s" % (name, k)), val.get("val_%s_frozen/%s" % (name, k))
            if _finite(c) and _finite(z):
                better = float(c) > float(z) if deform else float(c) < float(z)
                verdict = "better" if better else "worse"
            else:
                verdict = "n/a"
            parts.append("%s composed %s vs frozen %s (%s)" % (k, _fmt(c), _fmt(z), verdict))
        res.append(("cascade_gain_%s" % name, True, "INFO cascade_gain %s: %s" % (name, ", ".join(parts))))
    for t in tasks:
        dk, ck = "train_%s/data_ms" % t, "train_%s/compute_ms" % t
        d = _mean(r["data"][dk] for r in recs if dk in r["data"])
        c = _mean(r["data"][ck] for r in recs if ck in r["data"])
        if _finite(d) and _finite(c) and d + c > 0:
            detail = "INFO data wait %s: mean data %.1f ms, compute %.1f ms, fraction %.3f" % (t, d, c, d / (d + c))
        else:
            detail = "INFO data wait %s: no finite %s / %s logged" % (t, dk, ck)
        res.append(("data_wait_%s" % t, True, detail))
    return res + _g_gap(val, tasks)


def _g_gap(val, tasks):
    """INFO g_gap lines: <task> vs <task>_g validation sets, composed and frozen baselines."""
    res = []
    for t in tasks:
        if t not in LINEAR_TASKS:
            continue
        keys = G_GAP_KEYS + (G_GAP_AFFINE_KEYS if t == "affine" else ())
        for kind, suffix in (("composed", ""), ("frozen", "_frozen")):
            base, g = "val_%s%s/" % (t, suffix), "val_%s_g%s/" % (t, suffix)
            if not any(k.startswith(base) for k in val):
                continue
            name = "g_gap_%s%s" % (t, suffix)
            if not any(k.startswith(g) for k in val):
                res.append((name, True, "INFO g_gap %s %s: no %s_g validation set" % (t, kind, t)))
                continue
            parts = ["%s %s -> %s" % (k, _fmt(val.get(base + k)), _fmt(val.get(g + k))) for k in keys
                     if base + k in val or g + k in val]
            res.append((name, True, "INFO g_gap %s %s (%s -> %s_g): %s" % (t, kind, t, t, ", ".join(parts))))
    return res


def _check_bn_recal(recs):
    """Milestone 8 --expect-bn-recal: the bn_recalibration event (last one if several) and INFO
    gap lines per validation set."""
    events = [r["data"] for r in recs if r["data"].get("event") == "bn_recalibration"]
    if not events:
        return [("bn_recalibration", False, "no bn_recalibration event in metrics.jsonl")]
    ev = events[-1]
    n_up, stale = ev.get("n_layers_updated"), ev.get("layers_without_updates")
    ok = isinstance(n_up, int) and n_up > 0 and stale == []
    res = [("bn_recalibration", ok, "%d event(s); batches %s, skipped non-finite %s, n_layers_updated %s, "
            "layers_without_updates %s, freeze %s, %.1f s" % (
                len(events), ev.get("batches"), ev.get("skipped_nonfinite"), n_up, stale, ev.get("freeze"),
                float(ev.get("elapsed_s") or float("nan"))))]
    gap = ev.get("gap") or {}
    sets = {}
    for k in sorted(gap):
        when, rest = k.split("/", 1) if "/" in k else (k, "")
        if "/" not in rest:
            continue
        name, metric = rest.rsplit("/", 1)
        sets.setdefault(name, {}).setdefault(metric, {})[when] = gap[k]
    for name in sorted(sets):
        parts = ["%s pre %s post %s" % (m, _fmt(v.get("gap_pre")), _fmt(v.get("gap_post")))
                 for m, v in sorted(sets[name].items())]
        res.append(("bn_gap_%s" % name, True, "INFO bn_gap %s (|eval - trainmode|): %s" % (name, ", ".join(parts))))
    return res


def _check_skip(recs):
    vals = [r["data"][SKIP_KEY] for r in recs if SKIP_KEY in r["data"]]
    ok = bool(vals) and all(_finite(v) for v in vals)
    return [("cascade_skip", ok, "%d logged %s values, all finite, mean %s" % (len(vals), SKIP_KEY, _fmt(_mean(vals))))]


def grad_balance(recs, tasks, from_step=GRAD_BALANCE_FROM_STEP):
    """(verdict, {linear task: ratio}, detail): KEEP / FALLBACK / UNDECIDED from the ratio of the
    medians of ``train_<task>/diag_enc_grad_norm_eff`` (linear) and the deform one, each over
    the finite values of log records whose ``step`` (window end) is >= `from_step`."""
    def med(t):
        k = "train_%s/diag_enc_grad_norm_eff" % t
        v = [r["data"][k] for r in recs if r["step"] >= from_step and k in r["data"] and _finite(r["data"][k])]
        return _median(v) if v else None

    linear = [t for t in tasks if t in LINEAR_TASKS]
    d = med(DEFORM_TASK) if DEFORM_TASK in tasks else None
    meds = {t: med(t) for t in linear}
    missing = ([] if d is not None and d > 0 else [DEFORM_TASK]) + [t for t in linear if meds[t] is None]
    if not linear or missing:
        why = "no linear task" if not linear else "no finite (or zero deform) _eff values with step >= %d for %s" % (
            from_step, missing)
        return "UNDECIDED", {}, why
    ratios = {t: meds[t] / d for t in linear}
    lo, hi = GRAD_BALANCE_RANGE
    verdict = "KEEP" if all(lo <= r <= hi for r in ratios.values()) else "FALLBACK"
    return verdict, ratios, "median eff ratio linear/deform over records with step >= %d: %s (target [%g, %g])" % (
        from_step, ", ".join("%s %.3g" % (t, ratios[t]) for t in linear), lo, hi)


def _check_grad_balance(recs, tasks, from_step):
    verdict, _, why = grad_balance(recs, tasks, from_step)
    label = {"KEEP": "KEEP task-grad-norm", "FALLBACK": "FALLBACK deform:64",
             "UNDECIDED": "UNDECIDED (treat as FALLBACK and inspect the run)"}[verdict]
    return [("grad_balance", True, "INFO grad_balance %s: %s" % (label, why))]


def check(run_dir, tasks, steps, expect_best=True, expect_init_weights=None, expect_enc_grad=False,
          expect_cascade=False, expect_bn_recal=False, expect_skip=False, grad_balance_rule=False,
          grad_balance_from_step=GRAD_BALANCE_FROM_STEP):
    """Returns (ok, [(name, passed, detail)]). ``expect_init_weights`` None (not checked) |
    'none' (not checked) | 'full' | 'partial'; ``expect_enc_grad``: check the milestone-7
    encoder grad-norm diagnostics. Milestone 8 (module docstring): ``expect_cascade``,
    ``expect_bn_recal``, ``expect_skip``, ``grad_balance_rule`` (implies ``expect_enc_grad``;
    the verdict is INFO only) with ``grad_balance_from_step``."""
    res = []
    with open(os.path.join(run_dir, "logs", "metrics.jsonl")) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    recs = [r for r in recs if isinstance(r.get("step"), int)]
    final = max(r["step"] for r in recs) if recs else 0
    res.append(("final_step", final >= steps, "last logged step %d (expected >= %d)" % (final, steps)))
    for t in tasks:
        losses = [r["data"]["train_%s/loss" % t] for r in recs if "train_%s/loss" % t in r["data"]]
        res.append(("train_%s" % t, bool(losses) and all(_finite(v) for v in losses),
                    "%d logged train_%s/loss values, all finite" % (len(losses), t)))
    val = {}
    for r in recs:
        if r["step"] == final:
            val.update({k: v for k, v in r["data"].items() if k.startswith("val")})
    linear = [t for t in tasks if t in LINEAR_TASKS]
    pairs = []
    for t in tasks:
        if t == DEFORM_TASK:
            keys = ["val_deform/" + k for k in DEFORM_VAL_KEYS]
        else:
            keys = ["val_%s/center_rms_mm" % t, "val_%s/pair_rms_mm" % t]
            if t == "affine":
                keys += ["val_affine/" + k for k in AFFINE_KEYS]
            pairs.append(val.get("val_%s/pair_rms_mm" % t))
        missing = [k for k in keys if not _finite(val.get(k))]
        res.append(("val_%s" % t, not missing, "final-step validation %s" % (
            "complete" if not missing else "missing/non-finite: %s" % missing)))
    mean = val.get("val/mean_pair_rms_mm")
    if len(linear) < 2:
        res.append(("val_mean_pair_rms_mm", True, "skipped (%d linear task(s))" % len(linear)))
    else:
        mean_ok = (_finite(mean) and all(_finite(p) for p in pairs)
                   and abs(float(mean) - sum(pairs) / len(pairs)) < 1e-6)
        res.append(("val_mean_pair_rms_mm", mean_ok,
                    "val/mean_pair_rms_mm = %s, linear task pair RMS %s" % (mean, pairs)))
    summ = None
    for name in ("fit_summary.json", "overfit_summary.json"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                summ = json.load(f)
            break
    tc = (summ or {}).get("trace_counts")
    res.append(("trace_once", tc is not None and all(tc.get(t) == 1 for t in tasks),
                "trace counts %s (1 per task)" % tc))
    nf = (summ or {}).get("nonfinite_total", 0)
    res.append(("finite_steps", nf == 0, "non-finite steps: %s" % nf))
    pngs = ["%s_step%07d.png" % (t, final) for t in tasks]
    have = set(os.listdir(os.path.join(run_dir, "val_png"))) if os.path.isdir(os.path.join(run_dir, "val_png")) else set()
    res.append(("val_png", all(p in have for p in pngs), "PNGs %s present: %s" % (pngs, [p in have for p in pngs])))
    if expect_best:
        ok_best = _has_checkpoint(run_dir, "best", "best")
        res.append(("best_checkpoint", ok_best, "ckpt/best has a checkpoint: %s" % ok_best))
        if DEFORM_TASK in tasks and linear:
            ok_bd = _has_checkpoint(run_dir, "best_deform", "best_deform")
            res.append(("best_deform_checkpoint", ok_bd, "ckpt/best_deform has a checkpoint: %s" % ok_bd))
    if expect_init_weights not in (None, "none"):
        res += _check_init_weights(run_dir, tasks, expect_init_weights)
    if expect_enc_grad or grad_balance_rule:
        res += _check_enc_grad(recs, tasks)
    if expect_cascade:
        res += _check_cascade(run_dir, tasks, val, recs)
    if expect_bn_recal:
        res += _check_bn_recal(recs)
    if expect_skip:
        res += _check_skip(recs)
    if grad_balance_rule:
        res += _check_grad_balance(recs, tasks, grad_balance_from_step)
    return all(p for _, p, _ in res), res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--tasks", default="rigid,affine")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--expect-best", default="true", choices=["true", "false"])
    ap.add_argument("--expect-init-weights", default="none", choices=["none", "full", "partial"],
                    help="check config.json init_weights (warm start mode); none = not checked")
    ap.add_argument("--expect-enc-grad", default="false", choices=["true", "false"],
                    help="check train_<task>/diag_enc_grad_norm_{raw,eff} (milestone 7)")
    ap.add_argument("--expect-cascade", default="false", choices=["true", "false"],
                    help="milestone 8: check config.json cascade provenance and the final-step "
                    "val_<task>_frozen baselines; prints cascade_gain, data-wait and g_gap INFO lines")
    ap.add_argument("--expect-bn-recal", default="false", choices=["true", "false"],
                    help="milestone 8: check the bn_recalibration event (layers updated, none stale)")
    ap.add_argument("--expect-skip", default="false", choices=["true", "false"],
                    help="milestone 8: check that %s is logged and finite" % SKIP_KEY)
    ap.add_argument("--grad-balance-rule", default="false", choices=["true", "false"],
                    help="milestone 8 (implies --expect-enc-grad): print grad_balance KEEP task-grad-norm "
                    "/ FALLBACK deform:64 / UNDECIDED from the linear/deform median "
                    "diag_enc_grad_norm_eff ratios (KEEP iff all in [0.1, 10]); INFO only, never a failure")
    ap.add_argument("--grad-balance-from-step", type=int, default=GRAD_BALANCE_FROM_STEP,
                    help="the grad-balance medians use the log records whose 'step' field (the END of "
                    "the logging window) is >= this value; with --log-every 25 the first used window "
                    "is steps 26-50 (default %(default)s)")
    a = ap.parse_args(argv)
    tasks = [t for t in a.tasks.split(",") if t]
    unknown = [t for t in tasks if t not in LINEAR_TASKS + (DEFORM_TASK,)]
    if unknown:
        ap.error("unknown task(s) %s" % unknown)
    ok, res = check(a.run_dir, tasks, a.steps, a.expect_best == "true",
                    expect_init_weights=a.expect_init_weights, expect_enc_grad=a.expect_enc_grad == "true",
                    expect_cascade=a.expect_cascade == "true", expect_bn_recal=a.expect_bn_recal == "true",
                    expect_skip=a.expect_skip == "true", grad_balance_rule=a.grad_balance_rule == "true",
                    grad_balance_from_step=a.grad_balance_from_step)
    for name, passed, detail in res:
        label = "INFO" if passed and detail.startswith("INFO") else ("PASS" if passed else "FAIL")
        print("%s %-22s %s" % (label, name, detail))
    print("check_mixed_run: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
