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
  - ``--expect-skip true``: ``train/cascade_skip_frac`` logged and finite. Per-stage skip
    (retrain-prereqs design 4.5, config.json ``cascade.skip_mode == 'stage'``): instead, every
    ``train_<task>/cascade_skip_l<lv>`` with p > 0 in ``cascade.skip_probs`` logged and finite at
    least once, and INFO ``skip <task> l<lv>: observed <mean> vs p <p>`` (never a failure).
  - ``--grad-balance-rule true`` (implies --expect-enc-grad): per linear task the ratio of the
    medians of ``train_<task>/diag_enc_grad_norm_eff`` and ``train_deform/..._eff`` over the log
    records whose ``step`` field (the window end) is >= --grad-balance-from-step; prints
    ``grad_balance KEEP task-grad-norm`` when every ratio is in [0.1, 10], else
    ``grad_balance FALLBACK deform:64``, or ``grad_balance UNDECIDED (treat as FALLBACK and
    inspect the run)`` when values are missing. Informational: never a failure. A run trained
    with clip 12 + deform:64 (config.json ``args.task_grad_norm == 0``) is labelled
    ``KEEP current balance (deform64)`` instead of ``KEEP task-grad-norm``.
* milestone 9 phase 2 (`.agents/tasks/milestone-9-phase2/design-m9-phase2.md` §9), all optional
  and off by default (milestone-8 directories check exactly as before):
  - ``--expect-cascade-stages 2`` (implies --expect-cascade; default 1 = the milestone-8 checks):
    the level-1 3-stage cascade. config.json ``cascade.source_level == 2``, ``cascade.stages``
    with levels [4, 2], each with an integer ``fingerprint.n_vars > 0`` and an integer
    ``origin_step``; ``init_weights.via == 'cascade'`` and ``cross_level == [2, 1]``; final-step
    finite ``val_<task>_frozen4/{center,pair}_rms_mm`` per linear task and
    ``val_deform_frozen4/{template_dice,pair_dice}`` (besides the ``_frozen`` keys = levels 4 + 2).
    INFO lines: cascade_gain also against ``_frozen4``, g_gap also for ``_frozen4``,
    ``deform_extra`` (composed / frozen / frozen4 / identity for neg_jac_pct, inv_consistency_mm,
    ncc_template, ncc_pair; the frozen centrality keys are not compared: ``_frozen`` describes the
    level-2 residual SVF only), and ``S_rel_diff_vs_level2`` in the cascade_config line.
  - ``--expect-min-pairs none|1|n`` (default none): config.json ``args.min_pairs`` equals the
    value; with n every logged ``train_deform/n_pairs`` equals expected_pairs(N_deform, N_deform,
    p_pdice) and ``train_<lin>/n_pairs`` equals expected_pairs(N, N, p_pair) (the stdlib mirror
    of ``data_loading.postprocess.num_pairs``), capped at config.json ``args.max_pairs`` when > 0.
  - ``--memory-budget-mb M`` (INFO ``gpu_peak`` line: max logged ``gpu_peak_mb`` vs M) and
    ``--project-steps S`` (INFO ``step_time`` line: per-task mean compute + data ms, the
    ratio-weighted mean step, median / first ``val/wall_s`` and the projected wall time
    1.3 x (S x mean step + (S / val_every) x median val/wall_s + recal elapsed_s); missing keys
    print n/a; ``--project-val-every V`` > 0 uses V instead of the run's val_every, e.g. 500
    for the long run when analysing a VAL_EVERY 100 smoke). Never failures.
* nonfinite-fixes (`.agents/tasks/nonfinite-fixes/plan.md` D7), optional and off by default:
  - ``--max-window-nonfinite-frac F`` (> 0; scripts pass 0.05) with ``--nonfinite-window-steps W``
    (500) and ``--nonfinite-from-step S`` (100): per task one ``nonfinite_window_<task>`` result.
    The logged ``train_<task>/nonfinite_steps`` / ``window_steps`` of records with step > S are
    summed into W-step buckets; buckets with < 20 task steps are skipped; FAIL if a bucket's
    fraction > F (detail: first such bucket, counts, the first record step with a non-finite
    step). Runs without these keys (older code) PASS as "skipped".
  - ``loss_scale`` INFO line whenever ``optimizer/loss_scale`` is logged (AMP): last / min value
    and the number of log windows at the floor (<= 1); "WARN" in the text when the last value is
    at the floor. Never a failure.
INFO lines carry passed=True and the detail prefix 'INFO', so they never change the PASS/FAIL
tally; they are printed with the label INFO.
Prints one line per criterion and writes nothing.
"""
import argparse
import json
import math
import os
import struct
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
# Milestone 9 phase 2 (design §9): 3-stage (level-1) cascade, minimum pair count, INFO budgets.
CHAIN_SOURCE_LEVEL = 2
CHAIN_STAGE_LEVELS = [4, 2]
CHAIN_CROSS_LEVEL = [2, 1]
MIN_PAIRS_CHOICES = ("none", "1", "n")
PAIR_COUNT_KEY = "n_pairs"
DEFORM_EXTRA_KEYS = ("neg_jac_pct", "inv_consistency_mm", "ncc_template", "ncc_pair")
VAL_WALL_KEY = "val/wall_s"
GPU_PEAK_KEY = "gpu_peak_mb"
PROJECT_MARGIN = 1.3
# nonfinite-fixes (plan D7): per-task, per-window non-finite rate and the AMP loss scale.
LOSS_SCALE_KEY = "optimizer/loss_scale"
MIN_WINDOW_TASK_STEPS = 20       # buckets with fewer steps of the task are not judged
LOSS_SCALE_FLOOR = 1.0


def _f32(x):
    """Round a Python float to float32 (stdlib mirror of the TF float32 arithmetic)."""
    return struct.unpack("f", struct.pack("f", float(x)))[0]


def expected_pairs(m_floor, n, p, max_pairs=0):
    """Sampled ordered pairs per training step: min(max(m_floor, round(p n (n - 1))), n (n - 1)),
    with the product in float32 and round half-to-even (Python 3 round = tf.round), exactly as
    ``data_loading.postprocess.num_pairs`` (m_floor 1 for --min-pairs 1, n for --min-pairs n),
    then capped at ``max_pairs`` K when K > 0 (--max-pairs, retrain-prereqs design 16.3)."""
    total = n * (n - 1)
    k = int(round(_f32(_f32(total) * _f32(p))))
    k = min(max(m_floor, k), total)
    return min(k, int(max_pairs)) if max_pairs and int(max_pairs) > 0 else k


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


def _frozen4_sets(val):
    """Validation set names with level-4-alone baselines (val_<set>_frozen4/..., level-1 chain)."""
    names = set()
    for k in val:
        head = k.split("/", 1)[0]
        if head.startswith("val_") and head.endswith("_frozen4"):
            names.add(head[len("val_"):-len("_frozen4")])
    return sorted(names)


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _verdict(c, z, deform):
    if not (_finite(c) and _finite(z)):
        return "n/a"
    better = float(c) > float(z) if deform else float(c) < float(z)
    return "better" if better else "worse"


def _check_stages(cas, iw):
    """Milestone 9 phase 2 (--expect-cascade-stages 2): the cascade.stages provenance list and
    the level-2 -> level-1 init_weights record."""
    problems = []
    stages = cas.get("stages")
    if not isinstance(stages, list):
        problems.append("no cascade.stages list")
        stages = []
    levels = [s.get("level") for s in stages if isinstance(s, dict)]
    if levels != CHAIN_STAGE_LEVELS:
        problems.append("stages levels %r (expected %r)" % (levels, CHAIN_STAGE_LEVELS))
    for s in stages:
        if not isinstance(s, dict):
            continue
        n_vars = (s.get("fingerprint") or {}).get("n_vars")
        if not (_is_int(n_vars) and n_vars > 0):
            problems.append("level %s fingerprint.n_vars %r (expected > 0)" % (s.get("level"), n_vars))
        if not _is_int(s.get("origin_step")):
            problems.append("level %s origin_step %r is not an integer" % (s.get("level"), s.get("origin_step")))
    if iw.get("cross_level") != CHAIN_CROSS_LEVEL:
        problems.append("init_weights.cross_level %r (expected %r)" % (iw.get("cross_level"), CHAIN_CROSS_LEVEL))
    detail = "stages %s" % ", ".join("level %s key %s origin step %s" % (
        s.get("level"), s.get("checkpoint_key"), s.get("origin_step")) for s in stages if isinstance(s, dict))
    return [("cascade_stages", not problems, detail + ("" if not problems else "; " + "; ".join(problems)))]


def _deform_extra(val):
    """INFO deform_extra (level-1 chain): composed / frozen (4 + 2) / frozen4 / identity."""
    parts = []
    for k in DEFORM_EXTRA_KEYS:
        parts.append("%s %s / %s / %s / %s" % (
            k, _fmt(val.get("val_deform/" + k)), _fmt(val.get("val_deform_frozen/" + k)),
            _fmt(val.get("val_deform_frozen4/" + k)), _fmt(val.get("val_deform/%s_identity" % k))))
    return [("deform_extra", True, "INFO deform_extra (composed / frozen / frozen4 / identity): %s"
             % ", ".join(parts))]


def _check_cascade(run_dir, tasks, val, recs, stages=1):
    """Milestone 8 --expect-cascade: provenance, final-step frozen baselines (failures) and the
    cascade_gain / data-wait / g_gap INFO lines (never failures). ``stages`` 2 (milestone 9
    phase 2): the level-1 3-stage cascade (source level 2, stages [4, 2], ``_frozen4`` keys)."""
    cfg = _load_config(run_dir)
    cas = cfg.get("cascade") or {}
    iw = cfg.get("init_weights") or {}
    fp = cas.get("fingerprint") or {}
    step, n_vars = cas.get("source_step"), fp.get("n_vars")
    want_level = CHAIN_SOURCE_LEVEL if stages == 2 else CASCADE_SOURCE_LEVEL
    problems = []
    if not cas:
        problems.append("config.json has no cascade record")
    if not cas.get("source_checkpoint"):
        problems.append("no source_checkpoint")
    if cas.get("source_level") != want_level:
        problems.append("source_level %r (expected %d)" % (cas.get("source_level"), want_level))
    if not isinstance(step, int) or isinstance(step, bool):
        problems.append("source_step %r is not an integer" % (step,))
    if not (isinstance(n_vars, int) and not isinstance(n_vars, bool) and n_vars > 0):
        problems.append("fingerprint.n_vars %r (expected > 0)" % (n_vars,))
    if iw.get("via") != "cascade":
        problems.append("init_weights.via %r (expected 'cascade')" % (iw.get("via"),))
    s_rel = "S rel diff %s" % (cas.get("S_rel_diff_vs_level4"),)
    if "S_rel_diff_vs_level2" in cas:
        s_rel = "S rel diff vs level 4 %s, vs level 2 %s" % (cas.get("S_rel_diff_vs_level4"),
                                                           cas.get("S_rel_diff_vs_level2"))
    res = [("cascade_config", not problems, "source %s, level %s, step %s, fingerprint n_vars %s, %s%s" % (
        cas.get("source_checkpoint"), cas.get("source_level"), step, n_vars, s_rel,
        "" if not problems else "; " + "; ".join(problems)))]
    if stages == 2:
        res += _check_stages(cas, iw)
    suffixes = ("_frozen", "_frozen4") if stages == 2 else ("_frozen",)
    for suffix in suffixes:
        for t in tasks:
            if t == DEFORM_TASK:
                keys = ["val_deform%s/%s" % (suffix, k) for k in DEFORM_FROZEN_KEYS]
            else:
                keys = ["val_%s%s/%s" % (t, suffix, k) for k in LINEAR_FROZEN_KEYS]
            missing = [k for k in keys if not _finite(val.get(k))]
            res.append(("val_%s%s" % (t, suffix), not missing, "final-step %s baseline %s" % (
                "frozen-stage" if suffix == "_frozen" else "frozen level-4",
                "complete" if not missing else "missing/non-finite: %s" % missing)))
    with4 = set(_frozen4_sets(val))
    for name in sorted(set(_frozen_sets(val)) | with4):
        deform = name.split("_", 1)[0] == DEFORM_TASK
        parts = []
        for k in (DEFORM_FROZEN_KEYS if deform else LINEAR_FROZEN_KEYS):
            c, z = val.get("val_%s/%s" % (name, k)), val.get("val_%s_frozen/%s" % (name, k))
            part = "%s composed %s vs frozen %s (%s)" % (k, _fmt(c), _fmt(z), _verdict(c, z, deform))
            if name in with4:
                z4 = val.get("val_%s_frozen4/%s" % (name, k))
                part += " vs frozen4 %s (%s)" % (_fmt(z4), _verdict(c, z4, deform))
            parts.append(part)
        res.append(("cascade_gain_%s" % name, True, "INFO cascade_gain %s: %s" % (name, ", ".join(parts))))
    if stages == 2 and DEFORM_TASK in tasks:
        res += _deform_extra(val)
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
        for kind, suffix in (("composed", ""), ("frozen", "_frozen"), ("frozen4", "_frozen4")):
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


def _check_skip(recs, cfg=None):
    """--expect-skip: stage mode (config.json ``cascade.skip_mode == 'stage'``, retrain-prereqs
    design 4.5): every ``train_<task>/cascade_skip_l<lv>`` with p > 0 in ``cascade.skip_probs``
    logged and finite at least once, plus an INFO line observed mean vs p per key; else the
    milestone-8 legacy check of ``train/cascade_skip_frac``."""
    casc = (cfg or {}).get("cascade") or {}
    if casc.get("skip_mode") == "stage":
        res = []
        for task, probs in sorted((casc.get("skip_probs") or {}).items()):
            for lv, p in sorted((probs or {}).items(), key=lambda kv: -int(kv[0])):
                if not float(p) > 0:
                    continue
                key = "train_%s/cascade_skip_l%s" % (task, lv)
                vals = [r["data"][key] for r in recs if key in r["data"]]
                ok = bool(vals) and all(_finite(v) for v in vals)
                res.append(("cascade_skip_%s_l%s" % (task, lv), ok,
                            "%d logged %s values, all finite" % (len(vals), key)))
                res.append(("cascade_skip_rate_%s_l%s" % (task, lv), True,
                            "INFO skip %s l%s: observed %s vs p %g" % (task, lv, _fmt(_mean(vals)), float(p))))
        if not res:
            res.append(("cascade_skip", False, "config.json cascade.skip_mode is 'stage' but skip_probs "
                        "has no p > 0"))
        return res
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


def _check_grad_balance(recs, tasks, from_step, args=None):
    verdict, _, why = grad_balance(recs, tasks, from_step)
    # A clip 12 + deform:64 run (task_grad_norm 0) keeps its current balance, not task-grad-norm.
    keep = "KEEP current balance (deform64)" if (args or {}).get("task_grad_norm") == 0 else "KEEP task-grad-norm"
    label = {"KEEP": keep, "FALLBACK": "FALLBACK deform:64",
             "UNDECIDED": "UNDECIDED (treat as FALLBACK and inspect the run)"}[verdict]
    return [("grad_balance", True, "INFO grad_balance %s: %s" % (label, why))]


def _check_min_pairs(recs, tasks, args, expect):
    """Milestone 9 phase 2 --expect-min-pairs: config.json args.min_pairs, and with 'n' every
    logged train_<task>/n_pairs equals expected_pairs (deform: N_deform / p_pdice, linear: N / p_pair)."""
    got = args.get("min_pairs")
    res = [("min_pairs_config", got == expect, "config.json args.min_pairs %r (expected %r)" % (got, expect))]
    if expect != "n":
        return res
    n = args.get("group_size")
    # config.json stores the resolved value (= --group-size when unset); the or fallback is kept
    # for robustness.
    n_deform = args.get("deform_group_size") or n
    max_pairs = args.get("max_pairs") or 0   # --max-pairs (missing in older runs -> no cap)
    max_pairs = int(max_pairs) if _is_int(max_pairs) else 0
    for t in tasks:
        if t == DEFORM_TASK:
            size, p = n_deform, args.get("p_pdice")
        else:
            size, p = n, args.get("p_pair")
        key = "train_%s/%s" % (t, PAIR_COUNT_KEY)
        vals = [r["data"][key] for r in recs if key in r["data"]]
        if not (_is_int(size) and _finite(p)):
            res.append(("n_pairs_%s" % t, False, "config.json args lack the group size / pair probability "
                        "(size %r, p %r)" % (size, p)))
            continue
        want = expected_pairs(size, size, p, max_pairs)
        bad = [v for v in vals if not (_finite(v) and abs(float(v) - want) < 1e-6)]
        res.append(("n_pairs_%s" % t, bool(vals) and not bad,
                    "%d logged %s values, expected %d (N %d, p %g, max pairs %d)%s" % (
                        len(vals), key, want, size, float(p), max_pairs,
                        "" if not bad else "; mismatching %s" % bad[:5])))
    return res


def _gpu_peak(recs, budget_mb):
    """INFO gpu_peak: max logged gpu_peak_mb (process-lifetime allocator peak) vs the budget."""
    vals = [float(r["data"][GPU_PEAK_KEY]) for r in recs if _finite(r["data"].get(GPU_PEAK_KEY))]
    if not vals:
        return [("gpu_peak", True, "INFO gpu_peak n/a (no %s logged) vs A100 budget %.0f GiB"
                 % (GPU_PEAK_KEY, budget_mb / 1024.0))]
    peak = max(vals)
    verdict = "fits" if peak <= budget_mb else "exceeds -> keep rtx8000 or --lean-losses true"
    return [("gpu_peak", True, "INFO gpu_peak %.0f MiB (%.2f GiB) vs A100 budget %.0f GiB: %s"
             % (peak, peak / 1024.0, budget_mb / 1024.0, verdict))]


def _step_time(recs, tasks, args, project_steps, project_val_every=0):
    """INFO step_time: per-task mean compute + data ms, the ratio-weighted mean step, val/wall_s
    and the projected wall time 1.3 x (S x mean step + (S / val_every) x median val/wall_s +
    recal elapsed_s); the recal event's elapsed_s already contains its pre / post validation
    passes (design §8 accounting rule). ``project_val_every`` > 0 is the projected run's
    validation cadence (a smoke run validates more often than the long run); 0 = the analysed
    run's own ``val_every``. Missing values print n/a."""
    per = {}
    for t in tasks:
        c = _mean(r["data"]["train_%s/compute_ms" % t] for r in recs if "train_%s/compute_ms" % t in r["data"])
        d = _mean(r["data"]["train_%s/data_ms" % t] for r in recs if "train_%s/data_ms" % t in r["data"])
        per[t] = (c + d) / 1000.0 if _finite(c) and _finite(d) else float("nan")
    ratio = args.get("task_ratio_parsed") or {}
    w = {t: float(ratio.get(t, 0.0)) for t in tasks}
    if sum(w.values()) > 0 and all(_finite(per[t]) for t in tasks if w[t] > 0):
        mean_step = sum(w[t] * per[t] for t in tasks if w[t] > 0) / sum(w.values())
    else:
        mean_step = float("nan")
    walls = [float(r["data"][VAL_WALL_KEY]) for r in recs if _finite(r["data"].get(VAL_WALL_KEY))]
    med_wall = _median(walls) if walls else float("nan")
    first_wall = walls[0] if walls else float("nan")
    events = [r["data"] for r in recs if r["data"].get("event") == "bn_recalibration"]
    recal = float(events[-1].get("elapsed_s")) if events and _finite(events[-1].get("elapsed_s")) else 0.0
    val_every = project_val_every if project_val_every > 0 else args.get("val_every")
    n_val = project_steps / float(val_every) if _is_int(val_every) and val_every > 0 else 0.0
    if _finite(mean_step) and (n_val == 0 or _finite(med_wall)):
        total = PROJECT_MARGIN * (project_steps * mean_step + n_val * (med_wall if n_val else 0.0) + recal)
        proj = "%.0f s (%.2f h)" % (total, total / 3600.0)
    else:
        proj = "n/a"
    return [("step_time", True, "INFO step_time: per task %s; ratio-weighted mean step %s s; val/wall_s median %s "
             "(first %s, %d passes); recal %.1f s; projected %d steps = %.1f x (S x step + %g val x median "
             "val/wall_s + recal) = %s" % (
                 ", ".join("%s %s s" % (t, _fmt(per[t])) for t in tasks), _fmt(mean_step), _fmt(med_wall),
                 _fmt(first_wall), len(walls), recal, project_steps, PROJECT_MARGIN, n_val, proj))]


def _check_nonfinite_windows(recs, tasks, max_frac, window_steps, from_step):
    """Per task: aggregate the logged ``train_<t>/nonfinite_steps`` / ``window_steps`` of the train
    records with step > from_step into buckets (step - 1) // window_steps; FAIL if any bucket with
    >= MIN_WINDOW_TASK_STEPS task steps has a non-finite fraction > max_frac (detail: the first
    such bucket and the first record step in it with non-finite steps)."""
    res = []
    for t in tasks:
        nk, wk = "train_%s/nonfinite_steps" % t, "train_%s/window_steps" % t
        rows = [(r["step"], r["data"][nk], r["data"][wk]) for r in recs
                if nk in r["data"] and wk in r["data"]]
        name = "nonfinite_window_%s" % t
        if not rows:
            res.append((name, True, "skipped: no %s logged" % nk))
            continue
        buckets = {}
        for step, bad, n in rows:
            if step <= from_step:
                continue
            b = buckets.setdefault((step - 1) // window_steps, [0, 0, None])
            b[0] += int(bad)
            b[1] += int(n)
            if int(bad) > 0 and b[2] is None:
                b[2] = step
        judged = [(k, v) for k, v in sorted(buckets.items()) if v[1] >= MIN_WINDOW_TASK_STEPS]
        failing = [(k, v) for k, v in judged if v[0] > max_frac * v[1]]
        if failing:
            k, (bad, n, first) = failing[0]
            res.append((name, False, "task %s: %d of %d steps non-finite (%.1f %% > %.1f %%) in window "
                        "steps %d-%d (first logged at step %s); %d of %d windows over the limit"
                        % (t, bad, n, 100.0 * bad / n, 100.0 * max_frac, k * window_steps + 1,
                           (k + 1) * window_steps, first, len(failing), len(judged))))
        else:
            worst = max((v[0] / float(v[1]) for _, v in judged), default=0.0)
            res.append((name, True, "task %s: %d windows of %d steps after step %d judged, max "
                        "non-finite %.1f %% (limit %.1f %%)" % (t, len(judged), window_steps, from_step,
                                                                100.0 * worst, 100.0 * max_frac)))
    return res


def _loss_scale_info(recs):
    """INFO loss_scale line when ``optimizer/loss_scale`` is logged (never a failure)."""
    vals = [(r["step"], float(r["data"][LOSS_SCALE_KEY])) for r in recs
            if _finite(r["data"].get(LOSS_SCALE_KEY))]
    if not vals:
        return []
    last = vals[-1][1]
    lo_step, lo = min(vals, key=lambda sv: sv[1])
    floor = sum(1 for _, v in vals if v <= LOSS_SCALE_FLOOR)
    detail = "INFO last %g, min %g (step %d), %d of %d log windows at the floor (<= 1)" % (
        last, lo, lo_step, floor, len(vals))
    if last <= LOSS_SCALE_FLOOR:
        detail = ("INFO WARN loss scale at the floor (1): healthy tasks train with fp16 gradient "
                  "underflow risk; " + detail[len("INFO "):])
    return [("loss_scale", True, detail)]


def check(run_dir, tasks, steps, expect_best=True, expect_init_weights=None, expect_enc_grad=False,
          expect_cascade=False, expect_bn_recal=False, expect_skip=False, grad_balance_rule=False,
          grad_balance_from_step=GRAD_BALANCE_FROM_STEP, max_nonfinite_frac=0.0, expect_cascade_stages=1,
          expect_min_pairs=None, memory_budget_mb=0, project_steps=0, project_val_every=0,
          max_window_nonfinite_frac=0.0, nonfinite_window_steps=500, nonfinite_from_step=100):
    """Returns (ok, [(name, passed, detail)]). ``expect_init_weights`` None (not checked) |
    'none' (not checked) | 'full' | 'partial'; ``expect_enc_grad``: check the milestone-7
    encoder grad-norm diagnostics. Milestone 8 (module docstring): ``expect_cascade``,
    ``expect_bn_recal``, ``expect_skip``, ``grad_balance_rule`` (implies ``expect_enc_grad``;
    the verdict is INFO only) with ``grad_balance_from_step``. Milestone 9 phase 2:
    ``expect_cascade_stages`` 1 | 2 (2 implies ``expect_cascade``), ``expect_min_pairs`` None /
    'none' (not checked) | '1' | 'n', ``memory_budget_mb`` and ``project_steps`` (INFO only,
    0 = off; ``project_val_every`` > 0 overrides the run's val_every in the projection).
    nonfinite-fixes: ``max_window_nonfinite_frac`` F > 0 (default 0 = off) adds one
    ``nonfinite_window_<task>`` result per task (``nonfinite_window_steps`` W, records with step >
    ``nonfinite_from_step``); a ``loss_scale`` INFO line is added whenever
    ``optimizer/loss_scale`` is logged. The defaults reproduce the milestone-8 checks."""
    if expect_cascade_stages not in (1, 2):
        raise ValueError("expect_cascade_stages must be 1 or 2, got %r" % (expect_cascade_stages,))
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
    if max_nonfinite_frac > 0:
        # AMP: the dynamic loss scale starts high and halves on overflow, so a few skipped
        # steps (inf gradients, weights unchanged) are expected; allow up to a fraction of steps.
        limit = max_nonfinite_frac * max(int(steps), 1)
        res.append(("finite_steps", nf <= limit, "non-finite steps: %s (allowed <= %.1f, %.1f %% of %d "
                    "steps, AMP loss-scale overflows)" % (nf, limit, 100.0 * max_nonfinite_frac, int(steps))))
    else:
        res.append(("finite_steps", nf == 0, "non-finite steps: %s" % nf))
    if max_window_nonfinite_frac > 0:
        res += _check_nonfinite_windows(recs, tasks, max_window_nonfinite_frac,
                                        nonfinite_window_steps, nonfinite_from_step)
    res += _loss_scale_info(recs)
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
    args = _load_config(run_dir).get("args") or {}
    if expect_cascade or expect_cascade_stages == 2:
        res += _check_cascade(run_dir, tasks, val, recs, stages=expect_cascade_stages)
    if expect_bn_recal:
        res += _check_bn_recal(recs)
    if expect_skip:
        res += _check_skip(recs, _load_config(run_dir))
    if grad_balance_rule:
        res += _check_grad_balance(recs, tasks, grad_balance_from_step, args)
    if expect_min_pairs not in (None, "none"):
        res += _check_min_pairs(recs, tasks, args, expect_min_pairs)
    if memory_budget_mb > 0:
        res += _gpu_peak(recs, memory_budget_mb)
    if project_steps > 0:
        res += _step_time(recs, tasks, args, project_steps, project_val_every)
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
    ap.add_argument("--max-nonfinite-frac", type=float, default=0.0,
                    help="allowed fraction of non-finite (skipped) steps; 0 = none allowed. Use e.g. "
                    "0.06 with AMP, where the dynamic loss scale overflows a few times early on")
    ap.add_argument("--expect-cascade-stages", type=int, default=1, choices=[1, 2],
                    help="milestone 9 phase 2: 2 = level-1 3-stage cascade (implies --expect-cascade; "
                    "source level 2, cascade.stages [4, 2], val_<task>_frozen4 baselines); 1 = milestone 8")
    ap.add_argument("--expect-min-pairs", default="none", choices=list(MIN_PAIRS_CHOICES),
                    help="milestone 9 phase 2: config.json args.min_pairs equals this; with n every logged "
                    "train_<task>/n_pairs equals the expected pair count; none = not checked")
    ap.add_argument("--memory-budget-mb", type=float, default=0.0,
                    help="INFO gpu_peak line: max logged gpu_peak_mb vs this budget (e.g. 36864 = A100 "
                    "36 GiB); 0 = off, never a failure")
    ap.add_argument("--project-steps", type=int, default=0,
                    help="INFO step_time line: mean step times and the projected wall time of a run with "
                    "this many steps; 0 = off, never a failure")
    ap.add_argument("--project-val-every", type=int, default=0,
                    help="validation cadence of the projected run in the step_time line (e.g. 500 for "
                    "the long run when analysing a VAL_EVERY 100 smoke); 0 = the analysed run's val_every")
    ap.add_argument("--max-window-nonfinite-frac", type=float, default=0.0,
                    help="nonfinite-fixes: FAIL if any task's non-finite fraction in a window of "
                    "--nonfinite-window-steps steps (after --nonfinite-from-step) exceeds this; "
                    "0 = off (default). The scripts pass 0.05")
    ap.add_argument("--nonfinite-window-steps", type=int, default=500,
                    help="window length in steps for --max-window-nonfinite-frac (default %(default)s)")
    ap.add_argument("--nonfinite-from-step", type=int, default=100,
                    help="--max-window-nonfinite-frac ignores log records with step <= this (early AMP "
                    "loss-scale overflows; default %(default)s)")
    a = ap.parse_args(argv)
    if a.project_val_every < 0:
        ap.error("--project-val-every must be >= 0, got %d" % a.project_val_every)
    if not (math.isfinite(a.max_window_nonfinite_frac) and 0 <= a.max_window_nonfinite_frac <= 1):
        ap.error("--max-window-nonfinite-frac must be in [0, 1], got %r" % a.max_window_nonfinite_frac)
    if a.nonfinite_window_steps < 1 or a.nonfinite_from_step < 0:
        ap.error("--nonfinite-window-steps must be >= 1 and --nonfinite-from-step >= 0")
    tasks = [t for t in a.tasks.split(",") if t]
    unknown = [t for t in tasks if t not in LINEAR_TASKS + (DEFORM_TASK,)]
    if unknown:
        ap.error("unknown task(s) %s" % unknown)
    ok, res = check(a.run_dir, tasks, a.steps, a.expect_best == "true",
                    expect_init_weights=a.expect_init_weights, expect_enc_grad=a.expect_enc_grad == "true",
                    expect_cascade=a.expect_cascade == "true", expect_bn_recal=a.expect_bn_recal == "true",
                    expect_skip=a.expect_skip == "true", grad_balance_rule=a.grad_balance_rule == "true",
                    grad_balance_from_step=a.grad_balance_from_step,
                    max_nonfinite_frac=a.max_nonfinite_frac, expect_cascade_stages=a.expect_cascade_stages,
                    expect_min_pairs=a.expect_min_pairs, memory_budget_mb=a.memory_budget_mb,
                    project_steps=a.project_steps, project_val_every=a.project_val_every,
                    max_window_nonfinite_frac=a.max_window_nonfinite_frac,
                    nonfinite_window_steps=a.nonfinite_window_steps,
                    nonfinite_from_step=a.nonfinite_from_step)
    for name, passed, detail in res:
        label = "INFO" if passed and detail.startswith("INFO") else ("PASS" if passed else "FAIL")
        print("%s %-22s %s" % (label, name, detail))
    print("check_mixed_run: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
