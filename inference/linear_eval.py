"""``predict.py linear-eval`` (TTA design 7.3, plan D9): rigid / affine errors with ground truth.

Mindboggle (and any processed dataset) is MNI-affine aligned, so it has no linear ground truth of
its own. This command renders every processed subject with a seeded synthetic misalignment T_i
(`runtime.fixed_data.random_linear_transforms`, the training validation ranges
`params.DEFAULT_LINEAR_RANGES[task]`), runs the streamed linear cascade of the checkpoint on the
rendered group (``--tta none`` or ``flip8``) and scores A against the barycentric targets
A* = `transforms.barycentric_targets(T_target, 10)` with `runtime.metrics.group_errors`
(template-space point RMS over the group mask m = mean_i 1[I_i > 0] >= 0.5, rotation, translation,
affine scale / shear, all ordered pairs, cycles).

Ground-truth caveat (review finding 9): the targets treat the un-perturbed processed subjects as
mutually aligned. They are only MNI-affine aligned, so their residual misalignment adds an error
floor; it is the same in both TTA arms (same --seed = bit-identical rendered group), so the paired
comparison stays fair, but absolute errors are upper bounds of the model error (summary.json
``note``).

Outputs in --out: per_subject.csv, per_pair.csv, summary.json (written last, also on failure),
tta_spread.csv with --tta flip8. Exit codes as ``run``: 0 ok, 2 input (fewer than 2 subjects, grid
not the level-1 shape, invalid flags / checkpoint), 3 non-finite A (CSVs not written), 1 other.
"""
import csv
import logging
import os
import time

from inference import cli as CLI
from inference import predict as PR

logger = logging.getLogger(__name__)
PredictError = PR.PredictError

LINEAR_EVAL_KEYS = (
    "status", "message", "command", "git_commit", "git_commit_source", "configure_device",
    "checkpoint", "model_config", "precision", "task", "seed", "ranges", "n_subjects", "subjects",
    "subject_dirs", "levels", "tta", "tta_record", "store", "chunks", "timing_s", "gpu_peak_mb",
    "means", "p95", "nonfinite", "note", "outputs", "grid", "input_range", "warnings",
    "visible_devices", "device")
NOTE = "errors include the residual alignment of the processed subjects"
SUMMARY_JSON, PER_SUBJECT_CSV, PER_PAIR_CSV = "summary.json", "per_subject.csv", "per_pair.csv"
SUBJECT_ERRORS = {"rigid": ("rms_mm", "rot_deg", "trans_mm"),
                  "affine": ("rms_mm", "rot_deg", "trans_mm", "scale_pct", "shear_pct")}


def _value(v):
    return "%.9g" % float(v)


def _write_csv(path, header, rows):
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def _stat(arr, fn):
    import numpy as np
    a = np.asarray(arr, np.float64).reshape(-1)
    return float(fn(a)) if a.size else None


def group_mask(images):
    """Template-space group mask m = (mean_i 1[I_i > 0]) >= 0.5 of images [N, X, Y, Z] (bool)."""
    import numpy as np
    count = np.zeros(images.shape[1:], np.int32)
    for k in range(images.shape[0]):
        count += images[k] > 0
    return 2 * count >= images.shape[0]


def _run(args, configure_device, rec):
    timing = rec["timing_s"]
    if configure_device:
        PR.setup_device(args)
    import numpy as np
    import tensorflow as tf
    from data_loading import params as P
    from inference import runner as RN
    from inference import store as SToR
    from models import transforms as tfm
    from runtime import fixed_data as FD
    from runtime import metrics as MT
    task = args.task
    rec["visible_devices"] = [d.name for d in tf.config.get_visible_devices()]
    peek = PR._peek_config(args.checkpoint)
    plan = PR._setup_levels(args, rec, peek[0])
    tta_cfg = PR._setup_tta(args, rec, plan, peek[0])
    t = time.time()
    subs, images, labels, shape, aff, spacing = PR._read_subjects(args, rec)
    del labels
    timing["read"] = time.time() - t
    n = len(subs)
    if n < 2:
        raise PredictError(CLI.EXIT_INPUT, "linear-eval needs at least 2 subjects, got %d" % n)
    if tuple(shape) != tuple(P.LEVEL_SHAPE[1]):
        raise PredictError(CLI.EXIT_INPUT, "grid %s is not the level-1 shape %s"
                           % (tuple(shape), tuple(P.LEVEL_SHAPE[1])))
    try:
        S = tfm.brain_second_moment(group_mask(images), 1.0)
    except ValueError as e:
        raise PredictError(CLI.EXIT_INPUT, "group mask: %s" % e)
    rec["ranges"] = dict(P.DEFAULT_LINEAR_RANGES[task])
    T, T_target = FD.random_linear_transforms(np.random.default_rng(int(args.seed)), n, task)
    t = time.time()
    rendered = FD.render_group(images[..., None], T, P.LEVEL_SPACING[1], chunk=2)[..., 0]
    del images
    timing["render"] = time.time() - t
    A_star = tfm.barycentric_targets(tf.constant(T_target), 10).numpy()
    t = time.time()
    loaded = PR._load(args)
    timing["load"] = time.time() - t
    rec.update(checkpoint=PR._checkpoint_record(loaded), model_config=loaded.model_cfg.to_dict(),
               precision=loaded.precision)
    tdt = "float16" if loaded.precision["trainable"] == "mixed_float16" else "float32"
    est = SToR.estimate_store_gb(loaded.level, loaded.model_cfg, n, task, tdt)
    store_dir = args.store_dir or os.environ.get("TMPDIR") or os.path.join(args.out, "_stream_tmp")
    store = SToR.choose_store(args.store, est, store_dir)
    rec["store"] = {"kind": store.kind, "dir": getattr(store, "directory", None),
                    "estimate_gb": est, "choice": store.choice}
    runner = RN.StreamedRunner(loaded, store, chunks={"l1": args.chunk_l1, "l2": args.chunk_l2,
                                                      "l4": args.chunk_l4}, plan=plan, tta=tta_cfg)
    os.makedirs(args.out, exist_ok=True)
    logger.info("linear-eval: %d subjects, task %s, seed %d, tta %s, level-%d checkpoint step %s",
                n, task, int(args.seed), args.tta, loaded.level, loaded.step)
    t = time.time()
    try:
        summary = runner.run(rendered, task)
    finally:
        rec["store"]["peak_store_gb"] = store.peak_nbytes() / 1e9
        store.close()
    timing["model"] = time.time() - t
    del rendered
    for key, sec in sorted(summary["timing_s"].items()):
        timing[key] = sec
    rec["chunks"] = {"l1": summary["chunk_sizes"].get("l1"), "l2": summary["chunk_sizes"].get("l2"),
                     "l4": summary["chunk_sizes"].get("l4"), "capped": summary["capped"],
                     "bounds": summary["chunks"]}
    rec["gpu_peak_mb"] = summary["gpu_peak_mb"]
    A = np.asarray(summary["A"], np.float64)
    bad = int(A.size - np.count_nonzero(np.isfinite(A)))
    rec["nonfinite"] = {"A": bad}
    outputs = []
    tta_csv = PR._tta_finish(rec, summary, args, subs)
    if tta_csv is not None:
        outputs.append(tta_csv)
    if bad:
        rec["outputs"] = outputs + [SUMMARY_JSON]
        rec["status"] = "nonfinite"
        rec["message"] = "%d non-finite values in A; per_subject.csv / per_pair.csv not written" % bad
        bad_flips = {k: v for k, v in ((summary.get("tta") or {}).get("nonfinite_flips") or {}).items() if v}
        if bad_flips:
            rec["message"] += "; TTA flips with non-finite values (level/task: flip indices): %s" % bad_flips
        logger.error("%s", rec["message"])
        return CLI.EXIT_NONFINITE
    t = time.time()
    errs = MT.group_errors(A, A_star, T_target, S, affine=task == "affine")
    rec["means"] = {k: _stat(v, np.mean) for k, v in sorted(errs.items())}
    rec["p95"] = {k: _stat(v, lambda a: np.percentile(a, 95.0)) for k, v in sorted(errs.items())}
    cols = SUBJECT_ERRORS[task]
    i_idx, j_idx = MT.all_ordered_pairs(n)
    pair_rms = np.asarray(errs["pair_rms_mm"], np.float64)
    rows = []
    for k, s in enumerate(subs):
        rows.append([s["subject"], s["dir"]] + [_value(errs["center_" + c][k]) for c in cols]
                    + [_value(np.mean(pair_rms[i_idx == k]))])
    _write_csv(os.path.join(args.out, PER_SUBJECT_CSV),
               ["subject", "subject_dir"] + ["center_" + c for c in cols] + ["pair_rms_mm_mean"], rows)
    rows = [[subs[int(i)]["subject"], subs[int(j)]["subject"]]
            + [_value(errs["pair_" + c][p]) for c in cols] for p, (i, j) in enumerate(zip(i_idx, j_idx))]
    _write_csv(os.path.join(args.out, PER_PAIR_CSV), ["moving", "fixed"] + ["pair_" + c for c in cols], rows)
    outputs = [PER_SUBJECT_CSV, PER_PAIR_CSV] + outputs
    timing["metrics"] = time.time() - t
    rec["outputs"] = outputs + [SUMMARY_JSON]
    m = rec["means"]
    logger.info("linear-eval %s (tta %s): center rms %.3f mm, pair rms %.3f mm, rot %.3f deg, "
                "trans %.3f mm (%s)", task, args.tta, m["center_rms_mm"], m["pair_rms_mm"],
                m["center_rot_deg"], m["center_trans_mm"], NOTE)
    rec["status"] = "ok"
    return CLI.EXIT_OK


def run_linear_eval(args, configure_device=True):
    """``predict.py linear-eval`` -> exit code; <out>/summary.json written last (also on failure)."""
    rec = {k: None for k in LINEAR_EVAL_KEYS}
    rec.update(status="error", command=PR._command(args), configure_device=bool(configure_device),
               task=args.task, seed=int(args.seed), tta=getattr(args, "tta", None), device=args.device,
               note=NOTE, timing_s={}, warnings=[])
    return PR._command_wrapper(_run, args, configure_device, rec,
                               os.path.join(args.out, SUMMARY_JSON), "linear-eval summary.json")
