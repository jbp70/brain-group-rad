"""Train one pyramid level of the groupwise registration model (rigid / affine / deform).

Examples (inside tf10_nv.sif; see slurm/train_debug.sh and tests/gpu/run_overfit_glacier.sh)::

    python main.py --data <export_root> --level 4 --group-size 32 --steps 2000 --results <dir>
    python main.py --data <export_root> --dry-run              # build, 2 steps + 1 val group
    python main.py --overfit-one-group --data <export_root> --group-size 8
    python main.py --overfit-one-group --overfit-synthetic     # no data, no DALI
    python main.py --overfit-one-group --overfit-synthetic --task-ratio rigid:1,affine:1
    python main.py --data <export_root> --init-weights <other run>/ckpt/best   # warm start
    python main.py --data <export_root> --task-ratio deform:1 --group-size 16   # deformable
    python main.py --data <export_root> --task-ratio rigid:1,affine:1,deform:2  # three tasks
    python main.py --overfit-one-group --overfit-synthetic --task-ratio deform:1 --group-size 4

Deformable runs (milestone 6): the deform decoder is built when 'deform' is in --task-ratio;
the deform pipeline reads images + train18 labels (`DeformPipeline`), validation uses
`DeformValidationSet` (val images + labels); control files stay linear-only. With deform and a
linear task, ``ckpt/best`` keeps the linear criterion and ``ckpt/best_deform`` tracks
val_deform/pair_dice_loss.

Order: parse args (no TF) -> DALI check (unless overfit mode) -> configure_tf_precision() (TF32
off, memory growth) -> seeds -> mixed_float16 / XLA flags -> data -> model -> optimizer ->
run dir, logger, config.json, checkpoint (resume) -> Trainer.fit.

Exit code: 0 on success; in overfit mode 1 if the threshold was not reached.
"""
import json
import os
import random
import sys

from runtime import args as A

DALI_MISSING_MSG = ("main.py: nvidia.dali is not importable ({err}). The DALI training pipeline "
                    "needs it: run inside tf10_nv.sif, or use --overfit-one-group (TF-side warp, "
                    "no DALI needed).")


def _build_overfit_data(args, level, tf_mods):
    """One fixed group per trained task (same real or synthetic images), rendered TF-side.

    rigid: fixed pure-rigid transforms (--rigid-rotation-deg / --rigid-translation-mm, no
    scale/shear nuisance), drawn from default_rng(seed) right after the images (as before
    milestone 5). affine: fixed full affine transforms from the --affine-* ranges, drawn from
    default_rng([seed, 202]), so mixed runs alternate between two different fixed groups.
    deform: --deform-group-size images with labels, no transform (A = I): synthetic = the label
    phantom deformed by random diffeomorphisms (`fd.synthetic_deform_group(level, n, seed)`),
    real = train images + train18 labels (inter-subject; `load_level_images_and_labels`);
    validated on the same group (`DeformValidationSet.from_group`)."""
    import numpy as np
    fd, P, tfm, postprocess, dm = tf_mods
    n = args.group_size
    rng = np.random.default_rng(args.seed)
    shape, spacing = P.LEVEL_SHAPE[level], P.LEVEL_SPACING[level]
    if args.overfit_synthetic:
        vol, mask = fd.smooth_phantom(shape, rng, sigma=4.0 * 4 / level)
        base = vol
        S = tfm.brain_second_moment(mask, spacing)
        source_desc = "synthetic phantom"
    else:
        base, files = fd.load_level_images(args.data, level, "train", n=n, seed=args.seed)
        S = postprocess.brain_S_matrix(level, os.path.join(args.data, dm.TEMPLATE_DIR))
        source_desc = "train images: " + ", ".join(files)
    sources, val_sets = {}, {}
    for task in A.linear_tasks(args):
        if task == "rigid":
            ranges = {"rotation_deg": args.rigid_rotation_deg,
                      "translation_mm": args.rigid_translation_mm, "log_scale": 0.0, "shear": 0.0}
            T, T_target = fd.random_linear_transforms(rng, n, "rigid", ranges, nuisance=False)
        else:
            ranges = A.ranges_from_args(args)[task]
            T, T_target = fd.random_linear_transforms(
                np.random.default_rng([int(args.seed), fd.VAL_TASK_SEED[task]]), n, task, ranges)
        rendered = fd.render_group(base, T, spacing)
        sources[task] = fd.FixedGroupSource(fd.to_dali_batch(rendered, T, T_target, task, level))
        val_sets[task] = fd.ValidationSet.from_group(rendered, T, T_target, task, level,
                                                     args.target_iters)
    if A.DEFORM_TASK in args.task_ratio_parsed:
        nd = args.deform_group_size
        if args.overfit_synthetic:
            grp = fd.synthetic_deform_group(level, nd, args.seed,
                                            integration_steps=args.integration_steps)
            images, labels = grp["images"], grp["labels"]
            source_desc += "; deform: synthetic label phantom, %d diffeomorphisms" % nd
        else:
            images, labels, files, _ = fd.load_level_images_and_labels(args.data, level, "train",
                                                                       n=nd, seed=args.seed)
            source_desc += "; deform train images + train18 labels: " + ", ".join(files)
        # The residual affine is a DALI augmentation: the fixed group gets identity matrices.
        T = np.tile(np.eye(4), (nd, 1, 1)) if args.deform_residual_affine else None
        sources[A.DEFORM_TASK] = fd.FixedGroupSource(fd.to_deform_batch(images, labels, T, level))
        val_sets[A.DEFORM_TASK] = fd.DeformValidationSet.from_group(images, labels, level)
    return sources, val_sets, S, source_desc


def _build_dali_data(args, level, tf_mods, start_step=0):
    """DALI sources, validation sets, S, description and the task controls ({} unless
    --curriculum / --global-transform: then <results>/<run-name>/control/<task>.npy is written
    with the values of `start_step` (the resumed step, 0 for a new run) before the pipelines
    start, so the first prefetched batches of a resumed run already use its ranges and G)."""
    fd, P, tfm, postprocess, dm = tf_mods
    ranges = A.ranges_from_args(args)
    aug = P.IntensityAugConfig() if args.intensity_aug else P.IntensityAugConfig.off()
    controls = A.task_controls_from_args(args)
    control_kw = {}
    if controls:
        control_kw = {"control_dir": os.path.join(args.results, args.run_name, "control"),
                      "control_init": {t: c.values_at(int(start_step)) for t, c in controls.items()}}
    deform_kw = {}
    if A.DEFORM_TASK in args.task_ratio_parsed:
        deform_kw = {"deform_batch_size": args.deform_group_size,
                     "deform_residual_affine": args.deform_residual_affine}
    data = dm.LinearDataModule(args.data, level, args.group_size, seed=args.seed, split="train",
                               rigid_ranges=ranges["rigid"], affine_ranges=ranges["affine"],
                               intensity_aug=aug, num_threads=args.num_threads,
                               prefetch_queue_depth=args.prefetch, shuffle_mode=args.shuffle_mode,
                               **dict(control_kw, **deform_kw))
    for t, c in controls.items():
        c.writer = data.control_writers[t]
    sources = {t: data.iterator(t) for t in args.task_ratio_parsed}
    n_groups = 1 if args.dry_run else args.val_groups
    lin = A.linear_tasks(args)
    val_sets, n_val = {}, 0
    if lin:
        val_images, _ = fd.load_level_images(args.data, level, "val")
        n_val = val_images.shape[0]
        nv = args.val_group_size or min(args.group_size, val_images.shape[0])
        g_val = ({"rotation_deg": args.global_rot_max_deg, "translation_mm": args.global_trans_max_mm}
                 if args.global_transform else None)
        # Seeded by (val_seed, task, group index), not by the task's position in the ratio.
        val_sets = fd.build_validation_sets(val_images, lin, level, nv, n_groups,
                                            ranges, seed=args.val_seed, target_iters=args.target_iters,
                                            global_transform=g_val)
    if A.DEFORM_TASK in args.task_ratio_parsed:
        val_images_d, val_labels, _, _ = fd.load_level_images_and_labels(args.data, level, "val")
        n_val = max(n_val, val_images_d.shape[0])
        nvd = args.val_group_size or min(args.deform_group_size, val_images_d.shape[0])
        val_sets[A.DEFORM_TASK] = fd.DeformValidationSet.build(val_images_d, val_labels, level, nvd,
                                                               n_groups, seed=args.val_seed)
    S = postprocess.brain_S_matrix(level, data.template_dir)
    desc = "DALI %d train files, %d val images" % (len(data.files), n_val)
    return sources, val_sets, S, desc, controls


def main(argv=None):
    args = A.parse_args(argv)
    if not args.overfit_one_group:
        from data_loading import dali_loader
        if not dali_loader.DALI_AVAILABLE:
            raise SystemExit(DALI_MISSING_MSG.format(err=dali_loader._DALI_IMPORT_ERROR))

    import numpy as np
    import tensorflow as tf
    from models.precision import configure_tf_precision
    configure_tf_precision(tf32=False, memory_growth=not args.gpu_memory_limit_mb)   # FIRST
    if args.gpu_memory_limit_mb:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.set_logical_device_configuration(
                gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=int(args.gpu_memory_limit_mb))])

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    if args.amp:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if args.xla:
        tf.config.optimizer.set_jit(True)

    from data_loading import data_module as dm
    from data_loading import params as P
    from data_loading import postprocess
    from models import transforms as tfm
    from models.group_net import GroupRegLevelModel
    from runtime import checkpoint as C
    from runtime import fixed_data as fd
    from runtime import optimizers as O
    from runtime.logging import RunLogger, make_run_dir
    from runtime.run import Trainer

    level = args.level
    tf_mods = (fd, P, tfm, postprocess, dm)
    controls = {}
    if args.init_weights and args.resume and C.peek_last_step(
            os.path.join(args.results, args.run_name, "ckpt", "last")) > 0:
        raise SystemExit("main.py: --init-weights starts a NEW run at step 0, but %s already has a "
                         "checkpoint and --resume is on; use a new --run-name (or --resume false)"
                         % os.path.join(args.results, args.run_name))
    if args.overfit_one_group:
        sources, val_sets, S, data_desc = _build_overfit_data(args, level, tf_mods)
    else:
        # Resume: the control files must start at the resumed step (review-curriculum issue 5),
        # so the step is read from the checkpoint before the pipelines are built.
        start_step = 0
        if args.resume:
            start_step = C.peek_last_step(os.path.join(args.results, args.run_name, "ckpt", "last"))
        sources, val_sets, S, data_desc, controls = _build_dali_data(args, level, tf_mods, start_step)

    model_cfg = A.model_config_from_args(args)
    # The variables do not depend on N; with deform the build pass (rigid + deform forward) uses
    # the smaller of the two group sizes, so a large linear N does not size the deform forward.
    n_build = args.group_size
    if A.DEFORM_TASK in args.task_ratio_parsed:
        n_build = min(n_build, args.deform_group_size)
    model = GroupRegLevelModel(level, model_cfg, gradient_checkpointing=args.gradient_checkpointing
                               ).build_model(n_build)
    counts = model.count_params_by_part()
    init_info = None
    if args.init_weights:
        # Warm start (model variables only; fresh optimizer, step 0). Resume was excluded above.
        prefix, src_cfg = C.load_model_weights(model, model_cfg, args.init_weights)
        init_info = {"checkpoint": os.path.abspath(prefix),
                     "source_config": os.path.abspath(src_cfg) if src_cfg else None}
    optimizer = O.make_optimizer(args.optimizer, lr=args.lr, weight_decay=args.weight_decay,
                                 momentum=args.momentum, dampening=args.dampening,
                                 nesterov=args.nesterov, decoupled=args.decoupled,
                                 clipnorm=args.clipnorm, exclude_wd=args.exclude_from_weight_decay,
                                 amp=args.amp, jit_compile=args.xla)
    train_cfg = A.train_config_from_args(args)
    if args.dry_run:
        train_cfg.val_every = 0
        train_cfg.ckpt_every = 0

    paths = make_run_dir(args.results, args.run_name)
    if args.val_png:
        train_cfg.val_png_dir = os.path.join(paths["root"], "val_png")
    logger = RunLogger(paths, backend=args.log_backend, tensorboard=True, quiet=args.quiet)
    trainer = Trainer(model, optimizer, sources, S, train_cfg, logger=logger, val_sets=val_sets,
                      controls=controls)
    # Deform + linear: a second best file on the deform metric (ckpt/best keeps the linear rule).
    extra_best = ({name: os.path.join(paths["root"], "ckpt", "best_%s" % name) for name in trainer.extra_best}
                  or None)
    trainer.checkpoint = C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], model, optimizer,
                                         trainer.step, extra=trainer.checkpoint_variables(),
                                         extra_best=extra_best)
    resumed = False
    if init_info is not None:
        if not args.quiet:
            print("init weights from %s (source config %s)" % (init_info["checkpoint"],
                                                               init_info["source_config"]))
    elif args.resume and trainer.checkpoint.latest_last:
        C.check_config_compatible(paths["config"], model_cfg)
        resumed = trainer.checkpoint.restore_last()
    C.write_config(paths["config"], model_cfg, A.args_to_dict(args),
                   extra={"param_counts": dict(counts), "data": data_desc, "init_weights": init_info})
    logger.log_params(dict(A.args_to_dict(args), run_dir=paths["root"], resumed=resumed,
                           resumed_step=int(trainer.step.numpy()), data=data_desc,
                           trainable_params=counts["trainable"],
                           non_trainable_params=counts["non_trainable"],
                           tf32=bool(tf.config.experimental.tensor_float_32_execution_enabled())))
    if not args.quiet:
        print("run dir: %s | resumed: %s at step %d | data: %s" % (
            paths["root"], resumed, int(trainer.step.numpy()), data_desc[:200]))
        print("parameters: " + ", ".join("%s=%d" % kv for kv in counts.items()))

    if args.dry_run:
        summary = trainer.fit(2)
        val = trainer.validate(max_groups=1)
        logger.log_metrics(int(trainer.step.numpy()), val)
        trainer.checkpoint.save_last()
        logger.close()
        ok = all(h["finite"] for h in trainer.history[-2:]) and all(
            np.isfinite(v) for v in val.values())
        print("dry run: steps %d -> %d, trace counts %s, finite %s" % (
            summary["start_step"], summary["end_step"], summary["trace_counts"], ok))
        return 0 if ok else 1

    deform = A.DEFORM_TASK in args.task_ratio_parsed
    initial_val = None
    if args.overfit_one_group and deform and val_sets:
        # The validation before training (identity at init): baseline of the deform overfit.
        initial_val = trainer.validate()
    summary = trainer.fit()
    if not args.overfit_one_group:
        # Trace counts etc. of this fit() call (tests/gpu/check_mixed_run.py reads them).
        from runtime.logging import to_jsonable
        with open(os.path.join(paths["root"], "fit_summary.json"), "w") as f:
            json.dump(to_jsonable({k: v for k, v in summary.items() if k != "last_val"}), f, indent=1)
    final_val = {}
    if args.overfit_one_group and val_sets:
        # Eval mode is the source of truth: validate once more at the stopping step (fit already
        # did when the PNGs are on).
        stop = int(trainer.step.numpy())
        if trainer.val_history and trainer.val_history[-1]["step"] == stop:
            final_val = {k: v for k, v in trainer.val_history[-1].items() if k != "step"}
        else:
            final_val = trainer.validate()
            logger.log_metrics(stop, final_val)
    logger.close()
    if args.overfit_one_group:
        tasks = list(args.task_ratio_parsed)
        lin = A.linear_tasks(args)
        curve = [[h["step"], h["center_rms_mm"], h["pair_rms_mm"]] for h in trainer.history
                 if h["task"] != A.DEFORM_TASK]
        # val_curve rows: step, then (eval, train-mode) center RMS per linear task (one linear
        # task: the milestone-4 layout [step, eval, train mode]).
        val_cols = ["step"]
        for t in lin:
            val_cols += ["%s_eval_center_rms_mm" % t if len(lin) > 1 else "eval_center_rms_mm",
                         "%s_trainmode_center_rms_mm" % t if len(lin) > 1 else "trainmode_center_rms_mm"]

        def val_row(step, v):
            row = [step]
            for t in lin:
                row += [v.get("val_%s/center_rms_mm" % t), v.get("val_%s_trainmode/center_rms_mm" % t)]
            return row

        val_curve = [val_row(v["step"], v) for v in trainer.val_history]
        if final_val and not (trainer.val_history and trainer.val_history[-1]["step"] == stop):
            val_curve.append(val_row(int(trainer.step.numpy()), final_val))
        out = {"reached": bool(trainer.reached), "threshold_mm": args.overfit_threshold_mm,
               "window": train_cfg.overfit_window, "steps_run": len(trainer.history),
               "end_step": summary["end_step"], "tasks": tasks,
               "trace_counts": summary["trace_counts"],
               "best_window_center_rms_mm": summary["best_window_center_rms_mm"],
               "window_center_rms_by_task": summary["window_center_rms_by_task"],
               "task_curve": [h["task"] for h in trainer.history],
               "final_center_rms_mm": curve[-1][1] if curve else None,
               "optimizer": args.optimizer, "lr": args.lr, "schedule": args.schedule,
               "group_size": args.group_size, "level": level, "synthetic": args.overfit_synthetic,
               "model_config": model_cfg.to_dict(),
               "median_compute_ms": float(np.median([h["compute_ms"] for h in trainer.history[1:]]))
               if len(trainer.history) > 1 else None,
               "gpu_peak_mb": _peak_mb(tf),
               "curve_columns": ["step", "center_rms_mm", "pair_rms_mm"], "curve": curve,
               "val": trainer.val_history, "final_val": final_val,
               "val_curve_columns": val_cols, "val_curve": val_curve}
        if deform:
            out.update(_deform_overfit_summary(args, trainer, summary, final_val, initial_val))
        from runtime.logging import to_jsonable
        with open(os.path.join(paths["root"], "overfit_summary.json"), "w") as f:
            json.dump(to_jsonable(out), f, indent=1)
        if lin:
            per_task = "; ".join(
                "%s eval %.4f mm, train mode %.4f mm" % (
                    t, final_val.get("val_%s/center_rms_mm" % t, float("nan")),
                    final_val.get("val_%s_trainmode/center_rms_mm" % t, float("nan"))) for t in lin)
            print("overfit: reached=%s best %d-step mean center RMS (worst task) %.4f mm after %d steps; "
                  "final val center RMS: %s" % (
                      trainer.reached, train_cfg.overfit_window,
                      summary["best_window_center_rms_mm"] or float("nan"), len(trainer.history), per_task))
        if deform:
            wd = summary["window_deform"] or {}
            print("overfit deform: reached=%s after %d steps; %d-step window soft template Dice %.4f "
                  "(threshold %g), fold %.5f (max %g); final val template Dice %.4f, pair Dice %.4f "
                  "(identity %.4f / %.4f), neg J %.4f %%" % (
                      trainer.reached, len(trainer.history), train_cfg.overfit_window,
                      wd.get("soft_dice_tmpl", float("nan")), args.overfit_dice_threshold,
                      wd.get("fold_frac", float("nan")), args.overfit_fold_max,
                      final_val.get("val_deform/template_dice", float("nan")),
                      final_val.get("val_deform/pair_dice", float("nan")),
                      final_val.get("val_deform/template_dice_identity", float("nan")),
                      final_val.get("val_deform/pair_dice_identity", float("nan")),
                      final_val.get("val_deform/neg_jac_pct", float("nan"))))
        return 0 if trainer.reached else 1
    return 0


def _deform_overfit_summary(args, trainer, summary, final_val, initial_val):
    """Extra overfit_summary.json keys of a run that trains deform (the linear layout is kept)."""
    deform_curve = [[h["step"], h["loss"], h["soft_dice_tmpl"], h["soft_dice_pair"], h["diag_fold_frac"]]
                    for h in trainer.history if h["task"] == A.DEFORM_TASK]

    def row(step, v):
        return [step, v.get("val_deform/template_dice"), v.get("val_deform/pair_dice"),
                v.get("val_deform/neg_jac_pct")]

    val_curve = [row(v["step"], v) for v in trainer.val_history]
    stop = int(trainer.step.numpy())
    if final_val and not (trainer.val_history and trainer.val_history[-1]["step"] == stop):
        val_curve.append(row(stop, final_val))
    return {"dice_threshold": args.overfit_dice_threshold, "fold_max": args.overfit_fold_max,
            "deform_group_size": args.deform_group_size,
            "window_deform": summary["window_deform"],
            "best_window_soft_dice_tmpl": summary["best_window_soft_dice_tmpl"],
            "deform_curve_columns": ["step", "loss", "soft_dice_tmpl", "soft_dice_pair", "fold_frac"],
            "deform_curve": deform_curve,
            "deform_val_curve_columns": ["step", "template_dice", "pair_dice", "neg_jac_pct"],
            "deform_val_curve": val_curve, "initial_val": initial_val}


def _peak_mb(tf):
    try:
        if tf.config.list_physical_devices("GPU"):
            return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2.0 ** 20
    except (ValueError, RuntimeError):
        pass
    return None


if __name__ == "__main__":
    sys.exit(main())
