"""Train one pyramid level of the groupwise registration model (milestone 4: rigid / affine).

Examples (inside tf10_nv.sif; see slurm/train_debug.sh and tests/gpu/run_overfit_glacier.sh)::

    python main.py --data <export_root> --level 4 --group-size 32 --steps 2000 --results <dir>
    python main.py --data <export_root> --dry-run              # build, 2 steps + 1 val group
    python main.py --overfit-one-group --data <export_root> --group-size 8
    python main.py --overfit-one-group --overfit-synthetic     # no data, no DALI
    python main.py --overfit-one-group --overfit-synthetic --task-ratio rigid:1,affine:1
    python main.py --data <export_root> --init-weights <other run>/ckpt/best   # warm start

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
    default_rng([seed, 202]), so mixed runs alternate between two different fixed groups."""
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
    for task in args.task_ratio_parsed:
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
    data = dm.LinearDataModule(args.data, level, args.group_size, seed=args.seed, split="train",
                               rigid_ranges=ranges["rigid"], affine_ranges=ranges["affine"],
                               intensity_aug=aug, num_threads=args.num_threads,
                               prefetch_queue_depth=args.prefetch, shuffle_mode=args.shuffle_mode,
                               **control_kw)
    for t, c in controls.items():
        c.writer = data.control_writers[t]
    sources = {t: data.iterator(t) for t in args.task_ratio_parsed}
    val_images, _ = fd.load_level_images(args.data, level, "val")
    nv = args.val_group_size or min(args.group_size, val_images.shape[0])
    n_groups = 1 if args.dry_run else args.val_groups
    g_val = ({"rotation_deg": args.global_rot_max_deg, "translation_mm": args.global_trans_max_mm}
             if args.global_transform else None)
    # Seeded by (val_seed, task, group index), not by the task's position in the ratio.
    val_sets = fd.build_validation_sets(val_images, list(args.task_ratio_parsed), level, nv, n_groups,
                                        ranges, seed=args.val_seed, target_iters=args.target_iters,
                                        global_transform=g_val)
    S = postprocess.brain_S_matrix(level, data.template_dir)
    desc = "DALI %d train files, %d val images" % (len(data.files), val_images.shape[0])
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
    model = GroupRegLevelModel(level, model_cfg).build_model(args.group_size)
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
    trainer.checkpoint = C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], model, optimizer,
                                         trainer.step, extra=trainer.checkpoint_variables())
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
        curve = [[h["step"], h["center_rms_mm"], h["pair_rms_mm"]] for h in trainer.history]
        # val_curve rows: step, then (eval, train-mode) center RMS per task (rigid only: the
        # milestone-4 layout [step, eval, train mode]).
        val_cols = ["step"]
        for t in tasks:
            val_cols += ["%s_eval_center_rms_mm" % t if len(tasks) > 1 else "eval_center_rms_mm",
                         "%s_trainmode_center_rms_mm" % t if len(tasks) > 1 else "trainmode_center_rms_mm"]

        def val_row(step, v):
            row = [step]
            for t in tasks:
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
        from runtime.logging import to_jsonable
        with open(os.path.join(paths["root"], "overfit_summary.json"), "w") as f:
            json.dump(to_jsonable(out), f, indent=1)
        per_task = "; ".join(
            "%s eval %.4f mm, train mode %.4f mm" % (
                t, final_val.get("val_%s/center_rms_mm" % t, float("nan")),
                final_val.get("val_%s_trainmode/center_rms_mm" % t, float("nan"))) for t in tasks)
        print("overfit: reached=%s best %d-step mean center RMS (worst task) %.4f mm after %d steps; "
              "final val center RMS: %s" % (
                  trainer.reached, train_cfg.overfit_window,
                  summary["best_window_center_rms_mm"] or float("nan"), len(trainer.history), per_task))
        return 0 if trainer.reached else 1
    return 0


def _peak_mb(tf):
    try:
        if tf.config.list_physical_devices("GPU"):
            return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2.0 ** 20
    except (ValueError, RuntimeError):
        pass
    return None


if __name__ == "__main__":
    sys.exit(main())
