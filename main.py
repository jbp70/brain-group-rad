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
    python main.py --data <export_root> --task-ratio rigid:1,affine:1,deform:2 \
        --group-size 32 --deform-group-size 16 --init-weights <linear run>/ckpt/best  # m7
    python main.py --overfit-one-group --overfit-synthetic --task-ratio deform:1 --group-size 4

Deformable runs (milestone 6): the deform decoder is built when 'deform' is in --task-ratio;
the deform pipeline reads images + train18 labels (`DeformPipeline`), validation uses
`DeformValidationSet` (val images + labels); control files stay linear-only. With deform and a
linear task, ``ckpt/best`` keeps the linear criterion and ``ckpt/best_deform`` tracks
val_deform/pair_dice_loss.

Level cascade (milestone 8, `.agents/tasks/milestone-8/design.md` §2, §7, §9)::

    python main.py --data <export_root> --level 2 --cascade-from <level-4 run>/ckpt/best \
        --task-ratio rigid:1,affine:1,deform:2 --group-size 16 --deform-group-size 8 --bn-recal-batches 64

Fresh vs resume is decided ONCE before any model is built: resume iff --resume and the run's
``ckpt/last`` holds a checkpoint with step > 0 (``--resume false`` on an existing run dir is a
fresh run). Fresh: the source is resolved and checked (`C.load_cascade_source`, level 4, identical
model config; failure -> SystemExit), the frozen stage is restored from it
(`FrozenStage.from_checkpoint`), the level-2 model gets a verbatim copy, config.json gets the
``init_weights`` record (``via: cascade``) and the ``cascade`` block (source, step, frozen-weight
fingerprint, S_rel_diff_vs_level4 = ||S2 - S4||_F / ||S4||_F on real data, null in
overfit-synthetic runs, skip prob); after the PARAMETER record and the init_weights event,
--bn-recal-batches K > 0 runs `Trainer.recalibrate_bn` once and config.json is written again with
``bn_recalibration``. Resume: the source is NOT read; config.json must hold ``cascade``
(RuntimeError otherwise), the frozen stage (``frozen_stage`` in the run checkpoint) is restored
into `FrozenStage.from_config`, its fingerprint must match (RuntimeError), the BN freeze is
re-applied iff the SAVED ``bn_recalibration.freeze`` is true (the current --bn-recal-* flags are
ignored), and the three provenance blocks are carried over verbatim.

3-level cascade (milestone 9 phase 2, `.agents/tasks/milestone-9-phase2/design-m9-phase2.md`
§4.3); the source is a level-2 CASCADE run (its ``model`` + stored ``frozen_stage``)::

    python main.py --data <export_root> --level 1 --cascade-from <m8 level-2 run>/ckpt/best \
        --task-ratio rigid:1,affine:1,deform:2 --group-size 16 --deform-group-size 4 --amp true \
        --gradient-checkpointing full --lean-sampler true --split-group-conv true --bn-recal-batches 64

Fresh: `models.cascade.FrozenChain.from_checkpoint` (level-4 weights checked against the source
config.json ``cascade.fingerprint``; any failure -> SystemExit "main.py: --cascade-from ..."), the
level-1 model copies the level-2 weights, the ``cascade`` record gains ``S_rel_diff_vs_level2``,
``skip_stage`` 2 and ``stages`` (level 4 / key ``frozen_stage``, level 2 / key ``model``, with
fingerprints and origins); run-checkpoint keys ``frozen_stage_4`` / ``frozen_stage_2``. Resume:
config.json must hold ``cascade.stages`` (RuntimeError), every stage fingerprint is checked
(`_check_cascade_fingerprints`, RuntimeError naming the level).

Order: parse args (no TF) -> DALI check (unless overfit mode) -> configure_tf_precision() (TF32
off, memory growth) -> seeds -> mixed_float16 / XLA flags -> [cascade: fresh/resume decision,
source check] -> data -> model -> [cascade: frozen stage] -> optimizer -> run dir, logger,
config.json, checkpoint (resume) -> [cascade: BN recalibration] -> Trainer.fit.

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


def _build_n(args):
    """Group size of the build pass (`GroupRegLevelModel.build_model`). The variables do not
    depend on N; with deform the build pass (rigid + deform forward) uses the smaller of the two
    group sizes, so a large linear N does not size the deform forward. With --split-group-conv the
    build forward is the unsplit path (the variables are created there), so it runs at
    min(n_build, 2) to keep its group concat small (milestone 9 design 2.4, 3.5)."""
    n_build = args.group_size
    if A.DEFORM_TASK in args.task_ratio_parsed:
        n_build = min(n_build, args.deform_group_size)
    if getattr(args, "split_group_conv", False):
        n_build = min(n_build, 2)
    return n_build


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
    cascade_on = args.cascade_from is not None
    cascade_resume, cascade_src, saved_cfg = False, None, None
    if cascade_on:
        # NIT 4: fresh vs resume is decided ONCE, before any model is built. A fresh run reads
        # and checks the level-4 source; a resume never reads it (design section 9).
        run_root = os.path.join(args.results, args.run_name)
        cascade_resume = bool(args.resume) and C.peek_last_step(os.path.join(run_root, "ckpt", "last")) > 0
        if cascade_resume:
            saved_cfg = _saved_cascade_config(os.path.join(run_root, "config.json"))
            if args.level == 1 and not saved_cfg["cascade"].get("stages"):
                raise RuntimeError("resume of a level-1 cascade run without config.json cascade.stages: %s"
                                   % os.path.join(run_root, "config.json"))
        else:
            cascade_src = _load_cascade_source(C, args.cascade_from, A.model_config_from_args(args),
                                               level=args.level)
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
    model = GroupRegLevelModel(level, model_cfg, gradient_checkpointing=args.gradient_checkpointing,
                               lean_sampler=args.lean_sampler, split_group_conv=args.split_group_conv
                               ).build_model(_build_n(args))
    counts = model.count_params_by_part()
    init_info = None
    if args.init_weights:
        # Warm start (model variables only; fresh optimizer, step 0). Resume was excluded above.
        # A linear-only source into a deform model restores the encoder + linear heads and keeps
        # the deform decoder fresh (C.warm_start_model, mode 'partial').
        init_info = C.warm_start_model(model, model_cfg, args.init_weights)
        init_info["checkpoint"] = os.path.abspath(init_info["checkpoint"])
        if init_info["source_config"]:
            init_info["source_config"] = os.path.abspath(init_info["source_config"])
    frozen, cascade_rec = None, None
    if cascade_on:
        from models.cascade import FrozenChain, FrozenStage
        chain = level == 1                       # milestone 9 phase 2: frozen 4 -> 2 chain
        if cascade_resume:
            # Random init; the run checkpoint (key frozen_stage, or frozen_stage_4 / frozen_stage_2
            # for a chain) restores it below.
            frozen = FrozenChain.from_config(model_cfg) if chain else FrozenStage.from_config(model_cfg)
            cascade_rec = saved_cfg["cascade"]
        elif chain:
            prefix, src_cfg, src_info = cascade_src
            upstream = src_info["upstream_cascade"]
            try:
                frozen = FrozenChain.from_checkpoint(prefix, src_cfg, upstream["fingerprint"])
            except ValueError as e:
                raise SystemExit("main.py: --cascade-from %s: %s" % (args.cascade_from, e))
            model.copy_weights_from(frozen.source_model)
            init_info = C.cascade_init_record(model, src_info)
            s_rel4 = s_rel2 = None
            if not args.overfit_one_group:
                # Real data: level-4 / level-2 S (template_mask_{4,2}.npy) vs this run's level-1 S.
                tdir = os.path.join(args.data, dm.TEMPLATE_DIR)
                s1 = np.asarray(S, np.float64)
                Sk = {k: np.asarray(postprocess.brain_S_matrix(k, tdir), np.float64) for k in (4, 2)}
                s_rel4 = float(np.linalg.norm(s1 - Sk[4]) / np.linalg.norm(Sk[4]))
                s_rel2 = float(np.linalg.norm(s1 - Sk[2]) / np.linalg.norm(Sk[2]))
            fps = frozen.fingerprints
            cascade_rec = {"source_checkpoint": src_info["checkpoint"], "source_config": src_info["source_config"],
                           "source_level": int(src_info["source_level"]),
                           "source_step": int(src_info["source_step"]), "fingerprint": frozen.fingerprint,
                           "S_rel_diff_vs_level4": s_rel4, "S_rel_diff_vs_level2": s_rel2,
                           "skip_prob": float(args.cascade_skip_prob), "skip_stage": 2,
                           "stages": [
                               {"level": 4, "checkpoint_key": "frozen_stage", "fingerprint": fps[4],
                                "origin_checkpoint": upstream.get("source_checkpoint"),
                                "origin_step": upstream.get("source_step")},
                               {"level": 2, "checkpoint_key": "model", "fingerprint": fps[2],
                                "origin_checkpoint": src_info["checkpoint"],
                                "origin_step": int(src_info["source_step"])}]}
        else:
            prefix, src_cfg, src_info = cascade_src
            frozen = FrozenStage.from_checkpoint(prefix, src_cfg)
            model.copy_weights_from(frozen.model)
            init_info = C.cascade_init_record(model, src_info)
            s_rel = None
            if not args.overfit_one_group:
                # Real data (DALI path): level-4 S from template_mask_4.npy vs this run's level-2 S.
                S4 = np.asarray(postprocess.brain_S_matrix(4, os.path.join(args.data, dm.TEMPLATE_DIR)),
                                np.float64)
                s_rel = float(np.linalg.norm(np.asarray(S, np.float64) - S4) / np.linalg.norm(S4))
            cascade_rec = {"source_checkpoint": src_info["checkpoint"], "source_config": src_info["source_config"],
                           "source_level": int(src_info["source_level"]),
                           "source_step": int(src_info["source_step"]), "fingerprint": frozen.fingerprint,
                           "S_rel_diff_vs_level4": s_rel, "skip_prob": float(args.cascade_skip_prob)}
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
                      controls=controls, cascade=frozen)
    # Deform + linear: a second best file on the deform metric (ckpt/best keeps the linear rule).
    extra_best = ({name: os.path.join(paths["root"], "ckpt", "best_%s" % name) for name in trainer.extra_best}
                  or None)
    ckpt_extra = trainer.checkpoint_variables()
    if frozen is not None:
        # Cascade runs only: frozen_stage (level 2) or frozen_stage_4 / frozen_stage_2 (level 1).
        ckpt_extra = dict(ckpt_extra, **frozen.checkpoint_extra())
    trainer.checkpoint = C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], model, optimizer,
                                         trainer.step, extra=ckpt_extra, extra_best=extra_best)
    resumed = False
    if init_info is not None:
        if not args.quiet:
            print(_init_weights_lines(init_info))
    elif args.resume and trainer.checkpoint.latest_last:
        C.check_config_compatible(paths["config"], model_cfg)
        resumed = trainer.checkpoint.restore_last()
    config_extra = {"param_counts": dict(counts), "data": data_desc, "init_weights": init_info}
    if cascade_on:
        if cascade_resume:
            # Design section 9 steps 3-6: restored, fingerprint checked, BN freeze re-applied
            # from the SAVED record, provenance blocks carried over verbatim.
            if not resumed:
                raise RuntimeError("resume of cascade run %s: no checkpoint restored from %s"
                                   % (paths["root"], paths["ckpt_last"]))
            if cascade_rec.get("stages"):
                _check_cascade_fingerprints(cascade_rec, frozen, paths["config"])
            else:
                _check_cascade_fingerprint(cascade_rec.get("fingerprint"), frozen.fingerprint, paths["config"])
            bn_saved = saved_cfg.get("bn_recalibration")
            if (bn_saved or {}).get("freeze") is True:
                model.set_bn_stats_frozen(True)
                trainer.rebuild_steps()
            config_extra["init_weights"] = saved_cfg.get("init_weights")
            if "bn_recalibration" in saved_cfg:
                config_extra["bn_recalibration"] = bn_saved
        config_extra["cascade"] = cascade_rec
        if not args.quiet:
            print(_cascade_lines(cascade_rec, cascade_resume, saved_cfg))
    C.write_config(paths["config"], model_cfg, A.args_to_dict(args), extra=config_extra)
    logger.log_params(dict(A.args_to_dict(args), run_dir=paths["root"], resumed=resumed,
                           resumed_step=int(trainer.step.numpy()), data=data_desc,
                           trainable_params=counts["trainable"],
                           non_trainable_params=counts["non_trainable"],
                           tf32=bool(tf.config.experimental.tensor_float_32_execution_enabled())))
    if init_info is not None:
        # After the PARAMETER record (the first line of metrics.jsonl stays the run parameters).
        logger.log_event(0, dict(init_info, event="init_weights"))
    if cascade_on and not cascade_resume and args.bn_recal_batches > 0:
        # Once, at the start of a fresh cascade run (never on a resume: the recalibrated
        # statistics come back with the checkpoint); then config.json again with the record.
        recal = trainer.recalibrate_bn(args.bn_recal_batches, args.bn_recal_freeze)
        C.write_config(paths["config"], model_cfg, A.args_to_dict(args),
                       extra=dict(config_extra, bn_recalibration=recal))
        if not args.quiet:
            print("bn recalibration: %d batches %s, %d BN layers updated, freeze %s, %.1f s; gap %s" % (
                recal["num_batches"], recal["batches"], recal["n_layers_updated"], recal["freeze"],
                recal["elapsed_s"], ", ".join("%s %.4g" % kv for kv in sorted(recal["gap"].items())) or "none"))
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


def _init_weights_lines(info):
    """Human-readable warm-start summary (printed unless --quiet; the same dict is logged as the
    'init_weights' event and stored in config.json)."""
    restored = ", ".join("%s (%d vars)" % kv for kv in info["restored_parts"].items())
    fresh = ", ".join("%s (%d vars)" % kv for kv in info["fresh_parts"].items()) or "none"
    lines = ["init weights (%s) from %s (source config %s)" % (info["mode"], info["checkpoint"],
                                                               info["source_config"]),
             "  restored: %s" % restored,
             "  freshly initialised: %s" % fresh,
             "  config diff (source, this run): %s" % (info["config_diff"] or "none")]
    if info["ignored_source_parts"]:
        lines.append("  ignored source parts: %s" % ", ".join(info["ignored_source_parts"]))
    return "\n".join(lines)


def _load_cascade_source(C, path, model_cfg, level=2):
    """Fresh cascade run at `level`: (prefix, source ModelConfig, info) of the source run (level 4
    for level 2, a level-2 cascade run for level 1), checked against this run's model config;
    any problem -> SystemExit (exit 1) with the message."""
    try:
        prefix, src_cfg, info = C.load_cascade_source(path, level)
        C.check_cascade_config(model_cfg, src_cfg)
    except ValueError as e:
        msg = str(e)
        raise SystemExit("main.py: " + msg if msg.startswith("--cascade-from")
                         else "main.py: --cascade-from %s: %s" % (path, msg))
    return prefix, src_cfg, info


def _saved_cascade_config(path):
    """Resume of a cascade run: the saved config.json, which must hold the ``cascade`` record."""
    saved = None
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
    if not (saved or {}).get("cascade"):
        raise RuntimeError("resume of a cascade run without config.json cascade: %s" % path)
    return saved


FINGERPRINT_RTOL = 1e-12


def _check_cascade_fingerprint(saved, current, path):
    """RuntimeError unless the restored frozen stage's `current` weight fingerprint matches the
    `saved` config.json one (n_vars exact, abs_sum / sq_sum relative 1e-12;
    `runtime.checkpoint.fingerprints_match`)."""
    from runtime import checkpoint as C
    saved = saved or {}
    if not C.fingerprints_match(saved, current, rtol=FINGERPRINT_RTOL):
        raise RuntimeError("resume of a cascade run: the frozen-stage weight fingerprint after the restore %s "
                           "differs from config.json cascade.fingerprint %s (%s)" % (current, saved, path))


def _check_cascade_fingerprints(rec, frozen, path):
    """Resume of a level-1 cascade run: RuntimeError naming the level unless every stage of
    ``rec['stages']`` matches ``frozen.fingerprints[level]`` and the top-level
    ``rec['fingerprint']`` matches ``frozen.fingerprint`` (level 2)."""
    from runtime import checkpoint as C
    current = frozen.fingerprints
    for stage in rec.get("stages") or []:
        lv = stage.get("level")
        if lv not in current or not C.fingerprints_match(stage.get("fingerprint"), current[lv],
                                                         rtol=FINGERPRINT_RTOL):
            raise RuntimeError("resume of a cascade run: the frozen level-%s weight fingerprint after the restore "
                               "%s differs from config.json cascade.stages %s (%s)"
                               % (lv, current.get(lv), stage.get("fingerprint"), path))
    if not C.fingerprints_match(rec.get("fingerprint"), frozen.fingerprint, rtol=FINGERPRINT_RTOL):
        raise RuntimeError("resume of a cascade run: the frozen level-%d weight fingerprint after the restore %s "
                           "differs from config.json cascade.fingerprint %s (%s)"
                           % (frozen.level, frozen.fingerprint, rec.get("fingerprint"), path))


def _cascade_lines(rec, resumed, saved_cfg):
    """Human-readable cascade summary (printed unless --quiet)."""
    fp = rec["fingerprint"]
    lines = ["cascade: frozen level-%d stage from %s (step %d; source config %s)" % (
        rec["source_level"], rec["source_checkpoint"], rec["source_step"], rec["source_config"]),
        "  fingerprint: %d vars, abs sum %.10g, sq sum %.10g; S rel diff vs level 4: %s; skip prob %g" % (
            fp["n_vars"], fp["abs_sum"], fp["sq_sum"], rec["S_rel_diff_vs_level4"], rec["skip_prob"])]
    for st in rec.get("stages") or []:            # level-1 chain: one line per frozen stage
        sfp = st.get("fingerprint") or {}
        lines.append("  stage level %s (checkpoint key %s): %s vars, abs sum %.10g; origin %s (step %s)" % (
            st.get("level"), st.get("checkpoint_key"), sfp.get("n_vars"), float(sfp.get("abs_sum") or 0.0),
            st.get("origin_checkpoint"), st.get("origin_step")))
    if "S_rel_diff_vs_level2" in rec:
        lines.append("  S rel diff vs level 2: %s; skip stage %s" % (rec["S_rel_diff_vs_level2"],
                                                                    rec.get("skip_stage")))
    if resumed:
        bn = (saved_cfg or {}).get("bn_recalibration") or {}
        lines.append("  resumed: frozen stage restored from the run checkpoint (source not read), fingerprint "
                     "OK; BN statistics frozen from the saved record: %s" % (bn.get("freeze") is True))
    return "\n".join(lines)


def _peak_mb(tf):
    try:
        if tf.config.list_physical_devices("GPU"):
            return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2.0 ** 20
    except (ValueError, RuntimeError):
        pass
    return None


if __name__ == "__main__":
    sys.exit(main())
