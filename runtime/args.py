"""Command-line arguments of main.py (TF-free: ``python main.py --help`` imports no TF).

Pattern from the rigid project's ``runtime/args.py`` (argparse + ``str2bool``). Defaults:
* optimizer: Adam, peak LR 3e-4, ``global_clipnorm`` 12 (Adam is nearly invariant to gradient
  scale, so the clip barely matters). On real level-4 data every SGDW run (rigid project
  values: momentum 0.99, Nesterov, wd 3e-5, lr 3e-3; clipped, or unclipped with
  --loss-scale-mm2 100; ReLU or LeakyReLU) collapsed to the identity fixed point (pre-head
  features identical across the group, zero gradient through GroupCentering). Adam 3e-4 with
  LeakyReLU 0.01 learned (milestone-4 report). SGDW stays selectable with --optimizer sgdw
  --lr 3e-3;
* schedule: 1000 warmup steps, then cosine to lr * 1e-3 at ``--steps``;
* losses: lambda_pair 1, p_pair 0.1, flip probability 0.5 per axis, 10 target iterations
  (design 6.1, 3.3); objective divided by --loss-scale-mm2 (default 1, unscaled) and multiplied
  by the task's --task-loss-weights entry (default 1 for every task; milestone 7);
* augmentation ranges: `data_loading.params.DEFAULT_LINEAR_RANGES` (design 3.2);
* curriculum (--curriculum) and global transform (--global-transform): both OFF (design 3.2
  v3.5). With --curriculum the --<task>-* range flags are the END values and --<task>-*-start
  the start values (start <= end is checked only then);
* group size per level: 32 (level 4), 8 (level 2), 2 (level 1) (design 5.1);
* model: `models.config.ModelConfig` defaults (dense-path norm 'layer'); dropout 0.15, or 0.0
  in overfit mode.
* deformable task (milestone 6; 'deform' in --task-ratio): lambda_pdice 1, p_pdice 0.1,
  lambda_smooth 0.5 with first-order smoothness, 7 scaling-and-squaring steps, decoder widths
  half the encoder widths, the deform decoder built iff 'deform' is in the ratio
  (--deform-decoder auto), no residual affine, deform group size = --group-size, no gradient
  checkpointing; overfit stop: window soft template Dice >= 0.9 and folding fraction < 0.001;
  validation pairs capped at 64 per group. Curriculum, global transform and control files are
  linear-only. Best checkpoint: the linear rule over the linear tasks; deform-only runs use
  val_deform/pair_dice_loss (1 - mean pairwise hard Dice); deform + linear runs also keep
  ckpt/best_deform on that key.
"""
import argparse
import collections
import datetime
import math

from data_loading import params as P
from models.config import ModelConfig

DEFAULT_GROUP_SIZE = {4: 32, 2: 8, 1: 2}
# Curriculum start values (--<task>-*-start). None = the end value (--<task>-*): the rigid
# scale/shear nuisance does not ramp by default.
CURRICULUM_START_DEFAULTS = {
    "rigid": {"rotation_deg": 15.0, "translation_mm": 5.0, "log_scale": None, "shear": None},
    "affine": {"rotation_deg": 15.0, "translation_mm": 5.0, "log_scale": math.log(1.03), "shear": 0.01},
}
RANGE_KEYS = ("rotation_deg", "translation_mm", "log_scale", "shear")
TRAIN_TASKS = ("rigid", "affine")
LINEAR_TASKS = TRAIN_TASKS
DEFORM_TASK = "deform"
# Combined best-checkpoint key (same value as runtime.run.MEAN_PAIR_METRIC; args stays TF-free).
MEAN_PAIR_METRIC = "val/mean_pair_rms_mm"
# Deformable best-checkpoint key (same value as runtime.run.DEFORM_BEST_METRIC).
DEFORM_BEST_METRIC = "val_deform/pair_dice_loss"
ALL_TASKS = ("rigid", "affine", "deform")
_MC = ModelConfig()


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("yes", "true", "t", "y", "1", "on"):
        return True
    if s in ("no", "false", "f", "n", "0", "off"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected, got {!r}".format(v))


def int_list(v):
    if isinstance(v, (list, tuple)):
        return tuple(int(x) for x in v)
    return tuple(int(x) for x in str(v).split(",") if x.strip())


def parse_task_ratio(text):
    """'rigid:1,affine:1,deform:2' -> OrderedDict(task -> weight > 0)."""
    out = collections.OrderedDict()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        name, _, w = part.partition(":")
        name = name.strip()
        if name not in ALL_TASKS:
            raise ValueError("unknown task {!r} in task ratio {!r} (tasks: {})".format(name, text, ALL_TASKS))
        weight = float(w) if w.strip() else 1.0
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("task weight must be finite and >= 0, got {!r}".format(part))
        if weight > 0:
            out[name] = out.get(name, 0.0) + weight
    if not out:
        raise ValueError("task ratio {!r} has no task with a positive weight".format(text))
    return out


def parse_task_loss_weights(text):
    """'rigid:1,affine:1,deform:4' -> {task: weight} (finite, > 0) for every task in ALL_TASKS;
    tasks not named get 1.0. None or '' -> all 1.0."""
    out = {t: 1.0 for t in ALL_TASKS}
    if text is None:
        return out
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, w = part.partition(":")
        name = name.strip()
        if name not in ALL_TASKS:
            raise ValueError("unknown task {!r} in task loss weights {!r} (tasks: {})".format(name, text, ALL_TASKS))
        if not sep or not w.strip():
            raise ValueError("task loss weight needs task:weight, got {!r}".format(part))
        weight = float(w)
        if not (math.isfinite(weight) and weight > 0):
            raise ValueError("task loss weight must be finite and > 0, got {!r}".format(part))
        out[name] = weight
    return out


def linear_tasks(tasks):
    """The linear tasks (rigid / affine) of `tasks` (a ratio dict, a list or parsed args), in order."""
    if hasattr(tasks, "task_ratio_parsed"):
        tasks = tasks.task_ratio_parsed
    return [t for t in tasks if t in LINEAR_TASKS]


def default_best_metric(tasks):
    """Default best-checkpoint key (same rule as runtime.run.resolve_best_metric): over the linear
    tasks, `MEAN_PAIR_METRIC` with more than one, else ``val_<task>/pair_rms_mm``; a deform-only
    ratio uses `DEFORM_BEST_METRIC`."""
    lin = linear_tasks(tasks)
    if not lin:
        return DEFORM_BEST_METRIC
    return MEAN_PAIR_METRIC if len(lin) > 1 else "val_%s/pair_rms_mm" % lin[0]


def build_parser():
    p = argparse.ArgumentParser(description="Groupwise registration training (rigid / affine / deform, one level)")
    a = p.add_argument
    g = p.add_argument_group("data and run")
    g.add_argument("--data", default=None, help="npy export root (holds export_manifest.csv and template/), "
                   "e.g. .../GroupwiseReg/data/npy_v2. Required unless --overfit-one-group --overfit-synthetic")
    g.add_argument("--results", default="./results", help="results root; the run goes to <results>/<run-name>")
    g.add_argument("--run-name", default=None, help="default: l<level>_<ratio>_<YYYYmmdd-HHMMSS>")
    g.add_argument("--resume", type=str2bool, default=True, help="restore the last checkpoint of the run if one exists")
    g.add_argument("--level", type=int, default=4, choices=list(P.LEVELS))
    g.add_argument("--group-size", type=int, default=None, help="N per group (default 32 / 8 / 2 for level 4 / 2 / 1)")
    g.add_argument("--steps", type=int, default=50000, help="total training steps")
    g.add_argument("--val-every", type=int, default=250, help="validate every K steps (0 = never)")
    g.add_argument("--val-groups", type=int, default=None, help="validation groups (default: ceil(n_val / Nv))")
    g.add_argument("--val-group-size", type=int, default=None, help="validation group size (default min(N, n_val))")
    g.add_argument("--val-seed", type=int, default=1234)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--ckpt-every", type=int, default=None, help="save 'last' every K steps (default --val-every)")
    g.add_argument("--val-png", type=str2bool, default=True,
                   help="write <run>/val_png/<task>_step<step>.png at every validation and at the final step")
    g.add_argument("--val-png-k", type=int, default=8, help="images per PNG (all if the group is smaller)")
    g.add_argument("--val-png-group", type=int, default=0, help="validation group index of the PNGs")
    g.add_argument("--val-png-gt-row", type=str2bool, default=True, help="third row: ground-truth alignment")
    g.add_argument("--val-png-mean-col", type=str2bool, default=True, help="last column: mean over the K images")
    g.add_argument("--best-metric", default=None,
                   help="validation key for the 'best' checkpoint (lower is better). 'mean_pair_rms_mm' "
                   "(logged as val/mean_pair_rms_mm) = mean of val_<task>/pair_rms_mm over the trained "
                   "linear tasks. Default: mean_pair_rms_mm when more than one linear task is "
                   "trained, else val_<task>/pair_rms_mm of the linear task; deform-only: "
                   "val_deform/pair_dice_loss. Deform + linear runs also keep ckpt/best_deform "
                   "on val_deform/pair_dice_loss")
    g = p.add_argument_group("optimizer and schedule")
    g.add_argument("--optimizer", default="adam", choices=["sgdw", "sgd", "adam", "adamw"],
                   help="default adam: every real level-4 SGDW run (lr 3e-3, momentum 0.99; clipped or "
                   "unclipped with --loss-scale-mm2 100; ReLU or LeakyReLU) collapsed to the identity "
                   "fixed point, while Adam 3e-4 + LeakyReLU 0.01 learned (milestone-4 report)")
    g.add_argument("--lr", type=float, default=3e-4,
                   help="peak LR (default 3e-4 for Adam; SGDW used 3e-3)")
    g.add_argument("--momentum", type=float, default=0.99)
    g.add_argument("--dampening", type=float, default=0.0)
    g.add_argument("--nesterov", type=str2bool, default=True)
    g.add_argument("--decoupled", type=str2bool, default=False, help="decoupled weight decay (SGDW)")
    g.add_argument("--weight-decay", type=float, default=3e-5)
    g.add_argument("--exclude-from-weight-decay", type=str2bool, default=True, help="no decay on gamma/beta/bias")
    g.add_argument("--clipnorm", type=float, default=12.0, help="global_clipnorm (<= 0 disables)")
    g.add_argument("--schedule", default="cosine", choices=["cosine", "poly", "cycle", "constant"])
    g.add_argument("--warmup-steps", type=int, default=1000)
    g.add_argument("--min-lr-factor", type=float, default=1e-3)
    g.add_argument("--power", type=float, default=0.9)
    g.add_argument("--cycle-steps", type=int, default=None, help="first cycle length (default (steps - warmup) / 4)")
    g.add_argument("--amp", type=str2bool, default=False, help="mixed_float16 policy + loss scaling")
    g.add_argument("--xla", type=str2bool, default=False, help="tf.config.optimizer.set_jit(True)")
    g.add_argument("--gpu-memory-limit-mb", type=int, default=0,
                   help="cap TF's GPU memory (logical device limit) instead of memory growth; 0 = off. "
                   "Needed on 8 GB WSL GPUs, where growing into the last GB pages GPU memory to host RAM")
    g = p.add_argument_group("tasks and losses")
    g.add_argument("--task-ratio", default="rigid:1", help="e.g. 'rigid:1,affine:1' or 'rigid:1,affine:1,deform:2'")
    g.add_argument("--lambda-pair", type=float, default=1.0)
    g.add_argument("--loss-scale-mm2", type=float, default=1.0,
                   help="divide the training objective (center + lambda_pair * pair, mm^2) by this "
                   "(> 0). 1 = unscaled. E.g. 100 makes the objective O(1-20) at init, so the "
                   "gradients are not permanently clipped by --clipnorm. Logged losses and RMS "
                   "stay in unscaled mm^2 / mm")
    g.add_argument("--task-loss-weights", default=None,
                   help="per-task objective multipliers, e.g. 'deform:4' or 'rigid:1,affine:1,deform:2' "
                   "(finite, > 0; tasks not named = 1; default all 1 = unchanged). Applied after "
                   "--loss-scale-mm2 (linear: objective = w * loss / loss_scale_mm2; deform: w * "
                   "loss). Balances the tasks' gradients on the shared encoder through the one Adam "
                   "(train_<task>/diag_enc_grad_norm_{raw,eff}). Logged losses stay unweighted")
    g.add_argument("--p-pair", type=float, default=0.1)
    g.add_argument("--flip-prob", type=float, default=0.5, help="per-axis group flip probability")
    g.add_argument("--target-iters", type=int, default=10)
    g = p.add_argument_group("deformable task (milestone 6; 'deform' in --task-ratio)")
    g.add_argument("--lambda-pdice", type=float, default=1.0, help="weight of the pairwise soft Dice loss")
    g.add_argument("--p-pdice", type=float, default=0.1,
                   help="fraction of the ordered pairs of a group in the pairwise Dice loss (>= 1 pair)")
    g.add_argument("--lambda-smooth", type=float, default=0.5, help="weight of the displacement smoothness loss")
    g.add_argument("--smooth-order", type=int, default=1, choices=[1, 2],
                   help="1: mean squared forward differences of u; 2: bending energy")
    g.add_argument("--integration-steps", type=int, default=_MC.integration_steps,
                   help="scaling-and-squaring steps of the SVF integration (ModelConfig)")
    g.add_argument("--decoder-widths", type=int_list, default=None,
                   help="deform decoder widths D5..D1 (5 values; default half the matching encoder widths)")
    g.add_argument("--deform-decoder", default="auto", choices=["auto", "true", "false"],
                   help="build the deform decoder: auto = iff 'deform' is in --task-ratio; true also "
                   "for linear runs that will warm-start a deform run")
    g.add_argument("--deform-residual-affine", type=str2bool, default=False,
                   help="small random residual affine in the deformable pipeline (images linear, "
                   "labels nearest, same matrix)")
    g.add_argument("--deform-group-size", type=int, default=None, help="deform group size (default --group-size)")
    g.add_argument("--gradient-checkpointing", default="none", choices=["none", "integrator", "decoder", "both"],
                   help="tf.recompute_grad around the SVF integrator and/or the deform decoder (memory)")
    g.add_argument("--overfit-dice-threshold", type=float, default=0.9,
                   help="overfit stop for deform: window mean soft template Dice >= this ...")
    g.add_argument("--overfit-fold-max", type=float, default=0.001,
                   help="... and window mean folding fraction (J <= 0) < this")
    g.add_argument("--val-deform-max-pairs", type=int, default=64,
                   help="deform validation: all ordered pairs of a group if at most this many, else a "
                   "fixed seeded subset of this size")
    g = p.add_argument_group("augmentation (half-widths of uniform ranges)")
    for task in TRAIN_TASKS:
        d = P.DEFAULT_LINEAR_RANGES[task]
        g.add_argument("--%s-rotation-deg" % task, type=float, default=d["rotation_deg"],
                       help="(with --curriculum: the end value, likewise the other range flags)")
        g.add_argument("--%s-translation-mm" % task, type=float, default=d["translation_mm"])
        g.add_argument("--%s-log-scale" % task, type=float, default=d["log_scale"])
        g.add_argument("--%s-shear" % task, type=float, default=d["shear"])
    g.add_argument("--intensity-aug", type=str2bool, default=True)
    g = p.add_argument_group("curriculum and global transform (design 3.2 v3.5; both default off)")
    g.add_argument("--curriculum", type=str2bool, default=False,
                   help="linear ramp of the per-image ranges from the --<task>-*-start values to "
                   "the --<task>-* values (the end values) over --curriculum-ramp-steps steps")
    g.add_argument("--curriculum-ramp-steps", type=int, default=2000)
    for task in TRAIN_TASKS:
        d = CURRICULUM_START_DEFAULTS[task]
        g.add_argument("--%s-rotation-deg-start" % task, type=float, default=d["rotation_deg"])
        g.add_argument("--%s-translation-mm-start" % task, type=float, default=d["translation_mm"])
        g.add_argument("--%s-log-scale-start" % task, type=float, default=d["log_scale"],
                       help=None if d["log_scale"] is not None else "default: --%s-log-scale" % task)
        g.add_argument("--%s-shear-start" % task, type=float, default=d["shear"],
                       help=None if d["shear"] is not None else "default: --%s-shear" % task)
    g.add_argument("--global-transform", type=str2bool, default=False,
                   help="one shared random rigid G per group (all images of a batch), composed "
                   "last: T_i = G Rt_i S_i; the network undoes G and Rt_i. Adds val_<task>_g sets")
    g.add_argument("--global-rot-max-deg", type=float, default=45.0,
                   help="G angle U(-max, max) about an axis uniform on the sphere")
    g.add_argument("--global-trans-max-mm", type=float, default=10.0, help="G translation U(-max, max) mm per axis")
    g.add_argument("--global-resample-every", type=int, default=5, help="draw a new G every K steps")
    g.add_argument("--global-start-step", type=int, default=0,
                   help="G warm-up: G = identity (ranges 0) before this step. On real data, G from "
                   "step 0 collapsed a level-4 rigid run to the identity fixed point (milestone 5); "
                   "enable G after the per-image task is learned")
    g.add_argument("--global-ramp-steps", type=int, default=0,
                   help="then ramp the G ranges linearly from 0 to --global-rot-max-deg / "
                   "--global-trans-max-mm over this many steps (0 = full ranges at once)")
    g.add_argument("--init-weights", default=None,
                   help="warm start: load ONLY the model weights from another run (a checkpoint "
                   "prefix, or a run's ckpt/best or ckpt/last directory = its latest checkpoint). "
                   "Same ModelConfig required (checked against the source run's config.json if "
                   "present), except the deform-decoder fields (deform_decoder, decoder_widths, "
                   "integration_steps): then only the encoder + linear heads are restored and the "
                   "deform decoder stays freshly initialised (e.g. a linear-only run into a "
                   "three-task run). Fresh optimizer, step 0; not allowed when this run dir "
                   "already has a checkpoint and --resume is on")
    g = p.add_argument_group("DALI")
    g.add_argument("--num-threads", type=int, default=4)
    g.add_argument("--prefetch", type=int, default=1)
    g.add_argument("--shuffle-mode", default="epoch", choices=["epoch", "buffer"])
    g = p.add_argument_group("model (models.config.ModelConfig)")
    g.add_argument("--group-config", default=_MC.group_config, choices=["A", "B"])
    g.add_argument("--block", default=_MC.block, choices=["single", "residual"])
    g.add_argument("--norm", default=_MC.norm, choices=["batch", "instance", "none"])
    g.add_argument("--dense-norm", default=_MC.dense_norm, choices=["layer", "batch", "none"],
                   help="norm of the dense blocks and the pre-head norm-act (default layer). 'batch' "
                   "is ablation-only: over one group it is group-relative in training mode and "
                   "breaks eval mode (milestone-4 report)")
    g.add_argument("--head-activation", default=_MC.head_activation, choices=["relu", "none"],
                   help="activation of the pre-head norm-act: 'relu' (default, follows "
                   "--negative-slope) or 'none' (norm only)")
    g.add_argument("--filters", type=int_list, default=_MC.filters)
    g.add_argument("--reductions", type=int_list, default=_MC.reductions)
    g.add_argument("--dense", type=int_list, default=_MC.dense)
    g.add_argument("--dropout", type=float, default=None, help="default 0.15, or 0.0 with --overfit-one-group")
    g.add_argument("--dense-group-concat-first", type=str2bool, default=_MC.dense_group_concat_first)
    g.add_argument("--negative-slope", type=float, default=_MC.negative_slope)
    g.add_argument("--translation-scale", type=float, default=_MC.translation_scale)
    g.add_argument("--centering-iters", type=int, default=_MC.centering_iters)
    g.add_argument("--bn-momentum", type=float, default=_MC.bn_momentum)
    g = p.add_argument_group("misc and modes")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--nonfinite-patience", type=int, default=20, help="raise after this many consecutive skipped steps")
    g.add_argument("--quiet", type=str2bool, default=False)
    g.add_argument("--log-backend", default="auto", choices=["auto", "dllogger", "jsonl"])
    g.add_argument("--dry-run", action="store_true", help="build everything, run 2 steps + 1 validation group, exit")
    g.add_argument("--overfit-one-group", action="store_true",
                   help="one fixed group of N training images per trained task (rigid: fixed pure-rigid "
                   "transforms; affine: fixed transforms from the --affine-* ranges; deform: a fixed "
                   "group of --deform-group-size images with labels, synthetic = a label phantom "
                   "deformed by random diffeomorphisms), TF-side warp (no DALI); stops when every "
                   "linear task's 10-step mean center RMS is below the threshold and the deform "
                   "window passes --overfit-dice-threshold / --overfit-fold-max")
    g.add_argument("--overfit-synthetic", action="store_true", help="overfit on a synthetic phantom (no data needed)")
    g.add_argument("--overfit-threshold-mm", type=float, default=1.0)
    g.add_argument("--overfit-max-steps", type=int, default=1500)
    del a
    return p


def parse_args(argv=None):
    """Parse and validate; resolves group size, dropout, run name and checkpoint cadence."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.task_ratio_parsed = parse_task_ratio(args.task_ratio)
        args.task_loss_weights_parsed = parse_task_loss_weights(args.task_loss_weights)
    except ValueError as e:
        parser.error(str(e))
    lin = linear_tasks(args)
    deform = DEFORM_TASK in args.task_ratio_parsed
    if args.data is None and not (args.overfit_one_group and args.overfit_synthetic):
        parser.error("--data is required unless --overfit-one-group --overfit-synthetic is given")
    if args.overfit_synthetic and not args.overfit_one_group:
        parser.error("--overfit-synthetic needs --overfit-one-group")
    if args.group_size is None:
        args.group_size = DEFAULT_GROUP_SIZE[args.level]
    if not (math.isfinite(args.loss_scale_mm2) and args.loss_scale_mm2 > 0):
        parser.error("--loss-scale-mm2 must be finite and > 0, got %r" % args.loss_scale_mm2)
    if args.val_png_k < 1 or args.val_png_group < 0:
        parser.error("--val-png-k must be >= 1 and --val-png-group >= 0")
    if args.group_size < 2:
        parser.error("--group-size must be >= 2 (pairwise loss), got %d" % args.group_size)
    if args.deform_group_size is None:
        args.deform_group_size = args.group_size
    if args.deform_group_size < 2:
        parser.error("--deform-group-size must be >= 2 (pairwise loss), got %d" % args.deform_group_size)
    if deform and args.deform_decoder == "false":
        parser.error("'deform' in --task-ratio needs the deform decoder (--deform-decoder auto or true)")
    if args.decoder_widths is not None and (len(args.decoder_widths) != 5
                                            or any(w < 1 for w in args.decoder_widths)):
        parser.error("--decoder-widths needs 5 positive values (D5..D1), got %r" % (args.decoder_widths,))
    if args.integration_steps < 0:
        parser.error("--integration-steps must be >= 0, got %d" % args.integration_steps)
    if args.val_deform_max_pairs < 1:
        parser.error("--val-deform-max-pairs must be >= 1, got %d" % args.val_deform_max_pairs)
    if not (0.0 < args.p_pdice <= 1.0):
        parser.error("--p-pdice must be in (0, 1], got %r" % args.p_pdice)
    if args.dropout is None:
        args.dropout = 0.0 if args.overfit_one_group else _MC.dropout
    if args.ckpt_every is None:
        args.ckpt_every = args.val_every
    if args.run_name is None:
        ratio = "-".join("%s%g" % (k, v) for k, v in args.task_ratio_parsed.items())
        args.run_name = "l%d_%s_%s" % (args.level, ratio, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if args.overfit_one_group:
        args.steps = args.overfit_max_steps
    for task in TRAIN_TASKS:
        for k in ("log_scale", "shear"):
            name = "%s_%s_start" % (task, k)
            if getattr(args, name) is None:
                setattr(args, name, float(getattr(args, "%s_%s" % (task, k))))
    if args.curriculum or args.global_transform:
        if args.overfit_one_group:
            parser.error("--curriculum / --global-transform need the DALI pipeline (not --overfit-one-group)")
        if not lin:
            parser.error("--curriculum / --global-transform act on the linear tasks only; "
                         "--task-ratio %s has none" % args.task_ratio)
    if args.curriculum:
        if args.curriculum_ramp_steps < 0:
            parser.error("--curriculum-ramp-steps must be >= 0, got %d" % args.curriculum_ramp_steps)
        for task in lin:
            for k in RANGE_KEYS:
                s = getattr(args, "%s_%s_start" % (task, k))
                e = getattr(args, "%s_%s" % (task, k))
                if not (math.isfinite(s) and 0 <= s <= e):
                    parser.error("--%s-%s-start must be in [0, --%s-%s = %g], got %g"
                                 % (task, k.replace("_", "-"), task, k.replace("_", "-"), e, s))
    if args.global_transform:
        for name in ("global_rot_max_deg", "global_trans_max_mm"):
            v = getattr(args, name)
            if not (math.isfinite(v) and v >= 0):
                parser.error("--%s must be finite and >= 0, got %r" % (name.replace("_", "-"), v))
        if args.global_resample_every < 1:
            parser.error("--global-resample-every must be >= 1, got %d" % args.global_resample_every)
        if args.global_start_step < 0 or args.global_ramp_steps < 0:
            parser.error("--global-start-step and --global-ramp-steps must be >= 0")
    if args.best_metric is None:
        args.best_metric = default_best_metric(args.task_ratio_parsed)
    elif args.best_metric == "mean_pair_rms_mm":
        args.best_metric = MEAN_PAIR_METRIC
    elif args.best_metric.startswith("val_rigid/") and "rigid" not in args.task_ratio_parsed and lin:
        args.best_metric = args.best_metric.replace("val_rigid/", "val_%s/" % lin[0])
    return args


def deform_decoder_from_args(args):
    """--deform-decoder: auto -> True iff 'deform' is in the task ratio; true / false as given."""
    if args.deform_decoder == "auto":
        return DEFORM_TASK in args.task_ratio_parsed
    return args.deform_decoder == "true"


def model_config_from_args(args):
    return ModelConfig(filters=args.filters, reductions=args.reductions, dense=args.dense,
                       dropout=args.dropout, group_config=args.group_config,
                       dense_group_concat_first=args.dense_group_concat_first, norm=args.norm,
                       dense_norm=args.dense_norm, head_activation=args.head_activation,
                       block=args.block,
                       negative_slope=args.negative_slope, bn_momentum=args.bn_momentum,
                       centering_iters=args.centering_iters,
                       translation_scale=args.translation_scale, seed=args.seed,
                       deform_decoder=deform_decoder_from_args(args),
                       decoder_widths=args.decoder_widths,
                       integration_steps=args.integration_steps)


def ranges_from_args(args):
    """{task: dict(rotation_deg, translation_mm, log_scale, shear)} from the --<task>-* flags."""
    out = {}
    for task in TRAIN_TASKS:
        out[task] = {k: float(getattr(args, "%s_%s" % (task, k)))
                     for k in ("rotation_deg", "translation_mm", "log_scale", "shear")}
    return out


def curriculum_start_from_args(args):
    """{task: dict(rotation_deg, translation_mm, log_scale, shear)} from the --<task>-*-start flags."""
    return {task: {k: float(getattr(args, "%s_%s_start" % (task, k))) for k in RANGE_KEYS}
            for task in TRAIN_TASKS}


def controls_enabled(args):
    """True if a control file is needed (--curriculum or --global-transform)."""
    return bool(args.curriculum or args.global_transform)


def task_controls_from_args(args, seed_offset=None):
    """{task: `runtime.curriculum.TaskControl`} for the ratio's LINEAR tasks (the deformable task
    has no control file), or {} when both features are off. G uses seed ``args.seed +
    seed_offset[task]`` (default rigid 0, affine 1). The writers are attached by the caller
    (`LinearDataModule.control_writers`)."""
    if not controls_enabled(args):
        return {}
    from runtime.curriculum import GlobalTransformSampler, LinearCurriculum, TaskControl
    seed_offset = seed_offset or {"rigid": 0, "affine": 1}
    end = ranges_from_args(args)
    start = curriculum_start_from_args(args)
    out = {}
    for task in linear_tasks(args):
        cur = (LinearCurriculum(start[task], end[task], args.curriculum_ramp_steps)
               if args.curriculum else None)
        smp = (GlobalTransformSampler(args.global_rot_max_deg, args.global_trans_max_mm,
                                      args.global_resample_every, seed=args.seed + seed_offset[task],
                                      start_step=args.global_start_step,
                                      ramp_steps=args.global_ramp_steps)
               if args.global_transform else None)
        out[task] = TaskControl(end[task], cur, smp)
    return out


def schedule_config_from_args(args):
    from runtime.schedulers import ScheduleConfig
    return ScheduleConfig(name=args.schedule, base_lr=args.lr, total_steps=max(1, args.steps),
                          warmup_steps=args.warmup_steps, min_lr_factor=args.min_lr_factor,
                          power=args.power, cycle_steps=args.cycle_steps)


def train_config_from_args(args):
    """`runtime.run.TrainConfig` (imports TF lazily). ``val_png_dir`` stays None: main.py sets it
    from the run directory when --val-png is on."""
    from runtime.run import TrainConfig
    return TrainConfig(level=args.level, group_size=args.group_size, steps=args.steps,
                       val_every=args.val_every, log_every=args.log_every,
                       ckpt_every=args.ckpt_every, task_ratio=dict(args.task_ratio_parsed),
                       lambda_pair=args.lambda_pair, loss_scale_mm2=args.loss_scale_mm2,
                       task_loss_weights=dict(args.task_loss_weights_parsed),
                       p_pair=args.p_pair,
                       flip_prob=0.0 if args.overfit_one_group else args.flip_prob,
                       target_iters=args.target_iters, amp=args.amp, seed=args.seed,
                       schedule=schedule_config_from_args(args),
                       nonfinite_patience=args.nonfinite_patience, best_metric=args.best_metric,
                       overfit=args.overfit_one_group,
                       overfit_threshold_mm=args.overfit_threshold_mm,
                       val_png_k=args.val_png_k, val_png_group=args.val_png_group,
                       val_png_gt_row=args.val_png_gt_row, val_png_mean_col=args.val_png_mean_col,
                       lambda_pdice=args.lambda_pdice, p_pdice=args.p_pdice,
                       lambda_smooth=args.lambda_smooth, smooth_order=args.smooth_order,
                       deform_residual_affine=args.deform_residual_affine,
                       deform_group_size=args.deform_group_size,
                       overfit_dice_threshold=args.overfit_dice_threshold,
                       overfit_fold_max=args.overfit_fold_max,
                       val_deform_max_pairs=args.val_deform_max_pairs)


def args_to_dict(args):
    out = {}
    for k, v in sorted(vars(args).items()):
        if isinstance(v, tuple):
            v = list(v)
        elif isinstance(v, collections.OrderedDict):
            v = dict(v)
        out[k] = v
    return out
