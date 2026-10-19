"""Custom training loop for the linear tasks (design 5.2): task schedule, one tf.function train
step per task, NaN guard, validation, logging and checkpoint hooks.

Train step (one per task, traced once; input_signature = the DALI output shapes for the fixed N
of the level, `data_loading.dali_loader.linear_output_signature`)::

    step(images [N,X,Y,Z,1], T_dali [N,3,4] (, Rt_dali [N,3,4] for rigid))   all float32
      batch = LinearPostprocess(...)            # flips, mm matrices, targets, pairs (no tape)
      non-finite input images -> zeroed, step marked non-finite (the barycenter's matrix
        inverse raises on NaN instead of propagating it)
      backup BN moving statistics
      with GradientTape: out = model(images, task, training=True); terms = linear_loss(out["A"])
        objective = terms["loss"] / loss_scale_mm2     (terms stay in mm^2 for the logs)
      diag = collapse_diagnostics(out["pre_head"], out["params"])   (no gradient, every step)
      grads -> drop None (e.g. the scale/shear head in rigid steps; static, at trace time)
      finite = isfinite(loss) & isfinite(global_norm(grads))
      tf.cond(finite, apply_gradients, skip); if not finite: restore BN statistics
      step += 1;  returns loss terms, diag_*, grad_norm, finite, n_vars_updated

Collapse diagnostics (`collapse_diagnostics`, logged as ``train_<task>/diag_*`` means over the
log window): a degenerate fixed point where every image of a group gets the same pre-head
feature vector makes the heads output one transform for all images, GroupCentering cancels it
to the identity and the gradient is exactly zero. ``fit`` counts consecutive finite steps with
relative feature spread < ``collapse_rel_spread`` or grad norm < ``collapse_grad_norm``; it logs
the count as ``train/collapse_suspect`` and warns (stderr + a ``collapse_suspect`` event) when it
reaches ``collapse_patience``.

Images stay float32 (DALI cannot warp float16); the model casts them to its compute dtype.
Under --amp the LossScaleOptimizer is always called (it skips non-finite updates itself and
lowers its scale); the BN rollback and the logs use our finite flag.

Validation PNGs (`runtime.val_png`, when ``val_png_dir`` is set): every `validate()` writes
``<val_png_dir>/<task>_step<step:07d>.png`` from the eval-mode matrices of group
``val_png_group``; `fit` also validates at its final step if that was not a validation step.

Curriculum / global transform (``controls``, `runtime.curriculum`, default none): at the start
of each step, before fetching the batch, every task's control file gets the values of that
step (pure functions of the step, so a resume restores them). The batch's trailing G_dali and
control_version are split off before the step function (same traced signature) and logged as
``train_<task>/{g_mixed_batch, control_version_lag, control_rot_deg, control_trans_mm,
control_g_angle_deg}`` (window means). Flips still act as F T F on the emitted T and target.

`rebuild_steps()` re-creates every step function. Call it after changing anything read at trace
time, in particular `model.set_bn_stats_frozen(...)` (an already traced step keeps updating
the statistics otherwise).
"""
import collections
import dataclasses
import math
import os
import random
import time
from typing import Dict, Optional

import sys

import numpy as np
import tensorflow as tf

from data_loading import dali_loader
from data_loading import params as P
from data_loading.postprocess import LinearPostprocess
from models import transforms as tfm
from runtime import losses
from runtime import metrics as M
from runtime import optimizers as O
from runtime.args import parse_task_ratio  # noqa: F401  (re-exported)
from runtime.schedulers import ScheduleConfig, lr_at

POSTPROCESS_SEED_OFFSET = {"rigid": 101, "affine": 202}
DIAG_KEYS = ("diag_feat_spread", "diag_feat_rel_spread", "diag_dead_frac", "diag_rot10_norm",
             "diag_head_spread")


def collapse_diagnostics(pre_head, params):
    """Group-collapse diagnostics of one forward pass (TF, float32 scalars).

    Args:
      pre_head: [N, D] head features (after head_norm_act, before dropout).
      params: [N, 13 | 19] raw head outputs (10D rotation, 3 translations[, 6 scale/shear]).
    Returns dict:
      diag_feat_spread: mean over the D features of the std across the N images;
      diag_feat_rel_spread: that divided by the mean |pre_head| (0 if all features are 0);
      diag_dead_frac: fraction of the D channels that are <= 0 for every image;
      diag_rot10_norm: mean over images of the L2 norm of the 10 rotation outputs;
      diag_head_spread: mean over the 13 rigid-head outputs of the std across the N images.
    """
    f = tf.cast(pre_head, tf.float32)
    spread = tf.reduce_mean(tf.math.reduce_std(f, axis=0))
    p = tf.cast(params[:, :tfm.RIGID_PARAMS], tf.float32)
    return {"diag_feat_spread": spread,
            "diag_feat_rel_spread": tf.math.divide_no_nan(spread, tf.reduce_mean(tf.abs(f))),
            "diag_dead_frac": tf.reduce_mean(tf.cast(tf.reduce_all(f <= 0.0, axis=0), tf.float32)),
            "diag_rot10_norm": tf.reduce_mean(tf.norm(p[:, :tfm.ROTATION_DIMS], axis=-1)),
            "diag_head_spread": tf.reduce_mean(tf.math.reduce_std(p, axis=0))}


class TaskSchedule:
    """Stateless seeded task draw: task(step) depends only on (seed, step) and the ratio."""

    def __init__(self, ratio, seed=0):
        if isinstance(ratio, str):
            ratio = parse_task_ratio(ratio)
        self.ratio = collections.OrderedDict((k, float(v)) for k, v in ratio.items() if float(v) > 0)
        if not self.ratio:
            raise ValueError("empty task ratio")
        if "deform" in self.ratio:
            raise NotImplementedError("the deformable task arrives in milestone 6")
        self.seed = int(seed)
        total = sum(self.ratio.values())
        acc = 0.0
        self._cum = []
        for k, v in self.ratio.items():
            acc += v / total
            self._cum.append((acc, k))

    @property
    def tasks(self):
        return list(self.ratio.keys())

    def draw(self, step):
        if len(self._cum) == 1:
            return self._cum[0][1]
        u = random.Random(self.seed * 1000003 + int(step)).random()
        for c, k in self._cum:
            if u < c:
                return k
        return self._cum[-1][1]


@dataclasses.dataclass
class TrainConfig:
    level: int = 4
    group_size: int = 32
    steps: int = 50000
    val_every: int = 250
    log_every: int = 50
    ckpt_every: Optional[int] = None
    task_ratio: Dict[str, float] = dataclasses.field(default_factory=lambda: {"rigid": 1.0})
    lambda_pair: float = 1.0
    # The objective is (center + lambda_pair * pair) / loss_scale_mm2; logs stay in mm^2 / mm.
    loss_scale_mm2: float = 1.0
    p_pair: float = 0.1
    flip_prob: float = 0.5
    target_iters: int = 10
    amp: bool = False
    seed: int = 0
    schedule: ScheduleConfig = dataclasses.field(default_factory=ScheduleConfig)
    nonfinite_patience: int = 20
    best_metric: str = "val_rigid/pair_rms_mm"
    overfit: bool = False
    overfit_threshold_mm: float = 1.0
    overfit_window: int = 10
    # Collapse warning: this many consecutive finite steps with relative pre-head feature spread
    # below collapse_rel_spread or grad norm below collapse_grad_norm.
    collapse_rel_spread: float = 1e-3
    collapse_grad_norm: float = 1e-3
    collapse_patience: int = 50
    # Validation PNGs (runtime.val_png): written to val_png_dir (None = off) at every validation
    # and at the final step, from validation group val_png_group, K = val_png_k fixed images.
    val_png_dir: Optional[str] = None
    val_png_k: int = 8
    val_png_group: int = 0
    val_png_gt_row: bool = True
    val_png_mean_col: bool = True
    val_png_seed: int = 0


def _peak_gpu_mb():
    try:
        if tf.config.list_physical_devices("GPU"):
            return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2.0 ** 20
    except (ValueError, RuntimeError):
        pass
    return None


class Trainer:
    """Holds the model, ONE optimizer, the task sources and the step functions.

    Args:
      model: a built `GroupRegLevelModel`.
      optimizer: from `runtime.optimizers.make_optimizer` (built on all trainable variables here).
      sources: {task: iterator yielding the DALI-format tuple}.
      S: [4, 4] brain second moment (float64 numpy) of the level.
      cfg: `TrainConfig`.
      logger: optional `runtime.logging.RunLogger`; checkpoint: optional `RunCheckpoint` (can be
        attached later, it needs `trainer.step` and `checkpoint_variables()`).
      val_sets: optional {name: `runtime.fixed_data.ValidationSet`}; the name is the metric
        prefix ``val_<name>`` (e.g. 'rigid', or 'rigid_g' for the set with a global transform),
        the set's ``task`` selects the model head.
      controls: optional {task: `runtime.curriculum.TaskControl`} with writers attached
        (curriculum / global transform). Each step first writes every task's control values of
        that step; batches then carry G_dali and control_version, which are logged and dropped
        before the step function. None (default): no control file is touched.
    """

    def __init__(self, model, optimizer, sources, S, cfg, logger=None, checkpoint=None, val_sets=None,
                 controls=None):
        self.model = model
        self.optimizer = optimizer
        self.sources = dict(sources)
        self.cfg = cfg
        self.logger = logger
        self.checkpoint = checkpoint
        self.val_sets = dict(val_sets or {})
        self.controls = dict(controls or {})
        self.level = P.check_level(cfg.level)
        self.S_np = np.asarray(S, np.float64)
        self.S = tf.constant(self.S_np, tf.float32)
        self.schedule = TaskSchedule(cfg.task_ratio, cfg.seed)
        missing = [t for t in self.schedule.tasks if t not in self.sources]
        if missing:
            raise ValueError("no data source for task(s) %s" % missing)
        self.step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
        self.postprocess = {t: LinearPostprocess(t, self.level, flip_prob=cfg.flip_prob, p_pair=cfg.p_pair,
                                                 seed=cfg.seed + POSTPROCESS_SEED_OFFSET[t],
                                                 target_iters=cfg.target_iters)
                            for t in self.sources}
        self.trace_counts = {t: 0 for t in self.sources}
        self.val_trace_counts = {}
        self.nonfinite_total = 0
        self.collapse_streak = 0        # consecutive finite collapse-suspect steps
        self.collapse_warnings = 0
        self.val_png_paths = []
        self.history = []
        self.val_history = []
        self.reached = False
        O.build_optimizer(self.optimizer, self.model.trainable_variables)
        self.rebuild_steps()

    # ------------------------------------------------------------------ step functions
    def checkpoint_variables(self):
        """Extra trackables for `RunCheckpoint`: the post-processing RNGs."""
        return {"postprocess_%s" % t: pp.generator for t, pp in self.postprocess.items()}

    def rebuild_steps(self):
        """Re-create every train / validation tf.function (re-trace on next use)."""
        self._steps = {t: self._make_train_step(t) for t in self.sources}
        self._val_fns = {}

    def _make_train_step(self, task):
        model, opt, cfg = self.model, self.optimizer, self.cfg
        shapes, _ = dali_loader.linear_output_signature(task, self.level, cfg.group_size)
        signature = [tf.TensorSpec(s, tf.float32) for s in shapes]
        pp = self.postprocess[task]
        amp = bool(cfg.amp)
        lam = float(cfg.lambda_pair)
        loss_scale = float(cfg.loss_scale_mm2)
        if not (math.isfinite(loss_scale) and loss_scale > 0):
            raise ValueError("loss_scale_mm2 must be finite and > 0, got %r" % loss_scale)
        S = self.S

        def body(images, t_dali, rt_dali=None):
            self.trace_counts[task] += 1          # Python side effect: runs only while tracing
            batch = pp(images, t_dali, rt_dali)   # outside the tape
            # Non-finite inputs would make the barycenter's matrix inverse raise instead of
            # returning NaN, so they are zeroed and the step is skipped like any non-finite step.
            x = batch["images"]
            in_finite = tf.reduce_all(tf.math.is_finite(x))
            batch["images"] = tf.where(in_finite, x, tf.zeros_like(x))
            stats = list(model.non_trainable_variables)
            backup = [tf.identity(v) for v in stats]
            tvars = model.trainable_variables
            with tf.GradientTape() as tape:
                out = model(batch["images"], task=task, training=True)
                terms = losses.linear_loss(out["A"], batch, S, lam)
                objective = terms["loss"]
                if loss_scale != 1.0:
                    objective = objective / tf.constant(loss_scale, objective.dtype)
                scaled = opt.get_scaled_loss(objective) if amp else objective
            grads = tape.gradient(scaled, tvars)
            diag = collapse_diagnostics(out["pre_head"], out["params"])
            pairs = [(g, v) for g, v in zip(grads, tvars) if g is not None]   # static filter
            gs = [g for g, _ in pairs]
            vs = [v for _, v in pairs]
            if amp:
                gs = opt.get_unscaled_gradients(gs)
            gnorm = tf.linalg.global_norm(gs)
            finite = tf.logical_and(tf.math.is_finite(terms["loss"]), tf.math.is_finite(gnorm))
            finite = tf.logical_and(finite, in_finite)
            def apply():
                opt.apply_gradients(zip(gs, vs))
                return tf.constant(True)

            # Under AMP the LSO must see non-finite gradients (it skips them itself and lowers
            # its scale); zeroed non-finite inputs give finite gradients, so they are gated here.
            tf.cond(in_finite if amp else finite, apply, lambda: tf.constant(False))

            def restore():
                for v, b in zip(stats, backup):
                    v.assign(b)
                return tf.constant(True)

            if stats:
                tf.cond(finite, lambda: tf.constant(False), restore)
            self.step.assign_add(1)
            res = dict(terms)
            res.update(diag)
            res.update(grad_norm=gnorm, finite=finite, n_vars_updated=tf.constant(len(pairs)))
            return res

        if task == "rigid":
            return tf.function(lambda images, t_dali, rt_dali: body(images, t_dali, rt_dali),
                               input_signature=signature)
        return tf.function(lambda images, t_dali: body(images, t_dali), input_signature=signature)

    @staticmethod
    def split_batch(task, batch_tuple):
        """(step inputs, control extras): the first 3 (rigid) / 2 (affine) DALI outputs, and the
        trailing (G_dali, control_version) of a control-file pipeline (empty tuple otherwise)."""
        k = dali_loader.base_output_count(task)
        batch_tuple = tuple(batch_tuple)
        return batch_tuple[:k], batch_tuple[k:]

    def control_metrics(self, task, extras, values):
        """Per-step control logs: g_mixed_batch (1 if the batch's G_dali differ across samples),
        the current control values and the version lag (writer version - batch version)."""
        out = {"control_rot_deg": float(values["rotation_deg"]),
               "control_trans_mm": float(values["translation_mm"]),
               "control_g_angle_deg": float(values["g_angle_deg"])}
        if len(extras) >= 2:
            g = np.asarray(extras[0], np.float64)
            ver = np.asarray(extras[1], np.float64).reshape(-1)
            out["g_mixed_batch"] = float(np.abs(g - g[:1]).max() > 1e-6 or np.ptp(ver) > 0)
            writer = self.controls[task].writer
            out["control_version_lag"] = float(writer.version - ver.min()) if writer is not None else 0.0
        return out

    def train_step(self, task, batch_tuple):
        """Run one step of `task` on a DALI-format tuple (control extras are ignored). Returns
        Python scalars."""
        batch_tuple, _ = self.split_batch(task, batch_tuple)
        out = self._steps[task](*batch_tuple)
        res = {k: float(v.numpy()) for k, v in out.items() if k not in ("finite", "n_vars_updated")}
        res["finite"] = bool(out["finite"].numpy())
        res["n_vars_updated"] = int(out["n_vars_updated"].numpy())
        return res

    # ------------------------------------------------------------------ validation
    def _val_fn(self, task, nv, train_mode=False):
        """Validation forward -> A. ``train_mode=False``: eval mode (moving BN statistics), the
        source of truth. ``train_mode=True``: diagnostic forward with training-mode BN (batch
        statistics of the group) and dropout off; the BN moving statistics are restored
        afterwards, so it changes no state."""
        key = (task, int(nv), bool(train_mode))
        if key not in self._val_fns:
            shape = (int(nv),) + tuple(P.LEVEL_SHAPE[self.level]) + (1,)
            model = self.model
            self.val_trace_counts.setdefault(key, 0)

            def fwd(x):
                self.val_trace_counts[key] += 1
                return model(x, task=task, training=False)["A"]

            def fwd_train_mode(x):
                self.val_trace_counts[key] += 1
                stats = list(model.non_trainable_variables)
                backup = [tf.identity(v) for v in stats]
                # Dropout rates are read at trace time: zero them only while tracing.
                drops = [m for m in model.submodules if isinstance(m, tf.keras.layers.Dropout)]
                rates = [d.rate for d in drops]
                try:
                    for d in drops:
                        d.rate = 0.0
                    a = model(x, task=task, training=True)["A"]
                finally:
                    for d, r in zip(drops, rates):
                        d.rate = r
                with tf.control_dependencies([a]):
                    restores = [v.assign(b) for v, b in zip(stats, backup)]
                with tf.control_dependencies(restores):
                    return tf.identity(a)

            self._val_fns[key] = tf.function(fwd_train_mode if train_mode else fwd,
                                             input_signature=[tf.TensorSpec(shape, tf.float32)])
        return self._val_fns[key]

    def _val_errors(self, task, vs, groups, train_mode):
        """(pooled errors, [A per group] float64)."""
        fn = self._val_fn(task, vs.group_size, train_mode)
        errs, mats = [], []
        for g, grp in enumerate(groups):
            A = fn(tf.constant(grp["images"])).numpy().astype(np.float64)
            mats.append(A)
            errs.append(M.group_errors(A, grp["A_star"], grp["T_target"], self.S_np, seed=g))
        return M.concat_errors(errs), mats

    def _write_val_png(self, task, grp, A, metrics, step):
        """One validation PNG (runtime.val_png). ``task`` is the validation-set name ('rigid',
        'rigid_g', ...): file ``<name>_step<step>.png``. Never raises: failures are logged and
        skipped."""
        cfg = self.cfg
        try:
            from runtime import val_png as V
            idx = V.select_indices(grp["images"].shape[0], cfg.val_png_k, cfg.val_png_seed)
            A_sel, A_star = A[idx], np.asarray(grp["A_star"], np.float64)[idx]
            err = M.centered_errors(A_sel, A_star, self.S_np)["rms_mm"]
            z = P.LEVEL_SHAPE[self.level][2] // 2
            title = V.figure_title(task, step, metrics["val_%s/center_rms_mm" % task],
                                   metrics["val_%s/pair_rms_mm" % task], z)
            labels = ["val %d" % int(i) for i in np.asarray(grp["index"])[idx]]
            os.makedirs(cfg.val_png_dir, exist_ok=True)
            path = os.path.join(cfg.val_png_dir, "%s_step%07d.png" % (task, int(step)))
            V.write_val_png(path, grp["images"][idx], A_sel, A_star, P.LEVEL_SPACING[self.level],
                            title, err, gt_row=bool(cfg.val_png_gt_row),
                            mean_col=bool(cfg.val_png_mean_col), labels=labels)
            self.val_png_paths.append(path)
        except Exception as e:  # noqa: BLE001  (a PNG must never stop training)
            print("WARNING step %d: validation PNG for %s failed: %r" % (int(step), task, e),
                  file=sys.stderr, flush=True)
            if self.logger is not None:
                self.logger.log_event(int(step), {"event": "val_png_failed", "task": task,
                                                  "error": repr(e)})

    def validate(self, max_groups=None):
        """Metrics on the fixed validation sets, pooled over groups.

        ``val_<task>/...``: eval mode (moving BN statistics, dropout off), the source of truth
        (best checkpoint). ``val_<task>_trainmode/{center,pair}_rms_mm``: the same groups with
        training-mode BN (no statistics update, dropout off), to expose any train/eval gap.
        """
        out = {}
        for name, vs in self.val_sets.items():
            task = vs.task
            if task not in ("rigid", "affine"):
                continue
            groups = vs.groups if max_groups is None else vs.groups[:int(max_groups)]
            if not groups:
                continue
            t0 = time.perf_counter()
            errs, mats = self._val_errors(task, vs, groups, False)
            out.update(M.summarize(errs, "val_%s" % name))
            out["val_%s/groups" % name] = len(groups)
            out["val_%s/time_s" % name] = time.perf_counter() - t0
            pre = "val_%s_trainmode" % name
            tm = M.summarize(self._val_errors(task, vs, groups, True)[0], pre)
            for k in ("center_rms_mm", "pair_rms_mm"):
                out["%s/%s" % (pre, k)] = tm["%s/%s" % (pre, k)]
            g = int(self.cfg.val_png_group)
            if self.cfg.val_png_dir and 0 <= g < len(groups):
                self._write_val_png(name, groups[g], mats[g], out, int(self.step.numpy()))
        return out

    # ------------------------------------------------------------------ loop
    def _log(self, step, metrics):
        if self.logger is not None:
            self.logger.log_metrics(step, metrics)

    def _validate_and_log(self, step):
        val = self.validate()
        self.val_history.append(dict(val, step=step))
        self._log(step, val)
        if self.checkpoint is not None and self.cfg.best_metric in val:
            self.checkpoint.maybe_save_best(val[self.cfg.best_metric])
        return val

    def _update_collapse(self, step, task, res):
        """Count consecutive collapse-suspect finite steps; warn once per streak at the patience."""
        cfg = self.cfg
        suspect = (res["diag_feat_rel_spread"] < float(cfg.collapse_rel_spread)
                   or res["grad_norm"] < float(cfg.collapse_grad_norm))
        self.collapse_streak = self.collapse_streak + 1 if suspect else 0
        if self.collapse_streak != int(cfg.collapse_patience):
            return
        self.collapse_warnings += 1
        info = {"event": "collapse_suspect", "task": task, "consecutive_steps": self.collapse_streak,
                "rel_spread": res["diag_feat_rel_spread"], "grad_norm": res["grad_norm"],
                "dead_frac": res["diag_dead_frac"]}
        print("WARNING step %d: possible group collapse (%d consecutive steps with relative "
              "pre-head feature spread < %g or grad norm < %g; now rel_spread %.3g, grad_norm "
              "%.3g, dead_frac %.3f)" % (step, self.collapse_streak, cfg.collapse_rel_spread,
                                         cfg.collapse_grad_norm, res["diag_feat_rel_spread"],
                                         res["grad_norm"], res["diag_dead_frac"]),
              file=sys.stderr, flush=True)
        if self.logger is not None:
            self.logger.log_event(step, info)

    def _window_center_rms(self):
        w = int(self.cfg.overfit_window)
        vals = [h["center_rms_mm"] for h in self.history[-w:] if h["finite"]]
        if len(vals) < w:
            return None
        return float(np.mean(vals))

    def fit(self, num_steps=None):
        """Train until step == cfg.steps (num_steps=None) or for `num_steps` more steps."""
        cfg = self.cfg
        start = int(self.step.numpy())
        end = int(cfg.steps) if num_steps is None else start + int(num_steps)
        acc = M.MeanAccumulator()
        timing = M.MeanAccumulator()
        consecutive = 0
        saved_at = None
        last_val = {}
        best_window = None
        while int(self.step.numpy()) < end:
            s = int(self.step.numpy())
            task = self.schedule.draw(s)
            lr = lr_at(s, cfg.schedule)
            O.set_learning_rate(self.optimizer, lr)
            # Control files first (curriculum / G of this step), then the batch. With the DALI
            # prefetch the batch may still come from an earlier state (~1 batch); its emitted
            # matrices are the truth, so the targets stay correct either way.
            ctl_values = {t: c.update(s)[0] for t, c in self.controls.items()}
            t0 = time.perf_counter()
            batch, extras = self.split_batch(task, next(self.sources[task]))
            t1 = time.perf_counter()
            res = self.train_step(task, batch)
            t2 = time.perf_counter()
            done = s + 1
            rec = {"step": done, "task": task, "lr": lr, "data_ms": 1e3 * (t1 - t0),
                   "compute_ms": 1e3 * (t2 - t1)}
            rec.update(res)
            if task in ctl_values:
                cm = self.control_metrics(task, extras, ctl_values[task])
                rec.update(cm)
                acc.update({"train_%s/%s" % (task, k): v for k, v in cm.items()})
            self.history.append(rec)
            timing.update({"time/data_ms": rec["data_ms"], "time/compute_ms": rec["compute_ms"]})
            if not res["finite"]:
                self.nonfinite_total += 1
                consecutive += 1
                if self.logger is not None:
                    self.logger.log_event(done, {"event": "nonfinite_step_skipped", "task": task,
                                                 "loss": res["loss"], "grad_norm": res["grad_norm"],
                                                 "nonfinite_total": self.nonfinite_total})
                if consecutive > int(cfg.nonfinite_patience):
                    raise RuntimeError("%d consecutive non-finite steps (last at step %d); stopping"
                                       % (consecutive, done))
            else:
                consecutive = 0
                pre = "train_%s/" % task
                acc.update({pre + "loss": res["loss"], pre + "center": res["center"],
                            pre + "pair": res["pair"], "train/grad_norm": res["grad_norm"]})
                acc.update({pre + k: res[k] for k in DIAG_KEYS})
                self._update_collapse(done, task, res)
            if cfg.log_every and (done % int(cfg.log_every) == 0 or done == end):
                m = acc.result()
                for t in self.sources:
                    pre = "train_%s/" % t
                    if pre + "center" in m:
                        m[pre + "center_rms_mm"] = math.sqrt(max(m[pre + "center"], 0.0))
                        m[pre + "pair_rms_mm"] = math.sqrt(max(m[pre + "pair"], 0.0))
                m.update(timing.result())
                m["lr"] = lr
                m["nonfinite_total"] = self.nonfinite_total
                m["train/collapse_suspect"] = self.collapse_streak
                peak = _peak_gpu_mb()
                if peak is not None:
                    m["gpu_peak_mb"] = peak
                self._log(done, m)
                acc.reset()
                timing.reset()
            if self.val_sets and cfg.val_every and done % int(cfg.val_every) == 0:
                last_val = self._validate_and_log(done)
            ckpt_every = cfg.ckpt_every if cfg.ckpt_every is not None else cfg.val_every
            if self.checkpoint is not None and ckpt_every and done % int(ckpt_every) == 0:
                self.checkpoint.save_last()
                saved_at = done
            if cfg.overfit:
                w = self._window_center_rms()
                if w is not None:
                    best_window = w if best_window is None else min(best_window, w)
                    if w < float(cfg.overfit_threshold_mm):
                        self.reached = True
                        break
        final = int(self.step.numpy())
        # Validation PNGs also cover the final step: validate there unless that just happened.
        if (cfg.val_png_dir and self.val_sets and cfg.val_every and final > start
                and not (self.val_history and self.val_history[-1]["step"] == final)):
            last_val = self._validate_and_log(final)
        if self.checkpoint is not None and saved_at != final and final > start:
            self.checkpoint.save_last()
        if self.logger is not None:
            self.logger.flush()
        return {"start_step": start, "end_step": final, "nonfinite_total": self.nonfinite_total,
                "collapse_warnings": self.collapse_warnings,
                "trace_counts": dict(self.trace_counts), "reached": self.reached,
                "best_window_center_rms_mm": best_window, "last_val": last_val}
