"""Custom training loop for the linear tasks (design 5.2): task schedule, one tf.function train
step per task, NaN guard, validation, logging and checkpoint hooks.

Train step (one per task, traced once; input_signature = the DALI output shapes for the fixed N
of the level, `data_loading.dali_loader.linear_output_signature`)::

    step(images [N,X,Y,Z,1], T_dali [N,3,4] (, Rt_dali [N,3,4] for rigid))   all float32
      batch = LinearPostprocess(...)            # flips, mm matrices, targets, pairs (no tape)
      non-finite input images -> zeroed, step marked non-finite (the barycenter's matrix
        inverse raises on NaN instead of propagating it)
      backup BN moving statistics
      with GradientTape: A = model(images, task, training=True)["A"]; loss = linear_loss(...)
      grads -> drop None (e.g. the scale/shear head in rigid steps; static, at trace time)
      finite = isfinite(loss) & isfinite(global_norm(grads))
      tf.cond(finite, apply_gradients, skip); if not finite: restore BN statistics
      step += 1;  returns loss terms, grad_norm, finite, n_vars_updated

Images stay float32 (DALI cannot warp float16); the model casts them to its compute dtype.
Under --amp the LossScaleOptimizer is always called (it skips non-finite updates itself and
lowers its scale); the BN rollback and the logs use our finite flag.

`rebuild_steps()` re-creates every step function. Call it after changing anything read at trace
time, in particular `model.set_bn_stats_frozen(...)` (an already traced step keeps updating
the statistics otherwise).
"""
import collections
import dataclasses
import math
import random
import time
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from data_loading import dali_loader
from data_loading import params as P
from data_loading.postprocess import LinearPostprocess
from runtime import losses
from runtime import metrics as M
from runtime import optimizers as O
from runtime.args import parse_task_ratio  # noqa: F401  (re-exported)
from runtime.schedulers import ScheduleConfig, lr_at

POSTPROCESS_SEED_OFFSET = {"rigid": 101, "affine": 202}


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
      val_sets: optional {task: `runtime.fixed_data.ValidationSet`}.
    """

    def __init__(self, model, optimizer, sources, S, cfg, logger=None, checkpoint=None, val_sets=None):
        self.model = model
        self.optimizer = optimizer
        self.sources = dict(sources)
        self.cfg = cfg
        self.logger = logger
        self.checkpoint = checkpoint
        self.val_sets = dict(val_sets or {})
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
                scaled = opt.get_scaled_loss(terms["loss"]) if amp else terms["loss"]
            grads = tape.gradient(scaled, tvars)
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
            res.update(grad_norm=gnorm, finite=finite, n_vars_updated=tf.constant(len(pairs)))
            return res

        if task == "rigid":
            return tf.function(lambda images, t_dali, rt_dali: body(images, t_dali, rt_dali),
                               input_signature=signature)
        return tf.function(lambda images, t_dali: body(images, t_dali), input_signature=signature)

    def train_step(self, task, batch_tuple):
        """Run one step of `task` on a DALI-format tuple. Returns Python scalars."""
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
        fn = self._val_fn(task, vs.group_size, train_mode)
        errs = []
        for g, grp in enumerate(groups):
            A = fn(tf.constant(grp["images"])).numpy().astype(np.float64)
            errs.append(M.group_errors(A, grp["A_star"], grp["T_target"], self.S_np, seed=g))
        return M.concat_errors(errs)

    def validate(self, max_groups=None):
        """Metrics on the fixed validation sets, pooled over groups.

        ``val_<task>/...``: eval mode (moving BN statistics, dropout off), the source of truth
        (best checkpoint). ``val_<task>_trainmode/{center,pair}_rms_mm``: the same groups with
        training-mode BN (no statistics update, dropout off), to expose any train/eval gap.
        """
        out = {}
        for task, vs in self.val_sets.items():
            if task not in ("rigid", "affine"):
                continue
            groups = vs.groups if max_groups is None else vs.groups[:int(max_groups)]
            if not groups:
                continue
            t0 = time.perf_counter()
            out.update(M.summarize(self._val_errors(task, vs, groups, False), "val_%s" % task))
            out["val_%s/groups" % task] = len(groups)
            out["val_%s/time_s" % task] = time.perf_counter() - t0
            pre = "val_%s_trainmode" % task
            tm = M.summarize(self._val_errors(task, vs, groups, True), pre)
            for k in ("center_rms_mm", "pair_rms_mm"):
                out["%s/%s" % (pre, k)] = tm["%s/%s" % (pre, k)]
        return out

    # ------------------------------------------------------------------ loop
    def _log(self, step, metrics):
        if self.logger is not None:
            self.logger.log_metrics(step, metrics)

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
            t0 = time.perf_counter()
            batch = next(self.sources[task])
            t1 = time.perf_counter()
            res = self.train_step(task, batch)
            t2 = time.perf_counter()
            done = s + 1
            rec = {"step": done, "task": task, "lr": lr, "data_ms": 1e3 * (t1 - t0),
                   "compute_ms": 1e3 * (t2 - t1)}
            rec.update(res)
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
                peak = _peak_gpu_mb()
                if peak is not None:
                    m["gpu_peak_mb"] = peak
                self._log(done, m)
                acc.reset()
                timing.reset()
            if self.val_sets and cfg.val_every and done % int(cfg.val_every) == 0:
                last_val = self.validate()
                self.val_history.append(dict(last_val, step=done))
                self._log(done, last_val)
                if self.checkpoint is not None and cfg.best_metric in last_val:
                    self.checkpoint.maybe_save_best(last_val[cfg.best_metric])
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
        if self.checkpoint is not None and saved_at != final and final > start:
            self.checkpoint.save_last()
        if self.logger is not None:
            self.logger.flush()
        return {"start_step": start, "end_step": final, "nonfinite_total": self.nonfinite_total,
                "trace_counts": dict(self.trace_counts), "reached": self.reached,
                "best_window_center_rms_mm": best_window, "last_val": last_val}
