"""Custom training loop (design 5.2): task schedule, one tf.function train step per task (rigid,
affine and, milestone 6, deform), NaN guard, validation, logging and checkpoint hooks.

Train step (one per task, traced once; input_signature = the DALI output shapes for the fixed N
of the level, `data_loading.dali_loader.linear_output_signature`)::

    step(images [N,X,Y,Z,1], T_dali [N,3,4] (, Rt_dali [N,3,4] for rigid))   all float32
      batch = LinearPostprocess(...)            # flips, mm matrices, targets, pairs (no tape)
      non-finite input images -> zeroed, step marked non-finite (the barycenter's matrix
        inverse raises on NaN instead of propagating it)
      backup BN moving statistics
      with GradientTape: out = model(images, task, training=True); terms = linear_loss(out["A"])
        objective = w_task * terms["loss"] / loss_scale_mm2   (terms stay in mm^2 for the logs;
                    w_task = task_loss_weights[task], default 1; each factor skipped at trace
                    time when it is 1, so the default graph is unchanged)
      diag = collapse_diagnostics(out["pre_head"], out["params"])   (no gradient, every step)
      grads -> drop None (e.g. the scale/shear head in rigid steps; static, at trace time)
      finite = isfinite(loss) & isfinite(global_norm(grads))
      optional --task-grad-norm (design 5.2 v3.10; default off = no op): grads *= target /
        max(g, eps) when the raw norm g >= min (else 1), then * w_task (w not in the objective)
      enc_diag = diag_enc_grad_norm_raw / _eff (encoder-only gradient norms, read-only)
      tf.cond(finite, apply_gradients, skip); if not finite: restore BN statistics
      step += 1;  returns loss terms, diag_*, grad_norm, finite, n_vars_updated

Task balance diagnostics (milestone 7, logged for every task as window means):
``train_<task>/diag_enc_grad_norm_raw`` = encoder-only gradient norm of the unscaled task loss
(mm^2 for linear, Dice units for deform), ``train_<task>/diag_enc_grad_norm_eff`` = the encoder
norm actually applied (after loss scale, task weight and the global-clip factor min(1, clipnorm
/ grad_norm)), ``train_<task>/grad_norm`` = the task's pre-clip total norm (``train/grad_norm``
pools all tasks as before). One Adam is shared by all tasks, so _eff of the linear tasks vs
deform shows which task dominates its second-moment estimate on the encoder. With
--task-grad-norm, grad_norm stays the RAW (unweighted) norm, _eff includes the normalisation
scale and weight, and ``train_<task>/{grad_norm_scale, grad_norm_gated_frac, grad_norm_normed}``
are logged (`TASK_GRAD_NORM_KEYS`; window mean of gated_frac = fraction of gated steps).

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

Mixed tasks (design 5.2, milestone 5): `TaskSchedule` draws the task of each step from (seed,
step) only, every task has its own tf.function (traced once), and all of them share ONE
optimizer. That optimizer is built on ALL trainable variables in `__init__`, so the affine-only
scale/shear head has its Adam slots before the first affine step even when the run starts with
rigid steps (whose None gradients for that head are dropped at trace time); a later first affine
step creates no variable and retraces nothing.

`rebuild_steps()` re-creates every step function. Call it after changing anything read at trace
time, in particular `model.set_bn_stats_frozen(...)` (an already traced step keeps updating
the statistics otherwise).

Deformable step (milestone 6, traced once; input_signature =
`dali_loader.deform_output_signature` for the deform group size)::

    step(images float32 [N,X,Y,Z,1], labels uint8 [N,X,Y,Z,1] (, T_dali [N,3,4] residual affine))
      batch = DeformPostprocess(...)            # flips (images and labels), one-hot, pairs
      non-finite images -> zeroed and the step skipped (as linear); backup BN statistics
      with GradientTape: out = model(images, "deform", training=True)
        terms = deform_loss(out, batch)   # lambda_dice_tmpl template soft Dice + lambda_pdice
                                          # pairwise soft Dice + lambda_smooth smoothness
                                          # [+ lambda_ncc_tmpl (1 - template LNCC)
                                          #  + lambda_ncc_pair (1 - pairwise LNCC)] (v3.9; LNCC
                                          # terms only when their weight > 0; no mm^2 scaling)
        objective = w_deform * terms["loss"]   (task_loss_weights, default 1)
      logged: train_deform/{loss, tmpl, pdice, smooth, soft_dice_tmpl, soft_dice_pair} and, per
        active LNCC term, train_deform/{ncc_tmpl, ncc_tmpl_loss} / {ncc_pair, ncc_pair_loss}
        (ncc_* = mean squared LNCC, higher is better)
      diag = deform_diagnostics(out)      # same diag_feat_* / diag_dead_frac keys as linear on
                                          # the D1 decoder features, + v / u / folding stats
      static None filter, finite flag, optional --task-grad-norm (grads *= target / max(g, eps)
        when g >= min, then * w_deform; w not in the objective), tf.cond apply, BN rollback,
        step += 1 (as linear)

None gradients per step (dropped at trace time; the ONE optimizer has slots for every trainable
variable from `__init__`, so no step creates variables or retraces; BN statistics of a part that
did not run are untouched):

| step   | None-gradient variables                                                 |
|---|---|
| rigid  | scale/shear head, the whole deform decoder                              |
| affine | the whole deform decoder                                                |
| deform | the whole linear decoder (reductions, dense, heads, head_norm_act)      |

Deformable validation (`DeformValidationSet`, ``val_deform/...``, eval mode = source of truth):
template_dice / pair_dice (hard Dice of argmax-warped one-hots; pairs: all ordered pairs, or a
fixed seeded subset of ``val_deform_max_pairs``), neg_jac_pct, inv_consistency(_max)_mm,
centrality_mm, raw_centrality_mm, pair_dice_loss = 1 - pair_dice (deform best metric),
ncc_template / ncc_pair (v3.9: mean squared LNCC on the images with the training definition,
always computed, whatever the LNCC weights), the cached identity baseline pair_dice_identity /
template_dice_identity / ncc_template_identity / ncc_pair_identity, and
``val_deform_trainmode/{template_dice, pair_dice}``. ``val/mean_pair_rms_mm`` covers the linear
tasks only. Best files: linear tasks -> ``ckpt/best`` keeps the linear rule; deform-only ->
``ckpt/best`` on ``val_deform/pair_dice_loss``; deform + linear -> also ``extra_best`` (main.py
wires ``ckpt/best_deform``). Overfit stop (`overfit_status`): every linear task's window mean
center RMS below the threshold (unchanged) AND the deform window mean soft template Dice >=
``overfit_dice_threshold`` with window mean folding fraction < ``overfit_fold_max``.

Level cascade (milestone 8, `.agents/tasks/milestone-8/design.md` sections 4-8; ``Trainer(cascade=
models.cascade.FrozenStage)``, default None = every function below is built exactly as without
it). The frozen level-4 stage is never in the optimizer, runs outside the tape in eval mode and
its outputs are stop_gradient'ed. Chosen at trace time (``if self.cascade is not None``)::

    linear step(images, T_dali(, Rt_dali), skip bool[])          deform step(images, labels, skip)
      batch = pp(...); x = finite-zeroed images                    batch = pp(...); x = ... (same)
      A_prev = frozen.linear(x, task)          (pool to level 4)   U, U_inv = frozen.deform(x)
      [p_skip > 0: A_prev = I if skip]                             [p_skip > 0: U = U_inv = 0 if skip]
      x_in = x o A_prev^-1                                         x_in = warp(x, U)
      tape: out = model(x_in); A = Center(out.A_raw A_prev)        tape: out = model(x_in) (residual)
            linear_loss(A, batch, S)                                     (u, u_inv) = compose_cascade
                                                                         deform_loss(out_c, batch)

The deform losses use the ORIGINAL batch images / one-hot with the composed field (u_res = the
residual u, ``diag_u_res_max_vox`` logged). The skip input exists only in cascade signatures;
with ``cascade_skip_prob == 0`` its ``tf.where`` ops (name scope ``cascade_skip``) are not built.
`fit` draws skip = ``random.Random('cascade-skip-<seed>-<step>').random() < p`` (stateless) and
logs ``train/cascade_skip_frac`` only when p > 0; with a cascade it also logs
``train_<task>/{data_ms, compute_ms}`` window means. Validation: the same pool -> frozen (eval)
-> pre-warp -> trainable -> compose forward inside the traced validation function (never
skipped); eval passes also collect the frozen estimates, giving the cached frozen-stage
baselines ``val_<set>_frozen/...`` (`_frozen_baseline_metrics`) and, for deform sets,
``val_<set>/composed_mean_disp_mm`` (RMS over voxels of |mean_i u_i| in mm).
`recalibrate_bn` re-estimates the trainable stage's BN statistics (forward-only, exact average
of per-batch statistics) and optionally freezes them (design section 7).

3-level cascade (milestone 9 phase 2, `.agents/tasks/milestone-9-phase2/design-m9-phase2.md`
sections 2-3): at level 1 ``cascade`` is a `models.cascade.FrozenChain` (frozen level 4 ->
level 2). Dispatch is the Python int ``self._n_stages`` (0 none, 1 = `FrozenStage`, the code
above verbatim, 2 = chain; ``getattr(cascade, 'n_stages', 1)``). The chain's ``linear`` /
``deform`` give the composed level-2 estimate at level 1; everything after (pre-warp, compose,
losses) is unchanged. Skip (p > 0) drops the LAST frozen stage: A_prev = A4, U = U4 via
``linear_stages`` / ``deform_stages`` and `CS.skip_to`. Eval-mode validation adds the level-4
alone baselines ``val_<set>_frozen4/...`` next to ``val_<set>_frozen/...`` (levels 4 + 2), and
level-1 runs log the validation wall time ``val/wall_s``.
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
from data_loading.postprocess import MIN_PAIRS, DeformPostprocess, LinearPostprocess, num_pairs_static
from models import cascade as CS
from models import diffeo
from models import layers as L
from models import transforms as tfm
from runtime import losses
from runtime import metrics as M
from runtime import optimizers as O
from runtime.args import parse_task_ratio  # noqa: F401  (re-exported)
from runtime.schedulers import ScheduleConfig, lr_at

POSTPROCESS_SEED_OFFSET = {"rigid": 101, "affine": 202, "deform": 303}
LINEAR_TASKS = P.TASKS
DEFORM = P.DEFORM_TASK
# Combined best-checkpoint metric (milestone 5): mean of val_<task>/pair_rms_mm over the trained
# linear tasks; the default --best-metric when more than one linear task is trained.
MEAN_PAIR_METRIC = "val/mean_pair_rms_mm"
MEAN_PAIR_ALIASES = ("mean_pair_rms_mm", MEAN_PAIR_METRIC)
# Deformable best-checkpoint metric (milestone 6): 1 - mean pairwise hard Dice (lower is better).
DEFORM_BEST_METRIC = "val_deform/pair_dice_loss"


def resolve_best_metric(best_metric, tasks):
    """Validation key of the 'best' checkpoint. None: over the LINEAR tasks, `MEAN_PAIR_METRIC`
    if more than one is trained, else ``val_<task>/pair_rms_mm``; deform-only:
    `DEFORM_BEST_METRIC`. 'mean_pair_rms_mm' is an alias of `MEAN_PAIR_METRIC`; any other value
    is used as given."""
    lin = [t for t in tasks if t in LINEAR_TASKS]
    if best_metric is None:
        if not lin:
            return DEFORM_BEST_METRIC
        return MEAN_PAIR_METRIC if len(lin) > 1 else "val_%s/pair_rms_mm" % lin[0]
    if best_metric in MEAN_PAIR_ALIASES:
        return MEAN_PAIR_METRIC
    return best_metric


DIAG_KEYS = ("diag_feat_spread", "diag_feat_rel_spread", "diag_dead_frac", "diag_rot10_norm",
             "diag_head_spread")
DEFORM_LOSS_KEYS = ("loss", "tmpl", "pdice", "smooth", "soft_dice_tmpl", "soft_dice_pair")
# LNCC terms (v3.9), present in a deform step's results only when their weight is > 0.
DEFORM_NCC_KEYS = ("ncc_tmpl", "ncc_pair", "ncc_tmpl_loss", "ncc_pair_loss")
# Per-task encoder gradient norms (milestone 7, `Trainer._enc_grad_diagnostics`), logged as
# train_<task>/<key> window means next to train_<task>/grad_norm (the task's pre-clip norm).
ENC_GRAD_KEYS = ("diag_enc_grad_norm_raw", "diag_enc_grad_norm_eff")
DEFORM_DIAG_KEYS = ("diag_feat_spread", "diag_feat_rel_spread", "diag_dead_frac", "diag_v_rms_vox",
                    "diag_v_raw_mean_rms_vox", "diag_u_max_vox", "diag_fold_frac")
# Per-task gradient normalisation (design 5.2 v3.10, `task_grad_norm_scale`): eps of the
# division and the result keys, present in a step's results only with task_grad_norm > 0.
# grad_norm_scale = applied normalisation factor (1 when gated), grad_norm_gated_frac = 1 if the
# finite raw norm was below task_grad_norm_min, grad_norm_normed = global norm of the gradients
# passed to apply_gradients (after normalisation and task weight, before the optimizer's clip).
TASK_GRAD_NORM_EPS = 1e-12
TASK_GRAD_NORM_KEYS = ("grad_norm_scale", "grad_norm_gated_frac", "grad_norm_normed")
# Minimum pair count (milestone 9 phase 2, design-m9-phase2 section 5): number of sampled training
# pairs of a step, in its results (and logged as train_<task>/n_pairs) only with min_pairs 'n'.
PAIR_COUNT_KEY = "n_pairs"
# Level cascade (milestone 8): extra deform-step diagnostics (max |u_res| of the level-2 residual,
# cascade steps only), the per-step skip flag key and the BN recalibration gap metrics
# (|eval - trainmode| per validation set, `Trainer.recalibrate_bn`).
DEFORM_CASCADE_DIAG_KEYS = ("diag_u_res_max_vox",)
CASCADE_SKIP_KEY = "train/cascade_skip_frac"
# Frozen chain (milestone 9 phase 2, level 1): frozen fields of the eval-mode deform validation
# forward (U, U_inv, V = levels 4 + 2 composed; U4, U4_inv, V4 = level 4 alone) and the
# wall-time key of a whole validation pass (logged at level 1 only).
CHAIN_DEFORM_FROZEN_KEYS = ("U", "U_inv", "V", "U4", "U4_inv", "V4")
VAL_WALL_KEY = "val/wall_s"
BN_GAP_METRICS = {"linear": ("center_rms_mm", "pair_rms_mm"), DEFORM: ("template_dice", "pair_dice")}


def cascade_skip_draw(seed, step, p):
    """Stateless per-step skip draw of the cascade (design section 6): True with probability `p`."""
    return random.Random("cascade-skip-%d-%d" % (int(seed), int(step))).random() < float(p)


def bn_recal_plan(num_batches, ratio):
    """BN recalibration task plan (design section 7 step 2): ordered list of `num_batches` task names.

    Largest-remainder allocation of K over the (ordered) task ratio (ties -> earlier task), every
    task >= 1 (taken from the task with the most slots), interleaved round-robin in ratio order
    (K = 64, rigid:1,affine:1,deform:2 -> 16 / 16 / 32: r a d r a d ... then 16 x d)."""
    k = int(num_batches)
    tasks = [t for t, v in ratio.items() if float(v) > 0]
    if k < len(tasks):
        raise ValueError("bn recalibration needs >= %d batches (one per task), got %d" % (len(tasks), k))
    total = sum(float(ratio[t]) for t in tasks)
    quota = [k * float(ratio[t]) / total for t in tasks]
    counts = [int(math.floor(q)) for q in quota]
    order = sorted(range(len(tasks)), key=lambda i: (-(quota[i] - counts[i]), i))
    for i in order[:k - sum(counts)]:
        counts[i] += 1
    for i in range(len(tasks)):
        while counts[i] == 0:
            j = max(range(len(tasks)), key=lambda m: (counts[m], -m))
            counts[j] -= 1
            counts[i] += 1
    left = list(counts)
    plan = []
    while len(plan) < k:
        for i, t in enumerate(tasks):
            if left[i] > 0:
                plan.append(t)
                left[i] -= 1
    return plan


def task_grad_norm_scale(gnorm, target, min_norm, eps=TASK_GRAD_NORM_EPS):
    """(scale, gated) float32 scalars of the per-task gradient normalisation.

    scale = target / max(gnorm, eps) if gnorm is finite and >= min_norm, else 1 (gated steps and
    non-finite norms are never rescaled). gated = 1.0 if gnorm is finite and below min_norm.
    ``target * (1.0 / max(.))`` mirrors `tf.clip_by_global_norm`'s ``clip * (1.0 / norm)``, so
    normalising equals clipping exactly when the norm exceeds the clip (both branches are
    computed; the divided one is finite because of eps).
    """
    gnorm = tf.convert_to_tensor(gnorm, tf.float32)
    finite = tf.math.is_finite(gnorm)
    ok = tf.logical_and(finite, gnorm >= float(min_norm))
    scale = tf.where(ok, float(target) * (1.0 / tf.maximum(gnorm, float(eps))), tf.ones_like(gnorm))
    gated = tf.cast(tf.logical_and(finite, tf.logical_not(ok)), tf.float32)
    return scale, gated


def scale_gradients(gs, factor):
    """Every gradient times the scalar `factor` (cast to the gradient dtype; IndexedSlices kept)."""
    out = []
    for g in gs:
        if isinstance(g, tf.IndexedSlices):
            out.append(tf.IndexedSlices(g.values * tf.cast(factor, g.values.dtype), g.indices,
                                        g.dense_shape))
        else:
            out.append(g * tf.cast(factor, g.dtype))
    return out


def deform_diagnostics(out):
    """Diagnostics of one deform forward pass (TF, float32 scalars, no gradient needed).

    Same collapse keys as `collapse_diagnostics`, on the D1 decoder features ``dec_feat``
    [N, X, Y, Z, C]: diag_feat_spread (mean over voxels and channels of the std across the
    group), diag_feat_rel_spread (that / mean |dec_feat|), diag_dead_frac (fraction of channels
    <= 0 at every voxel of every image). Plus diag_v_rms_vox (RMS |v| of the centered SVF),
    diag_v_raw_mean_rms_vox (RMS |mean_i v_raw_i|: what the centering removed), diag_u_max_vox
    (max |u|) and diag_fold_frac (mean folding fraction J <= 0 over the group).
    """
    f = tf.cast(out["dec_feat"], tf.float32)
    spread = tf.reduce_mean(tf.math.reduce_std(f, axis=0))
    v = tf.cast(out["v"], tf.float32)
    v_raw_mean = tf.reduce_mean(tf.cast(out["v_raw"], tf.float32), axis=0)
    u = tf.cast(out["u"], tf.float32)
    return {"diag_feat_spread": spread,
            "diag_feat_rel_spread": tf.math.divide_no_nan(spread, tf.reduce_mean(tf.abs(f))),
            "diag_dead_frac": tf.reduce_mean(tf.cast(tf.reduce_all(f <= 0.0, axis=[0, 1, 2, 3]),
                                                     tf.float32)),
            "diag_v_rms_vox": tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(v), axis=-1))),
            "diag_v_raw_mean_rms_vox": tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(v_raw_mean), axis=-1))),
            "diag_u_max_vox": tf.reduce_max(tf.norm(u, axis=-1)),
            "diag_fold_frac": tf.reduce_mean(diffeo.folding_fraction(u))}


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
    # Per-task objective multipliers (milestone 7; missing task = 1.0): linear objective =
    # w * loss / loss_scale_mm2, deform objective = w * loss. Logged losses stay unweighted.
    task_loss_weights: Dict[str, float] = dataclasses.field(default_factory=dict)
    # Per-task gradient normalisation (design 5.2 v3.10; 0 = off): grads *= target / max(g, eps)
    # if the raw norm g >= task_grad_norm_min, then * task weight (the weight leaves the objective).
    task_grad_norm: float = 0.0
    task_grad_norm_min: float = 1e-3
    p_pair: float = 0.1
    flip_prob: float = 0.5
    target_iters: int = 10
    amp: bool = False
    seed: int = 0
    schedule: ScheduleConfig = dataclasses.field(default_factory=ScheduleConfig)
    nonfinite_patience: int = 20
    # None: val/mean_pair_rms_mm with more than one task, else val_<task>/pair_rms_mm
    # (`resolve_best_metric`; 'mean_pair_rms_mm' is accepted as an alias).
    best_metric: Optional[str] = None
    # Overfit mode stops when the overfit_window-step mean center RMS of EVERY trained task
    # (its own last overfit_window finite steps) is below overfit_threshold_mm.
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
    # Deformable task (milestone 6): loss = template soft Dice + lambda_pdice * pairwise soft Dice
    # (p_pdice of the ordered pairs) + lambda_smooth * smoothness(u, smooth_order).
    lambda_pdice: float = 1.0
    p_pdice: float = 0.1
    lambda_smooth: float = 0.5
    smooth_order: int = 1
    n_classes: int = P.N_TRAIN_LABELS
    deform_residual_affine: bool = False
    deform_group_size: Optional[int] = None        # None: group_size
    # Overfit stop of the deform task: window mean soft template Dice >= overfit_dice_threshold
    # and window mean folding fraction < overfit_fold_max.
    overfit_dice_threshold: float = 0.9
    overfit_fold_max: float = 0.001
    # Deform validation: all ordered pairs of a group if at most this many, else a fixed subset.
    val_deform_max_pairs: int = 64
    # LNCC (design 6.2 v3.9): deform loss = lambda_dice_tmpl * template Dice + ... +
    # lambda_ncc_tmpl * (1 - template LNCC) + lambda_ncc_pair * (1 - pairwise LNCC). The LNCC
    # weights default to 0 (terms not built); ncc_window None = P.ncc_window_for_level(level).
    lambda_dice_tmpl: float = 1.0
    lambda_ncc_tmpl: float = 0.0
    lambda_ncc_pair: float = 0.0
    ncc_window: Optional[int] = None
    # Level cascade (milestone 8, design section 6 / section 11): per-step probability of skipping the frozen
    # stage in training (0 = off, no skip ops built) and the deform validation pair chunk size
    # (16 = the milestone-6 value).
    cascade_skip_prob: float = 0.0
    val_pair_chunk: int = 16
    # --val-image-chunk (level-1 validation memory): > 0 = the deform validation metrics are computed
    # k images (and val_pair_chunk pairs) at a time (`_deform_errors_chunked`); 0 = unchunked (today).
    val_image_chunk: int = 0
    # Memory knobs of the deform TRAINING step (milestone 9 design 3.3 / 3.4; validation keeps
    # today's paths): lean_sampler must equal the model's `lean_sampler` (checked in Trainer);
    # it is passed to compose_cascade and deform_loss; lean_losses to deform_loss.
    lean_sampler: bool = False
    lean_losses: bool = False
    # Minimum sampled training pairs per step for the pairwise losses (p_pair / p_pdice; design-m9-phase2
    # section 5): '1' = today, 'n' = the group size (k == N: random N-cycle). Validation unaffected.
    min_pairs: str = "1"
    # TF-side intensity augmentation (--intensity-aug-device tf): a `params.IntensityAugConfig`
    # passed to every post-process (DALI then runs without it), or None (default: no op).
    intensity_aug_tf: Optional[object] = None
    # --pair-recompute: the deform TRAINING step's pairwise Dice / LNCC one pair at a time under
    # tf.recompute_grad (`losses.pair_terms_recompute`); False = today's ops. Validation unaffected.
    pair_recompute: bool = False


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
      val_sets: optional {name: `runtime.fixed_data.ValidationSet` or `DeformValidationSet`};
        the name is the metric prefix ``val_<name>`` (e.g. 'rigid', 'rigid_g' for the set with a
        global transform, 'deform'), the set's ``task`` selects the model head.
      controls: optional {task: `runtime.curriculum.TaskControl`} with writers attached
        (curriculum / global transform). Each step first writes every task's control values of
        that step; batches then carry G_dali and control_version, which are logged and dropped
        before the step function. None (default): no control file is touched.
      cascade: optional `models.cascade.FrozenStage` (milestone 8, level 2) or
        `models.cascade.FrozenChain` (milestone 9, level 1): the frozen previous level(s)
        (module docstring "Level cascade" / "3-level cascade"). It is called outside the tape in eval
        mode and its variables are never given to the optimizer (`O.build_optimizer` stays on
        ``model.trainable_variables``). None (default): no cascade code is traced.

    Memory knobs (milestone 9): ``cfg.lean_sampler`` must equal ``model.lean_sampler``
    (ValueError otherwise); the deform training step passes it to `diffeo.compose_cascade` and
    `losses.deform_loss`, and ``cfg.lean_losses`` to `losses.deform_loss`. Validation keeps the
    default paths (no tape, same forward values).
    """

    def __init__(self, model, optimizer, sources, S, cfg, logger=None, checkpoint=None, val_sets=None,
                 controls=None, cascade=None):
        self.model = model
        self.cascade = cascade
        self.optimizer = optimizer
        self.sources = dict(sources)
        self.cfg = cfg
        self.logger = logger
        self.checkpoint = checkpoint
        self.val_sets = dict(val_sets or {})
        self.controls = dict(controls or {})
        self.level = P.check_level(cfg.level)
        model_lean = bool(getattr(model, "lean_sampler", False))
        if model_lean != bool(cfg.lean_sampler):
            raise ValueError("lean_sampler: model %s vs TrainConfig %s" % (model_lean, bool(cfg.lean_sampler)))
        # Chain dispatch (milestone 9 phase 2, design 2.1): a Python int fixed at construction,
        # 0 = no cascade, 1 = single frozen stage (milestone-8 code verbatim), 2 = frozen chain.
        self._n_stages = 0 if cascade is None else int(getattr(cascade, "n_stages", 1))
        if self._n_stages not in (0, 1, 2):
            raise ValueError("cascade: n_stages must be 1 or 2, got %r" % self._n_stages)
        if self._n_stages == 2 and self.level != 1:
            raise ValueError("cascade: a frozen chain (n_stages 2) needs a level-1 trainer, got level %d"
                             % self.level)
        if cascade is not None:
            if CS.PREV_LEVEL.get(self.level) != cascade.level:
                raise ValueError("cascade: level %d has no frozen level-%d stage (PREV_LEVEL %s)"
                                 % (self.level, cascade.level, CS.PREV_LEVEL))
            if cfg.deform_residual_affine:
                raise ValueError("cascade: deform_residual_affine is not supported (design section 5)")
            p = float(cfg.cascade_skip_prob)
            if not (math.isfinite(p) and 0.0 <= p < 1.0):
                raise ValueError("cascade_skip_prob must be in [0, 1), got %r" % cfg.cascade_skip_prob)
        self.S_np = np.asarray(S, np.float64)
        self.S = tf.constant(self.S_np, tf.float32)
        self.schedule = TaskSchedule(cfg.task_ratio, cfg.seed)
        missing = [t for t in self.schedule.tasks if t not in self.sources]
        if missing:
            raise ValueError("no data source for task(s) %s" % missing)
        for t, w in dict(cfg.task_loss_weights or {}).items():
            if t not in LINEAR_TASKS + (DEFORM,):
                raise ValueError("task_loss_weights: unknown task %r" % t)
            if not (math.isfinite(float(w)) and float(w) > 0):
                raise ValueError("task_loss_weights[%r] must be finite and > 0, got %r" % (t, w))
        for k in ("task_grad_norm", "task_grad_norm_min"):
            val = float(getattr(cfg, k))
            if not (math.isfinite(val) and val >= 0):
                raise ValueError("%s must be finite and >= 0, got %r" % (k, getattr(cfg, k)))
        self.task_grad_norm = float(cfg.task_grad_norm)
        self.task_grad_norm_min = float(cfg.task_grad_norm_min)
        for k in ("lambda_dice_tmpl", "lambda_ncc_tmpl", "lambda_ncc_pair"):
            val = float(getattr(cfg, k))
            if not (math.isfinite(val) and val >= 0):
                raise ValueError("%s must be finite and >= 0, got %r" % (k, getattr(cfg, k)))
        self.ncc_window = losses.check_ncc_window(
            P.ncc_window_for_level(self.level) if cfg.ncc_window is None else cfg.ncc_window,
            P.LEVEL_SHAPE[self.level])
        self.best_metric = resolve_best_metric(cfg.best_metric, self.schedule.tasks)
        self.linear_tasks = [t for t in self.schedule.tasks if t in LINEAR_TASKS]
        # Deform + linear: 'best' keeps the linear rule, the deform best goes to an extra file.
        self.extra_best = ({DEFORM: DEFORM_BEST_METRIC}
                           if DEFORM in self.schedule.tasks and self.linear_tasks else {})
        if cfg.min_pairs not in MIN_PAIRS:
            raise ValueError("min_pairs must be one of %s, got %r" % (MIN_PAIRS, cfg.min_pairs))
        self.step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
        self.postprocess = {}
        for t in self.sources:
            if t == DEFORM:
                self.postprocess[t] = DeformPostprocess(
                    self.level, flip_prob=cfg.flip_prob, p_pdice=cfg.p_pdice,
                    seed=cfg.seed + POSTPROCESS_SEED_OFFSET[t], n_classes=cfg.n_classes,
                    residual_affine=cfg.deform_residual_affine, min_pairs=cfg.min_pairs,
                    intensity_aug=cfg.intensity_aug_tf)
            else:
                self.postprocess[t] = LinearPostprocess(t, self.level, flip_prob=cfg.flip_prob,
                                                        p_pair=cfg.p_pair,
                                                        seed=cfg.seed + POSTPROCESS_SEED_OFFSET[t],
                                                        target_iters=cfg.target_iters,
                                                        min_pairs=cfg.min_pairs,
                                                        intensity_aug=cfg.intensity_aug_tf)
        self._identity_cache = {}
        self._frozen_cache = {}         # cascade: frozen-stage validation baselines
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
        """Extra trackables for `RunCheckpoint`: the post-processing RNGs (plus the TF intensity
        augmentation generators ``postprocess_<task>_aug``, only when TF augmentation is on)."""
        out = {"postprocess_%s" % t: pp.generator for t, pp in self.postprocess.items()}
        for t, pp in self.postprocess.items():
            if pp.aug_generator is not None:
                out["postprocess_%s_aug" % t] = pp.aug_generator
        return out

    def rebuild_steps(self):
        """Re-create every train / validation tf.function (re-trace on next use)."""
        self._steps = {t: self._make_deform_step() if t == DEFORM else self._make_train_step(t)
                       for t in self.sources}
        self._val_fns = {}

    def _task_weight(self, task):
        """Objective multiplier of `task` (``cfg.task_loss_weights``, default 1.0)."""
        return float(dict(self.cfg.task_loss_weights or {}).get(task, 1.0))

    def _normalise_gradients(self, gs, gnorm, w):
        """Per-task gradient normalisation (traced only with task_grad_norm > 0): normalise the
        raw (unweighted, AMP-unscaled) gradients `gs` of global norm `gnorm` to task_grad_norm
        (scale 1 when gated or non-finite), THEN multiply by the task weight `w`. Returns
        (applied gradients, factor = scale * w, {TASK_GRAD_NORM_KEYS: tensors})."""
        scale, gated = task_grad_norm_scale(gnorm, self.task_grad_norm, self.task_grad_norm_min)
        factor = scale if w == 1.0 else scale * tf.constant(w, scale.dtype)
        applied = scale_gradients(gs, factor)
        return applied, factor, {"grad_norm_scale": scale, "grad_norm_gated_frac": gated,
                                 "grad_norm_normed": tf.linalg.global_norm(applied)}

    def _enc_grad_diagnostics(self, gs, vs, gnorm, mult, post=None):
        """Encoder-only gradient norms of one step (traced; no effect on the update).

        diag_enc_grad_norm_raw: norm of the gradient of the UNSCALED task loss w.r.t. the shared
          encoder variables (the objective's gradient `gs` divided by ``mult`` = task weight /
          loss_scale_mm2 for linear, task weight for deform; with --task-grad-norm the weight is
          not in the objective, so ``mult`` = 1 / loss_scale_mm2 resp. 1);
        diag_enc_grad_norm_eff: the encoder norm actually applied: the objective's encoder
          gradient times ``post`` (with --task-grad-norm: normalisation scale x task weight; None
          = 1) times the global-clip factor min(1, clipnorm / (grad_norm x post)) (1 without clip).
        """
        enc_refs = {v.ref() for v in self.model.encoder.trainable_variables}
        enc = [g for g, v in zip(gs, vs) if v.ref() in enc_refs]
        enc_norm = tf.linalg.global_norm(enc) if enc else tf.constant(0.0)
        clip = getattr(O.inner_optimizer(self.optimizer), "global_clipnorm", None)
        eff = enc_norm
        if post is None:
            if clip:
                eff = enc_norm * tf.minimum(1.0, float(clip) / tf.maximum(gnorm, 1e-12))
        else:
            eff = enc_norm * post
            if clip:
                eff = eff * tf.minimum(1.0, float(clip) / tf.maximum(gnorm * post, 1e-12))
        return {"diag_enc_grad_norm_raw": enc_norm / float(mult), "diag_enc_grad_norm_eff": eff}

    def _group_size(self, task):
        """Group size of `task`'s batches (the deform group size may differ)."""
        if task == DEFORM and self.cfg.deform_group_size is not None:
            return int(self.cfg.deform_group_size)
        return int(self.cfg.group_size)

    def _make_deform_step(self):
        model, opt, cfg = self.model, self.optimizer, self.cfg
        task = DEFORM
        residual = bool(cfg.deform_residual_affine)
        shapes, dtypes = dali_loader.deform_output_signature(self.level, self._group_size(task), residual)
        signature = [tf.TensorSpec(s, tf.as_dtype(d)) for s, d in zip(shapes, dtypes)]
        pp = self.postprocess[task]
        amp = bool(cfg.amp)
        lam_p, lam_s, order = float(cfg.lambda_pdice), float(cfg.lambda_smooth), int(cfg.smooth_order)
        ncc_kw = dict(lambda_dice_tmpl=float(cfg.lambda_dice_tmpl), lambda_ncc_tmpl=float(cfg.lambda_ncc_tmpl),
                      lambda_ncc_pair=float(cfg.lambda_ncc_pair), ncc_window=self.ncc_window)
        lean = bool(cfg.lean_sampler)            # trace-time switches (design 3.3 / 3.4)
        if lean:
            ncc_kw["lean_sampler"] = True
        if cfg.lean_losses:
            ncc_kw["lean_losses"] = True
        if cfg.pair_recompute:                  # trace-time switch: off = today's ops
            ncc_kw["pair_recompute"] = True
            ncc_kw["n_pairs"] = num_pairs_static(self._group_size(task), cfg.p_pdice, cfg.min_pairs)
        cascade_kw = {"lean": True} if lean else {}
        w = self._task_weight(task)
        norm_on = self.task_grad_norm > 0      # trace-time switch: off adds no op
        pairs_on = cfg.min_pairs == "n"         # trace-time switch: '1' adds no op / result key
        cascade = self.cascade                 # trace-time switch (design section 5): None adds no op
        skip_on = cascade is not None and float(cfg.cascade_skip_prob) > 0
        chain = self._n_stages == 2            # trace-time switch: frozen 4 -> 2 chain (level 1)

        def body(images, labels, t_dali=None, skip=None):
            self.trace_counts[task] += 1          # Python side effect: runs only while tracing
            batch = pp(images, labels, t_dali)    # outside the tape
            x = batch["images"]
            in_finite = tf.reduce_all(tf.math.is_finite(x))
            batch["images"] = tf.where(in_finite, x, tf.zeros_like(x))
            x_in = batch["images"]
            if cascade is not None and not chain:  # frozen stage + pre-warp, outside the tape
                prev = cascade.deform(batch["images"])
                U, U_inv = prev["U"], prev["U_inv"]
                if skip_on:
                    U, U_inv = CS.skip_fields(skip, [U, U_inv])
            elif chain:                           # frozen 4 -> 2 chain; skip drops level 2
                if skip_on:
                    st = cascade.deform_stages(batch["images"])
                    U = CS.skip_to(skip, st["U4"], st["U"])
                    U_inv = CS.skip_to(skip, st["U4_inv"], st["U_inv"])
                else:
                    prev = cascade.deform(batch["images"])
                    U, U_inv = prev["U"], prev["U_inv"]
            if cascade is not None:
                x_in = CS.prewarp_deform(batch["images"], U)
            stats = list(model.non_trainable_variables)
            backup = [tf.identity(v) for v in stats]
            tvars = model.trainable_variables
            with tf.GradientTape() as tape:
                out = model(x_in, task=task, training=True)
                if cascade is not None:           # composed field; losses on the ORIGINAL batch
                    u, u_inv = diffeo.compose_cascade(U, U_inv, out["u"], out["u_inv"], **cascade_kw)
                    out = dict(out, u=u, u_inv=u_inv, u_res=out["u"])
                terms = losses.deform_loss(out, batch, lam_p, lam_s, order, **ncc_kw)
                objective = terms["loss"]
                if w != 1.0 and not norm_on:
                    objective = objective * tf.constant(w, objective.dtype)
                scaled = opt.get_scaled_loss(objective) if amp else objective
            grads = tape.gradient(scaled, tvars)
            diag = deform_diagnostics(out)
            if cascade is not None:
                diag["diag_u_res_max_vox"] = tf.reduce_max(tf.norm(tf.cast(out["u_res"], tf.float32), axis=-1))
            pairs = [(g, v) for g, v in zip(grads, tvars) if g is not None]   # static filter
            gs = [g for g, _ in pairs]
            vs = [v for _, v in pairs]
            if amp:
                gs = opt.get_unscaled_gradients(gs)
            gnorm = tf.linalg.global_norm(gs)
            finite = tf.logical_and(tf.math.is_finite(terms["loss"]), tf.math.is_finite(gnorm))
            finite = tf.logical_and(finite, in_finite)
            norm_res = {}
            if norm_on:
                gs_applied, factor, norm_res = self._normalise_gradients(gs, gnorm, w)
                enc_diag = self._enc_grad_diagnostics(gs, vs, gnorm, 1.0, post=factor)
            else:
                gs_applied = gs
                enc_diag = self._enc_grad_diagnostics(gs, vs, gnorm, w)

            def apply():
                opt.apply_gradients(zip(gs_applied, vs))
                return tf.constant(True)

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
            res.update(enc_diag)
            res.update(norm_res)
            if pairs_on:
                res[PAIR_COUNT_KEY] = tf.size(batch["pair_i"])
            res.update(grad_norm=gnorm, finite=finite, n_vars_updated=tf.constant(len(pairs)))
            return res

        if cascade is not None:                   # residual affine excluded (Trainer.__init__)
            return tf.function(lambda images, labels, skip: body(images, labels, None, skip),
                               input_signature=signature + [tf.TensorSpec([], tf.bool)])
        if residual:
            return tf.function(lambda images, labels, t_dali: body(images, labels, t_dali),
                               input_signature=signature)
        return tf.function(lambda images, labels: body(images, labels), input_signature=signature)

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
        w = self._task_weight(task)
        norm_on = self.task_grad_norm > 0      # trace-time switch: off adds no op
        pairs_on = cfg.min_pairs == "n"         # trace-time switch: '1' adds no op / result key
        S = self.S
        cascade = self.cascade                 # trace-time switch (design section 4): None adds no op
        skip_on = cascade is not None and float(cfg.cascade_skip_prob) > 0
        chain = self._n_stages == 2            # trace-time switch: frozen 4 -> 2 chain (level 1)
        spacing = P.LEVEL_SPACING[self.level]

        def body(images, t_dali, rt_dali=None, skip=None):
            self.trace_counts[task] += 1          # Python side effect: runs only while tracing
            batch = pp(images, t_dali, rt_dali)   # outside the tape
            # Non-finite inputs would make the barycenter's matrix inverse raise instead of
            # returning NaN, so they are zeroed and the step is skipped like any non-finite step.
            x = batch["images"]
            in_finite = tf.reduce_all(tf.math.is_finite(x))
            batch["images"] = tf.where(in_finite, x, tf.zeros_like(x))
            x_in = batch["images"]
            if cascade is not None and not chain:  # frozen stage + pre-warp, outside the tape
                A_prev = cascade.linear(batch["images"], task)
                if skip_on:
                    A_prev = CS.skip_linear(skip, A_prev)
            elif chain:                           # frozen 4 -> 2 chain; skip drops level 2
                if skip_on:
                    st = cascade.linear_stages(batch["images"], task)
                    A_prev = CS.skip_to(skip, st["A4"], st["A"])
                else:
                    A_prev = cascade.linear(batch["images"], task)
            if cascade is not None:
                x_in = CS.prewarp_linear(batch["images"], A_prev, spacing)
            stats = list(model.non_trainable_variables)
            backup = [tf.identity(v) for v in stats]
            tvars = model.trainable_variables
            with tf.GradientTape() as tape:
                out = model(x_in, task=task, training=True)
                A = out["A"] if cascade is None else CS.compose_linear(out["A_raw"], A_prev, model.centering)
                terms = losses.linear_loss(A, batch, S, lam)
                objective = terms["loss"]
                if loss_scale != 1.0:
                    objective = objective / tf.constant(loss_scale, objective.dtype)
                if w != 1.0 and not norm_on:
                    objective = objective * tf.constant(w, objective.dtype)
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
            norm_res = {}
            if norm_on:
                gs_applied, factor, norm_res = self._normalise_gradients(gs, gnorm, w)
                enc_diag = self._enc_grad_diagnostics(gs, vs, gnorm, 1.0 / loss_scale, post=factor)
            else:
                gs_applied = gs
                enc_diag = self._enc_grad_diagnostics(gs, vs, gnorm, w / loss_scale)

            def apply():
                opt.apply_gradients(zip(gs_applied, vs))
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
            res.update(enc_diag)
            res.update(norm_res)
            if pairs_on:
                res[PAIR_COUNT_KEY] = tf.size(batch["pair_i"])
            res.update(grad_norm=gnorm, finite=finite, n_vars_updated=tf.constant(len(pairs)))
            return res

        if cascade is not None:                   # one extra trailing input: skip bool[] (design section 6)
            skip_spec = [tf.TensorSpec([], tf.bool)]
            if task == "rigid":
                return tf.function(lambda images, t_dali, rt_dali, skip: body(images, t_dali, rt_dali, skip),
                                   input_signature=signature + skip_spec)
            return tf.function(lambda images, t_dali, skip: body(images, t_dali, None, skip),
                               input_signature=signature + skip_spec)
        if task == "rigid":
            return tf.function(lambda images, t_dali, rt_dali: body(images, t_dali, rt_dali),
                               input_signature=signature)
        return tf.function(lambda images, t_dali: body(images, t_dali), input_signature=signature)

    def split_batch(self, task, batch_tuple):
        """(step inputs, control extras): the first 3 (rigid) / 2 (affine) / 2 or 3 (deform, with
        the residual affine) DALI outputs, and the trailing (G_dali, control_version) of a
        control-file pipeline (empty tuple otherwise)."""
        k = dali_loader.output_count(task, self.cfg.deform_residual_affine)
        batch_tuple = tuple(batch_tuple)
        return batch_tuple[:k], batch_tuple[k:]

    def control_metrics(self, task, extras, values, step=None):
        """Per-step control logs: g_mixed_batch (1 if the batch's G_dali differ across samples),
        the current control values, the G warm-up maximum angle control_g_rot_max_deg (with a
        sampler and `step`) and the version lag (writer version - batch version)."""
        out = {"control_rot_deg": float(values["rotation_deg"]),
               "control_trans_mm": float(values["translation_mm"]),
               "control_g_angle_deg": float(values["g_angle_deg"])}
        sampler = getattr(self.controls.get(task), "sampler", None)
        if sampler is not None and step is not None:
            out["control_g_rot_max_deg"] = float(sampler.rot_max_at(step))
        if len(extras) >= 2:
            g = np.asarray(extras[0], np.float64)
            ver = np.asarray(extras[1], np.float64).reshape(-1)
            out["g_mixed_batch"] = float(np.abs(g - g[:1]).max() > 1e-6 or np.ptp(ver) > 0)
            writer = self.controls[task].writer
            out["control_version_lag"] = float(writer.version - ver.min()) if writer is not None else 0.0
        return out

    def train_step(self, task, batch_tuple, skip=False):
        """Run one step of `task` on a DALI-format tuple (control extras are ignored). Returns
        Python scalars. ``skip`` (cascade only, design section 6): skip the frozen stage in this step
        (passed as the step's trailing bool input; ignored without a cascade)."""
        batch_tuple, _ = self.split_batch(task, batch_tuple)
        if self.cascade is None:
            out = self._steps[task](*batch_tuple)
        else:
            out = self._steps[task](*batch_tuple, tf.constant(bool(skip)))
        res = {k: float(v.numpy()) for k, v in out.items() if k not in ("finite", "n_vars_updated")}
        res["finite"] = bool(out["finite"].numpy())
        res["n_vars_updated"] = int(out["n_vars_updated"].numpy())
        return res

    # ------------------------------------------------------------------ validation
    def _val_fn(self, task, nv, train_mode=False):
        """Validation forward -> A. ``train_mode=False``: eval mode (moving BN statistics), the
        source of truth. ``train_mode=True``: diagnostic forward with training-mode BN (batch
        statistics of the group) and dropout off; the BN moving statistics are restored
        afterwards, so it changes no state. With a cascade: `_cascade_val_fn` (same key, returns a
        dict)."""
        if self.cascade is not None:
            return self._cascade_val_fn(task, nv, train_mode)
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

    def _train_mode_forward(self, x, task, extract):
        """Traced train-mode diagnostic forward of the trainable model (as `_val_fn`): dropout
        rates zeroed while tracing, BN batch statistics, moving statistics restored afterwards.
        Returns ``extract(model output)`` (a dict of tensors) after the restores."""
        model = self.model
        stats = list(model.non_trainable_variables)
        backup = [tf.identity(v) for v in stats]
        drops = [m for m in model.submodules if isinstance(m, tf.keras.layers.Dropout)]
        rates = [d.rate for d in drops]
        try:
            for d in drops:
                d.rate = 0.0
            res = extract(model(x, task=task, training=True))
        finally:
            for d, r in zip(drops, rates):
                d.rate = r
        with tf.control_dependencies(list(res.values())):
            restores = [v.assign(b) for v, b in zip(stats, backup)]
        with tf.control_dependencies(restores):
            return {k: tf.identity(v) for k, v in res.items()}

    def _cascade_val_fn(self, task, nv, train_mode=False):
        """Cascade linear validation forward (design section 8), key (task, nv, train_mode), traced once:
        pool -> frozen (eval) -> pre-warp -> trainable (eval, or train-mode BN restored as
        `_val_fn`) -> compose. Returns {'A': composed, 'A_prev': frozen A4} in eval mode and
        {'A': composed} in train mode. Never skips. Frozen chain (level 1, eval mode): 'A_prev' =
        the composed frozen 4 + 2 estimate and 'A_prev4' = the level-4 stage alone."""
        key = (task, int(nv), bool(train_mode))
        if key not in self._val_fns:
            shape = (int(nv),) + tuple(P.LEVEL_SHAPE[self.level]) + (1,)
            model, cascade = self.model, self.cascade
            spacing = P.LEVEL_SPACING[self.level]
            chain = self._n_stages == 2
            self.val_trace_counts.setdefault(key, 0)

            def fwd(x):
                self.val_trace_counts[key] += 1
                if chain and not train_mode:
                    st = cascade.linear_stages(x, task)
                    A_prev = st["A"]
                else:
                    A_prev = cascade.linear(x, task)
                x_in = CS.prewarp_linear(x, A_prev, spacing)
                if train_mode:
                    return self._train_mode_forward(
                        x_in, task, lambda out: {"A": CS.compose_linear(out["A_raw"], A_prev, model.centering)})
                out = model(x_in, task=task, training=False)
                res = {"A": CS.compose_linear(out["A_raw"], A_prev, model.centering), "A_prev": A_prev}
                if chain:
                    res["A_prev4"] = st["A4"]
                return res

            self._val_fns[key] = tf.function(fwd, input_signature=[tf.TensorSpec(shape, tf.float32)])
        return self._val_fns[key]

    def _val_errors(self, task, vs, groups, train_mode, frozen_outs=None):
        """(pooled errors, [A per group] float64). With a cascade the validation function returns
        a dict (composed ``A``); ``frozen_outs`` (a list, eval-mode cascade passes only) receives
        the frozen A4 [N, 4, 4] float64 of every group (frozen chain: dict A = levels 4 + 2,
        A4 = level 4 alone)."""
        fn = self._val_fn(task, vs.group_size, train_mode)
        errs, mats = [], []
        for g, grp in enumerate(groups):
            out = fn(tf.constant(grp["images"]))
            A = (out["A"] if self.cascade is not None else out).numpy().astype(np.float64)
            if frozen_outs is not None and self._n_stages == 2:
                frozen_outs.append({"A": out["A_prev"].numpy().astype(np.float64),
                                    "A4": out["A_prev4"].numpy().astype(np.float64)})
            elif frozen_outs is not None:
                frozen_outs.append(out["A_prev"].numpy().astype(np.float64))
            mats.append(A)
            errs.append(M.group_errors(A, grp["A_star"], grp["T_target"], self.S_np, seed=g,
                                       affine=task == "affine" and not train_mode))
        return M.concat_errors(errs), mats

    def _val_deform_fn(self, nv, train_mode=False):
        """Deform validation forward -> dict(u, u_inv, v, v_raw) float32. Eval mode, or (with
        ``train_mode``) training-mode BN with dropout off and the BN statistics restored, as
        `_val_fn`. With a cascade: `_cascade_val_deform_fn` (same key)."""
        if self.cascade is not None:
            return self._cascade_val_deform_fn(nv, train_mode)
        key = (DEFORM, int(nv), bool(train_mode))
        if key not in self._val_fns:
            shape = (int(nv),) + tuple(P.LEVEL_SHAPE[self.level]) + (1,)
            model = self.model
            self.val_trace_counts.setdefault(key, 0)
            names = ("u", "u_inv", "v", "v_raw")

            def fwd(x):
                self.val_trace_counts[key] += 1
                out = model(x, task=DEFORM, training=False)
                return {k: tf.cast(out[k], tf.float32) for k in names}

            def fwd_train_mode(x):
                self.val_trace_counts[key] += 1
                stats = list(model.non_trainable_variables)
                backup = [tf.identity(v) for v in stats]
                drops = [m for m in model.submodules if isinstance(m, tf.keras.layers.Dropout)]
                rates = [d.rate for d in drops]
                try:
                    for d in drops:
                        d.rate = 0.0
                    out = model(x, task=DEFORM, training=True)
                finally:
                    for d, r in zip(drops, rates):
                        d.rate = r
                res = {k: tf.cast(out[k], tf.float32) for k in names}
                with tf.control_dependencies(list(res.values())):
                    restores = [v.assign(b) for v, b in zip(stats, backup)]
                with tf.control_dependencies(restores):
                    return {k: tf.identity(v) for k, v in res.items()}

            self._val_fns[key] = tf.function(fwd_train_mode if train_mode else fwd,
                                             input_signature=[tf.TensorSpec(shape, tf.float32)])
        return self._val_fns[key]

    def _cascade_val_deform_fn(self, nv, train_mode=False):
        """Cascade deform validation forward (design section 8), key ('deform', nv, train_mode): pool ->
        frozen (eval) -> pre-warp warp(x, U) -> trainable residual -> compose_cascade. Returns the
        composed u, u_inv and the residual v, v_raw (float32); eval mode also U, U_inv, V (the
        upsampled frozen fields, for the frozen baselines). Never skips. Frozen chain (level 1,
        eval mode): U, U_inv, V = the frozen 4 + 2 chain, plus U4, U4_inv, V4 (level 4 alone)."""
        key = (DEFORM, int(nv), bool(train_mode))
        if key not in self._val_fns:
            shape = (int(nv),) + tuple(P.LEVEL_SHAPE[self.level]) + (1,)
            model, cascade = self.model, self.cascade
            chain = self._n_stages == 2
            frozen_keys = CHAIN_DEFORM_FROZEN_KEYS if chain else ("U", "U_inv", "V")
            self.val_trace_counts.setdefault(key, 0)

            def composed(prev, out):
                u, u_inv = diffeo.compose_cascade(prev["U"], prev["U_inv"], out["u"], out["u_inv"])
                return {"u": tf.cast(u, tf.float32), "u_inv": tf.cast(u_inv, tf.float32),
                        "v": tf.cast(out["v"], tf.float32), "v_raw": tf.cast(out["v_raw"], tf.float32)}

            def fwd(x):
                self.val_trace_counts[key] += 1
                prev = cascade.deform_stages(x) if chain and not train_mode else cascade.deform(x)
                x_in = CS.prewarp_deform(x, prev["U"])
                if train_mode:
                    return self._train_mode_forward(x_in, DEFORM, lambda out: composed(prev, out))
                res = composed(prev, model(x_in, task=DEFORM, training=False))
                res.update({k: tf.cast(prev[k], tf.float32) for k in frozen_keys})
                return res

            self._val_fns[key] = tf.function(fwd, input_signature=[tf.TensorSpec(shape, tf.float32)])
        return self._val_fns[key]

    def _val_pairs(self, n, seed):
        """Ordered validation pairs (i, j): all if n (n - 1) <= val_deform_max_pairs, else a fixed
        seeded subset of that size (sorted)."""
        i, j = M.all_ordered_pairs(int(n))
        cap = int(self.cfg.val_deform_max_pairs)
        if len(i) > cap:
            sel = np.sort(np.random.default_rng(int(seed)).choice(len(i), cap, replace=False))
            i, j = i[sel], j[sel]
        return np.asarray(i, np.int32), np.asarray(j, np.int32)

    def _deform_errors(self, labels, u, u_inv, v, v_raw, seed, chunk=None, images=None):
        """`M.deform_group_errors` of one group (eager TF warps, numpy metrics). labels uint8
        [N, X, Y, Z(, 1)]; fields float32 [N, X, Y, Z, 3]. Also returns the warped one-hot hard
        segmentation [N, X, Y, Z] and the label template [X, Y, Z] (for the PNG).

        ``images`` float32 [N, X, Y, Z, 1] (optional): also ncc_template [N] and ncc_pair [P]
        (squared LNCC with the training definition: leave-one-out template, label ROI, window
        ``self.ncc_window``). ``chunk``: pairs warped at once (None = ``cfg.val_pair_chunk``,
        default 16; the validation callers never pass it)."""
        chunk = int(self.cfg.val_pair_chunk if chunk is None else chunk)
        n_cls = int(self.cfg.n_classes)
        lab = np.asarray(labels).astype(np.int32)
        if lab.ndim == 5:
            lab = lab[..., 0]
        if int(self.cfg.val_image_chunk) > 0:     # Python switch: 0 = the unchunked path below
            return self._deform_errors_chunked(lab, u, u_inv, v, v_raw, seed, chunk, images)
        onehot = tf.one_hot(tf.constant(lab), n_cls, dtype=tf.float32)
        u, u_inv = tf.convert_to_tensor(u, tf.float32), tf.convert_to_tensor(u_inv, tf.float32)
        warped = diffeo.warp(onehot, u)
        seg_warped = tf.argmax(warped, axis=-1).numpy()
        seg_template = tf.argmax(tf.reduce_mean(warped, axis=0), axis=-1).numpy()
        ncc = {}
        if images is not None:
            images = tf.convert_to_tensor(np.asarray(images, np.float32))
            fg_tmpl = losses.foreground(tf.reduce_mean(warped, axis=0, keepdims=True))
            ncc["ncc_template"] = losses.template_ncc_loss(diffeo.warp(images, u), fg_tmpl,
                                                           self.ncc_window)[1].numpy()
        del warped
        i, j = self._val_pairs(lab.shape[0], seed)
        pred, ncc_pair = [], []
        for s in range(0, len(i), int(chunk)):
            ii, jj = i[s:s + chunk], j[s:s + chunk]
            disp = diffeo.pairwise_displacement(u, u_inv, ii, jj)
            moved = diffeo.warp(tf.gather(onehot, ii), disp)
            pred.append(tf.argmax(moved, axis=-1).numpy())
            if images is not None:
                ncc_pair.append(losses.pair_ncc_loss(images, disp, ii, jj, losses.foreground(moved),
                                                     losses.foreground(tf.gather(onehot, jj)),
                                                     self.ncc_window)[1].numpy())
        seg_pair_pred = np.concatenate(pred, axis=0)
        jac = diffeo.jacobian_determinant(u).numpy()
        inv = diffeo.compose(u, u_inv).numpy()
        errs = M.deform_group_errors(seg_warped, seg_template, seg_pair_pred, lab[j], jac, inv,
                                     np.asarray(v), np.asarray(v_raw), P.LEVEL_SPACING[self.level], n_cls)
        if images is not None:
            ncc["ncc_pair"] = np.concatenate(ncc_pair, axis=0)
            errs.update(ncc)
        return errs, {"seg_warped": seg_warped, "seg_template": seg_template, "jac": jac}

    def _deform_errors_chunked(self, lab, u, u_inv, v, v_raw, seed, chunk, images=None):
        """`_deform_errors` with --val-image-chunk k > 0 (level-1 validation memory): every GPU
        tensor holds at most k images (warped one-hot, warped image, LNCC moments, Jacobian,
        inverse-consistency field) or ``chunk`` pairs; the one-hot is built per chunk from the
        labels, the label template from a running sum, the leave-one-out intensity template from
        the host-side warped images. Warps use the lean sampler iff ``cfg.lean_sampler`` (same
        forward values). Per-chunk results go to the host; the metrics are the same numpy
        `M.deform_group_errors` as the unchunked path (equal up to float summation order)."""
        k = int(self.cfg.val_image_chunk)
        n_cls = int(self.cfg.n_classes)
        lean = bool(self.cfg.lean_sampler)
        n = lab.shape[0]
        u, u_inv = tf.convert_to_tensor(u, tf.float32), tf.convert_to_tensor(u_inv, tf.float32)
        spans = [(s, min(s + k, n)) for s in range(0, n, k)]

        def onehot_of(idx):
            return tf.one_hot(tf.constant(lab[np.asarray(idx)]), n_cls, dtype=tf.float32)

        seg_warped, tmpl_sum, warped_img, jac, inv = [], None, [], [], []
        for a, b in spans:
            w = diffeo.warp(onehot_of(np.arange(a, b)), u[a:b], lean=lean)
            seg_warped.append(tf.argmax(w, axis=-1).numpy())
            s = tf.reduce_sum(w, axis=0)
            tmpl_sum = s if tmpl_sum is None else tmpl_sum + s
            del w
            if images is not None:
                warped_img.append(diffeo.warp(tf.constant(np.asarray(images[a:b], np.float32)), u[a:b],
                                              lean=lean).numpy())
            jac.append(diffeo.jacobian_determinant(u[a:b]).numpy())
            inv.append(diffeo.compose(u[a:b], u_inv[a:b]).numpy())
        tmpl = tmpl_sum / float(n)
        del tmpl_sum
        seg_warped = np.concatenate(seg_warped, axis=0)
        seg_template = tf.argmax(tmpl, axis=-1).numpy()
        ncc = {}
        if images is not None:
            fg_tmpl = losses.foreground(tmpl[None])
            roi = losses.ncc_roi(fg_tmpl, self.ncc_window)
            total = np.sum(np.concatenate(warped_img, axis=0), axis=0, keepdims=True, dtype=np.float32)
            nt = []
            for (a, b), wi in zip(spans, warped_img):
                w = tf.constant(wi)
                t = (tf.constant(total) - w) / max(n - 1.0, 1.0)
                nt.append(losses.masked_mean(losses.local_ncc(w, t, self.ncc_window), roi).numpy())
            ncc["ncc_template"] = np.concatenate(nt, axis=0)
            imgs = np.asarray(images, np.float32)
        del tmpl
        i, j = self._val_pairs(n, seed)
        pred, ncc_pair = [], []
        for s in range(0, len(i), int(chunk)):
            ii, jj = i[s:s + chunk], j[s:s + chunk]
            disp = diffeo.pairwise_displacement(u, u_inv, ii, jj, lean=lean)
            moved = diffeo.warp(onehot_of(ii), disp, lean=lean)
            pred.append(tf.argmax(moved, axis=-1).numpy())
            if images is not None:
                mi = diffeo.warp(tf.constant(imgs[ii]), disp, lean=lean)
                fg = tf.maximum(losses.foreground(moved), losses.foreground(onehot_of(jj)))
                roi_p = losses.ncc_roi(fg, self.ncc_window)
                ncc_pair.append(losses.masked_mean(losses.local_ncc(mi, tf.constant(imgs[jj]), self.ncc_window),
                                                   roi_p).numpy())
            del disp, moved
        seg_pair_pred = np.concatenate(pred, axis=0)
        jac = np.concatenate(jac, axis=0)
        inv = np.concatenate(inv, axis=0)
        errs = M.deform_group_errors(seg_warped, seg_template, seg_pair_pred, lab[j], jac, inv,
                                     np.asarray(v), np.asarray(v_raw), P.LEVEL_SPACING[self.level], n_cls)
        if images is not None:
            ncc["ncc_pair"] = np.concatenate(ncc_pair, axis=0)
            errs.update(ncc)
        return errs, {"seg_warped": seg_warped, "seg_template": seg_template, "jac": jac}

    def _val_deform_errors(self, vs, groups, train_mode, frozen_outs=None):
        """(pooled errors, [dict(u, seg_warped, seg_template, jac) per group]). With a cascade in
        eval mode the errors also hold composed_mean_disp_mm (one value per group: RMS over voxels
        of |mean_i u_i| in mm of the composed u) and ``frozen_outs`` (a list, when given) receives
        the frozen dict(U, U_inv, V) (numpy float32) of every group (frozen chain: also U4,
        U4_inv, V4 = level 4 alone)."""
        fn = self._val_deform_fn(vs.group_size, train_mode)
        errs, extras = [], []
        for g, grp in enumerate(groups):
            out = fn(tf.constant(grp["images"]))
            e, ex = self._deform_errors(grp["labels"], out["u"], out["u_inv"], out["v"].numpy(),
                                        out["v_raw"].numpy(), seed=g, images=grp["images"])
            if self.cascade is not None and not train_mode:
                mean_u = diffeo.to_mm(tf.reduce_mean(out["u"], axis=0), P.LEVEL_SPACING[self.level])
                e["composed_mean_disp_mm"] = np.asarray(
                    [float(tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(mean_u), axis=-1))))])
                if frozen_outs is not None:
                    keys = CHAIN_DEFORM_FROZEN_KEYS if self._n_stages == 2 else ("U", "U_inv", "V")
                    frozen_outs.append({k: out[k].numpy() for k in keys})
            ex["u"] = out["u"].numpy()
            errs.append(e)
            extras.append(ex)
        return M.concat_errors(errs), extras

    def _identity_deform_metrics(self, name, vs, groups):
        """Identity baseline (u = 0) of a deform validation set: pair / template hard Dice and
        LNCC, computed once per (set name, number of groups) and cached."""
        key = (name, len(groups))
        if key not in self._identity_cache:
            errs = []
            for g, grp in enumerate(groups):
                z = np.zeros(tuple(grp["images"].shape[:4]) + (3,), np.float32)
                errs.append(self._deform_errors(grp["labels"], z, z, z, z, seed=g, images=grp["images"])[0])
            pooled = M.concat_errors(errs)
            self._identity_cache[key] = {
                "val_%s/pair_dice_identity" % name: float(np.mean(pooled["pair_dice"])),
                "val_%s/template_dice_identity" % name: float(np.mean(pooled["template_dice"])),
                "val_%s/ncc_template_identity" % name: float(np.mean(pooled["ncc_template"])),
                "val_%s/ncc_pair_identity" % name: float(np.mean(pooled["ncc_pair"]))}
        return dict(self._identity_cache[key])

    def _frozen_baseline_metrics(self, name, vs, groups, frozen_outs):
        """Frozen-stage baselines of a validation set (cascade, design section 8), computed once per
        (set name, number of groups) from the eval-mode pass's frozen estimates ``frozen_outs``
        (ignored on a cache hit) and cached (the frozen weights never change). Linear sets:
        ``M.summarize(M.group_errors(A4, ...), 'val_<name>_frozen')`` (same groups, N, S and
        seeds as the composed metrics; affine sets incl. the affine factor errors). Deform sets:
        ``_deform_errors(labels, U, U_inv, V, V, ...)`` -> ``val_<name>_frozen/{template_dice,
        pair_dice, neg_jac_pct, inv_consistency_mm, ..., ncc_template, ncc_pair}``. Frozen chain
        (level 1): ``val_<name>_frozen/...`` = levels 4 + 2 composed (A / U, U_inv, V; V = the
        level-2 residual SVF, so its centrality keys describe that residual) and
        ``val_<name>_frozen4/...`` = level 4 alone (A4 / U4, U4_inv, V4), same groups and seeds."""
        key = (name, len(groups))
        if key not in self._frozen_cache:
            if frozen_outs is None or len(frozen_outs) != len(groups):
                raise ValueError("frozen baseline of %r needs the frozen outputs of all %d groups"
                                 % (name, len(groups)))
            if self._n_stages == 2:
                # Frozen chain: '_frozen' = levels 4 + 2 composed, '_frozen4' = level 4 alone.
                variants = (("", ("U", "U_inv", "V"), "A"), ("4", ("U4", "U4_inv", "V4"), "A4"))
            else:
                variants = (("", ("U", "U_inv", "V"), None),)
            res = {}
            for suffix, (ku, kui, kv), ka in variants:
                errs = []
                for g, (grp, fo) in enumerate(zip(groups, frozen_outs)):
                    if vs.task == DEFORM:
                        errs.append(self._deform_errors(grp["labels"], fo[ku], fo[kui], fo[kv], fo[kv],
                                                        seed=g, images=grp["images"])[0])
                    else:
                        errs.append(M.group_errors(fo if ka is None else fo[ka], grp["A_star"],
                                                   grp["T_target"], self.S_np, seed=g,
                                                   affine=vs.task == "affine"))
                res.update(M.summarize(M.concat_errors(errs), "val_%s_frozen%s" % (name, suffix)))
            self._frozen_cache[key] = res
        return dict(self._frozen_cache[key])

    def _frozen_outs_list(self, name, groups):
        """[] when this eval pass must collect the frozen estimates (cascade, cache miss), else None."""
        if self.cascade is None or (name, len(groups)) in self._frozen_cache:
            return None
        return []

    def _write_deform_png(self, name, grp, extra, metrics, step):
        """Deformable validation PNG (`runtime.val_png.write_deform_png`), file
        ``<name>_step<step>.png``. Never raises."""
        cfg = self.cfg
        try:
            from runtime import val_png as V
            idx = V.select_indices(grp["images"].shape[0], cfg.val_png_k, cfg.val_png_seed)
            images = np.asarray(grp["images"], np.float32)[idx]
            u = extra["u"][idx]
            warped = diffeo.warp(tf.constant(images), tf.constant(u)).numpy()
            jac = extra["jac"][idx]
            fold = 100.0 * np.mean((jac <= 0).reshape((jac.shape[0], -1)), axis=1)
            z = P.LEVEL_SHAPE[self.level][2] // 2
            pre = "val_%s" % name
            title = V.deform_figure_title(step, metrics[pre + "/template_dice"],
                                          metrics[pre + "/pair_dice"], metrics[pre + "/neg_jac_pct"], z)
            labels = ["val %d" % int(i) for i in np.asarray(grp["index"])[idx]]
            os.makedirs(cfg.val_png_dir, exist_ok=True)
            path = os.path.join(cfg.val_png_dir, "%s_step%07d.png" % (name, int(step)))
            V.write_deform_png(path, images, warped, extra["seg_warped"][idx], extra["seg_template"],
                               jac, title, fold, labels=labels, mean_col=bool(cfg.val_png_mean_col))
            self.val_png_paths.append(path)
        except Exception as e:  # noqa: BLE001  (a PNG must never stop training)
            print("WARNING step %d: validation PNG for %s failed: %r" % (int(step), name, e),
                  file=sys.stderr, flush=True)
            if self.logger is not None:
                self.logger.log_event(int(step), {"event": "val_png_failed", "task": name,
                                                  "error": repr(e)})

    def _validate_deform(self, name, vs, groups):
        """Metrics of one deform validation set (module docstring)."""
        out = {}
        pre = "val_%s" % name
        t0 = time.perf_counter()
        frozen_outs = self._frozen_outs_list(name, groups)
        errs, extras = self._val_deform_errors(vs, groups, False, frozen_outs)
        out.update(M.summarize(errs, pre))
        out[pre + "/pair_dice_loss"] = M.deform_pair_dice_loss(out, pre)
        out[pre + "/groups"] = len(groups)
        out[pre + "/time_s"] = time.perf_counter() - t0
        tm = M.summarize(self._val_deform_errors(vs, groups, True)[0], pre + "_trainmode")
        for k in ("template_dice", "pair_dice"):
            out["%s_trainmode/%s" % (pre, k)] = tm["%s_trainmode/%s" % (pre, k)]
        out.update(self._identity_deform_metrics(name, vs, groups))
        if self.cascade is not None:
            out.update(self._frozen_baseline_metrics(name, vs, groups, frozen_outs))
        g = int(self.cfg.val_png_group)
        if self.cfg.val_png_dir and 0 <= g < len(groups):
            self._write_deform_png(name, groups[g], extras[g], out, int(self.step.numpy()))
        return out

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
        (best checkpoint). Affine sets also report center_/pair_{scale_pct, shear_pct,
        qr_rot_deg} (`runtime.metrics.affine_matrix_errors`).
        ``val_<task>_trainmode/{center,pair}_rms_mm``: the same groups with training-mode BN (no
        statistics update, dropout off), to expose any train/eval gap.
        ``val/mean_pair_rms_mm``: mean of ``val_<task>/pair_rms_mm`` over the trained LINEAR
        tasks (sets named exactly like the task, i.e. without G), when every one was validated.
        Deform sets (task 'deform'): `_validate_deform` (module docstring).
        With a cascade every metric above is of the COMPOSED result, plus the cached frozen-stage
        baselines ``val_<name>_frozen/...`` (`_frozen_baseline_metrics`) and, for deform sets,
        ``val_<name>/composed_mean_disp_mm``.
        """
        out = {}
        for name, vs in self.val_sets.items():
            task = vs.task
            if task not in ("rigid", "affine", DEFORM):
                continue
            groups = vs.groups if max_groups is None else vs.groups[:int(max_groups)]
            if not groups:
                continue
            if task == DEFORM:
                out.update(self._validate_deform(name, vs, groups))
                continue
            t0 = time.perf_counter()
            frozen_outs = self._frozen_outs_list(name, groups)
            errs, mats = self._val_errors(task, vs, groups, False, frozen_outs)
            out.update(M.summarize(errs, "val_%s" % name))
            out["val_%s/groups" % name] = len(groups)
            out["val_%s/time_s" % name] = time.perf_counter() - t0
            pre = "val_%s_trainmode" % name
            tm = M.summarize(self._val_errors(task, vs, groups, True)[0], pre)
            for k in ("center_rms_mm", "pair_rms_mm"):
                out["%s/%s" % (pre, k)] = tm["%s/%s" % (pre, k)]
            if self.cascade is not None:
                out.update(self._frozen_baseline_metrics(name, vs, groups, frozen_outs))
            g = int(self.cfg.val_png_group)
            if self.cfg.val_png_dir and 0 <= g < len(groups):
                self._write_val_png(name, groups[g], mats[g], out, int(self.step.numpy()))
        mean_pair = M.mean_pair_rms(out, self.linear_tasks)
        if mean_pair is not None:
            out[MEAN_PAIR_METRIC] = mean_pair
        return out

    # ------------------------------------------------------------------ BN recalibration
    def _bn_parts(self, task):
        """Model parts a `task` forward runs through (their BN layers are recalibrated)."""
        if task == DEFORM:
            return [self.model.encoder, self.model.deform_decoder]
        return [self.model.encoder, self.model.linear_decoder]

    @staticmethod
    def _bn_layers(parts):
        """Unique `BatchNormWithSeparateAffine` layers inside `parts` (stable order;
        `models.layers.batch_norm_layers`)."""
        return L.batch_norm_layers(list(parts))

    def _recal_input(self, task, inputs):
        """Eager input path of one recalibration batch, exactly as the cascade training step:
        postprocess (flips on), then pool -> frozen (eval) -> pre-warp. Returns (x_in, finite):
        finite False (x_in None) when the post-processed images are not all finite."""
        batch = self.postprocess[task](*inputs)
        x = batch["images"]
        if not bool(tf.reduce_all(tf.math.is_finite(x)).numpy()):
            return None, False
        if task == DEFORM:
            return CS.prewarp_deform(x, self.cascade.deform(x)["U"]), True
        A_prev = self.cascade.linear(x, task)
        return CS.prewarp_linear(x, A_prev, P.LEVEL_SPACING[self.level]), True

    def _recal_validate(self, prefix, step):
        """`validate()` with PNGs off, logged at `step` with every key prefixed by `prefix`."""
        png_dir = self.cfg.val_png_dir
        self.cfg.val_png_dir = None
        try:
            val = self.validate()
        finally:
            self.cfg.val_png_dir = png_dir
        self._log(step, {prefix + k: v for k, v in val.items()})
        return val

    def _bn_gap(self, val, prefix):
        """{'<prefix>/<set>/<metric>': |eval - trainmode|} of the validation metrics `val`."""
        out = {}
        for name, vs in self.val_sets.items():
            metrics = BN_GAP_METRICS[DEFORM] if vs.task == DEFORM else BN_GAP_METRICS["linear"]
            for k in metrics:
                a, b = val.get("val_%s/%s" % (name, k)), val.get("val_%s_trainmode/%s" % (name, k))
                if a is not None and b is not None:
                    out["%s/%s/%s" % (prefix, name, k)] = abs(float(a) - float(b))
        return out

    def recalibrate_bn(self, num_batches, freeze, val=True):
        """Re-estimate the trainable stage's BN moving statistics on cascade training batches
        (design section 7; cascade trainers only, called once at the start of a fresh run).

        1. ``val``: pre validation (PNGs off), logged at the current step as ``bnrecal_pre/...``.
        2. Plan: `bn_recal_plan` (largest remainder over the task ratio, each task >= 1,
           interleaved round-robin, no RNG).
        3. Slot b = 0..K-1 (the GLOBAL plan index) of task t: ``self.controls[t].update(b)`` only
           if t has a control (G / curriculum resampled as during steps 0..K-1; `fit` restarts
           at update(0), the control values being pure functions of the step), then EXACTLY one
           ``next(sources[t])`` and one postprocess draw (flips on, the training `pp`). So the
           recalibration consumes K source batches and K postprocess draws: K > 0 changes the
           data order vs K = 0 (reproducible on resume via the checkpointed generators). A batch
           with non-finite images consumes its slot without a forward (no replacement draw;
           stderr warning, ``skipped_nonfinite[t]``); a task with 0 used batches -> RuntimeError;
           StopIteration -> RuntimeError('bn recalibration: source <t> exhausted'). Forward:
           pool -> frozen (eval) -> pre-warp, then ``model(x_in, t, training=True)`` EAGERLY and
           without a tape (dropout as in training; a tape-free forward runs a recompute_grad
           decoder once).
        4. Exact averages: before each forward every `BatchNormWithSeparateAffine` m of the
           parts t runs gets ``m.bn.momentum = c_m / (c_m + 1)`` (c_m = its earlier recal
           updates, over all tasks), so the moving statistics become the mean of that layer's
           per-batch statistics (the first update replaces the level-4 values). Every layer's
           own momentum is restored in ``finally``.
        5. ``val``: post validation (statistics still unfrozen) ``bnrecal_post/...``.
        6. ``freeze``: ``model.set_bn_stats_frozen(True)`` + `rebuild_steps()`.
        7. Event ``bn_recalibration`` (logged at the current step) = the returned dict: batches
           (used per task), skipped_nonfinite, planned, num_batches, n_layers_updated,
           layers_without_updates (layers of a used part whose statistics are bitwise unchanged;
           must be [] else RuntimeError; a repeated recalibration that reproduces the current
           statistics exactly would also be flagged), elapsed_s, freeze, gap (``gap_pre/<set>/<metric>`` and
           ``gap_post/...`` = |eval - trainmode| of center_rms_mm / pair_rms_mm resp.
           template_dice / pair_dice; {} when ``val`` is False).

        Only the trainable model's BN statistics change: no trainable variable, optimizer slot,
        ``self.step`` or frozen-stage variable is touched.
        """
        if self.cascade is None:
            raise RuntimeError("bn recalibration needs a cascade trainer")
        t_start = time.perf_counter()
        step = int(self.step.numpy())
        plan = bn_recal_plan(num_batches, self.schedule.ratio)
        tasks = [t for t in self.schedule.tasks]
        task_layers = {t: self._bn_layers(self._bn_parts(t)) for t in tasks}
        layers = self._bn_layers([p for t in tasks for p in self._bn_parts(t)])
        gap = {}
        if val:
            gap.update(self._bn_gap(self._recal_validate("bnrecal_pre/", step), "gap_pre"))
        before = {id(m): [v.numpy().copy() for v in m.bn.non_trainable_variables] for m in layers}
        orig = {id(m): m.bn.momentum for m in layers}
        counts = {id(m): 0 for m in layers}
        used = {t: 0 for t in tasks}
        skipped = {t: 0 for t in tasks}
        try:
            for b, t in enumerate(plan):
                if t in self.controls:
                    self.controls[t].update(b)
                try:
                    batch_tuple = next(self.sources[t])
                except StopIteration:
                    raise RuntimeError("bn recalibration: source %s exhausted" % t)
                inputs, _ = self.split_batch(t, batch_tuple)
                x_in, finite = self._recal_input(t, inputs)
                if not finite:
                    skipped[t] += 1
                    print("WARNING bn recalibration: non-finite input images in %s batch %d (slot "
                          "skipped)" % (t, b), file=sys.stderr, flush=True)
                    continue
                for m in task_layers[t]:
                    c = counts[id(m)]
                    m.bn.momentum = c / (c + 1.0)
                self.model(x_in, task=t, training=True)
                for m in task_layers[t]:
                    counts[id(m)] += 1
                used[t] += 1
        finally:
            for m in layers:
                m.bn.momentum = orig[id(m)]
        empty = [t for t in tasks if used[t] == 0]
        if empty:
            raise RuntimeError("bn recalibration: no finite batch for task(s) %s (all %s slots non-finite)"
                               % (empty, [plan.count(t) for t in empty]))
        stale = [m.bn.moving_mean.name.split(":")[0] for m in layers
                 if all(np.array_equal(a, v.numpy())
                        for a, v in zip(before[id(m)], m.bn.non_trainable_variables))]
        if stale:
            raise RuntimeError("bn recalibration: %d BN layer(s) got no update (stale level-4 statistics): %s"
                               % (len(stale), stale))
        if val:
            gap.update(self._bn_gap(self._recal_validate("bnrecal_post/", step), "gap_post"))
        if freeze:
            self.model.set_bn_stats_frozen(True)
            self.rebuild_steps()
        event = {"event": "bn_recalibration", "num_batches": len(plan),
                 "planned": {t: plan.count(t) for t in tasks}, "batches": used,
                 "skipped_nonfinite": skipped, "n_layers_updated": len(layers) - len(stale),
                 "layers_without_updates": stale, "elapsed_s": time.perf_counter() - t_start,
                 "freeze": bool(freeze), "gap": gap}
        if self.logger is not None:
            self.logger.log_event(step, event)
        return event

    # ------------------------------------------------------------------ loop
    def _log(self, step, metrics):
        if self.logger is not None:
            self.logger.log_metrics(step, metrics)

    def _validate_and_log(self, step):
        t0 = time.perf_counter()
        val = self.validate()
        if self.level == 1:                    # level-1 runs only (lower levels: keys unchanged)
            val[VAL_WALL_KEY] = time.perf_counter() - t0
        self.val_history.append(dict(val, step=step))
        self._log(step, val)
        if self.checkpoint is not None and self.best_metric in val:
            self.checkpoint.maybe_save_best(val[self.best_metric])
        for name, key in self.extra_best.items():
            # Only when the checkpoint was built with this extra best file (main.py wires it).
            if (self.checkpoint is not None and key in val
                    and name in getattr(self.checkpoint, "extra_best", {})):
                self.checkpoint.maybe_save_best(val[key], name=name)
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

    def window_center_rms_by_task(self):
        """{task: mean center RMS of its last overfit_window finite steps} for the trained tasks
        (None for a task with fewer steps)."""
        w = int(self.cfg.overfit_window)
        out = {}
        for t in self.linear_tasks:
            vals = []
            for h in reversed(self.history):
                if h["task"] == t and h["finite"]:
                    vals.append(h["center_rms_mm"])
                    if len(vals) == w:
                        break
            out[t] = float(np.mean(vals)) if len(vals) == w else None
        return out

    def _window_center_rms(self):
        """Worst linear task's window mean (`window_center_rms_by_task`), None until every linear
        task has a full window (or without linear tasks). Equals the plain window mean with a
        single linear task."""
        vals = list(self.window_center_rms_by_task().values())
        if not vals or any(v is None for v in vals):
            return None
        return max(vals)

    def window_deform_by_task(self):
        """{'deform': dict(soft_dice_tmpl, fold_frac) means of its last overfit_window finite
        steps, or None with fewer steps} when deform is trained, else {}."""
        if DEFORM not in self.schedule.tasks:
            return {}
        w = int(self.cfg.overfit_window)
        recs = []
        for h in reversed(self.history):
            if h["task"] == DEFORM and h["finite"]:
                recs.append(h)
                if len(recs) == w:
                    break
        if len(recs) < w:
            return {DEFORM: None}
        return {DEFORM: {"soft_dice_tmpl": float(np.mean([h["soft_dice_tmpl"] for h in recs])),
                         "fold_frac": float(np.mean([h["diag_fold_frac"] for h in recs]))}}

    def overfit_status(self):
        """{task: True / False / None (window not full yet)}: linear = window mean center RMS <
        overfit_threshold_mm; deform = window mean soft template Dice >= overfit_dice_threshold
        and window mean folding fraction < overfit_fold_max."""
        cfg = self.cfg
        out = {}
        for t, v in self.window_center_rms_by_task().items():
            out[t] = None if v is None else bool(v < float(cfg.overfit_threshold_mm))
        for t, d in self.window_deform_by_task().items():
            out[t] = None if d is None else bool(d["soft_dice_tmpl"] >= float(cfg.overfit_dice_threshold)
                                                 and d["fold_frac"] < float(cfg.overfit_fold_max))
        return out

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
        best_window_dice = None
        p_skip = float(cfg.cascade_skip_prob) if self.cascade is not None else 0.0
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
            skip = cascade_skip_draw(cfg.seed, s, p_skip) if p_skip > 0 else False
            res = self.train_step(task, batch, skip=skip)
            t2 = time.perf_counter()
            done = s + 1
            rec = {"step": done, "task": task, "lr": lr, "data_ms": 1e3 * (t1 - t0),
                   "compute_ms": 1e3 * (t2 - t1)}
            rec.update(res)
            if p_skip > 0:
                rec["cascade_skip"] = bool(skip)
                acc.update({CASCADE_SKIP_KEY: float(skip)})
            if self.cascade is not None:
                acc.update({"train_%s/data_ms" % task: rec["data_ms"],
                            "train_%s/compute_ms" % task: rec["compute_ms"]})
            if task in ctl_values:
                cm = self.control_metrics(task, extras, ctl_values[task], step=s)
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
                acc.update({pre + k: res[k] for k in ENC_GRAD_KEYS})
                acc.update({pre + "grad_norm": res["grad_norm"]})
                acc.update({pre + k: res[k] for k in TASK_GRAD_NORM_KEYS if k in res})
                if PAIR_COUNT_KEY in res:
                    acc.update({pre + PAIR_COUNT_KEY: res[PAIR_COUNT_KEY]})
                if task == DEFORM:
                    acc.update({pre + k: res[k] for k in DEFORM_LOSS_KEYS})
                    acc.update({pre + k: res[k] for k in DEFORM_NCC_KEYS if k in res})
                    acc.update({"train/grad_norm": res["grad_norm"]})
                    acc.update({pre + k: res[k] for k in DEFORM_DIAG_KEYS})
                    acc.update({pre + k: res[k] for k in DEFORM_CASCADE_DIAG_KEYS if k in res})
                else:
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
                wd = self.window_deform_by_task().get(DEFORM)
                if wd is not None:
                    best_window_dice = (wd["soft_dice_tmpl"] if best_window_dice is None
                                        else max(best_window_dice, wd["soft_dice_tmpl"]))
                status = self.overfit_status()
                if status and all(v is True for v in status.values()):
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
                "best_window_center_rms_mm": best_window,
                "window_center_rms_by_task": self.window_center_rms_by_task(),
                "window_deform": self.window_deform_by_task().get(DEFORM),
                "best_window_soft_dice_tmpl": best_window_dice, "last_val": last_val}
