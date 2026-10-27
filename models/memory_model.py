"""Calibrated analytic GPU memory model, TF-free (milestone 9 phase 1, R3).

`.agents/tasks/milestone-9/design-m9-phase1.md` sections 2.1-2.5 (tensor inventory, phases) and
4.1-4.5 (Workload / predict, inventory construction, coefficients and fit, level-1 grid and
recommendation, step-time model). Stdlib + numpy only; scipy (``least_squares``) is imported
inside `fit`. Python 3.8 syntax (cluster container). No TensorFlow import.

Inventory (design 2.1): bytes per image per voxel of the level, per phase
  F   end of the forward pass (taped tensors + loss terms, ``max + c_loss (sum - max)``),
  Fw  forward moment of each lean-sampler call (timed rule; ``fwd_transient_bound=True`` = the
      order-independent estimate (one overlapping sampler forward), plan NIT N3),
  Fb  loss backward at each lean-sampler call,
  I   integrator backward (recomputed block, or the taped integration backward I0),
  D   deform-decoder backward, one sub-phase per decoder block with the skip-gradient term B_sg
      (``skip_grad_bound=True`` counts every skip gradient at every sub-phase),
  E   encoder backward (all skip gradients alive),
  Z   frozen-stage inference before the tape (cascade only),
  V   validation (eval forward + the eager ``_deform_errors`` pair chunk).
Peak = resident + c_0 + N * V_level * max over F..E, Z (Z counted because the frozen forward runs
inside the measured training step). V is reported separately (``val_peak_gib``).

Model-form decisions recorded in the milestone-9 FEAT-002 findings:
  * Resident: `runtime.optimizers.build_optimizer` creates the Adam slots for ALL trainable
    variables, so every record holds weights + 2 slots (12 B) for every trainable parameter; the
    gradient (4 B) and the AMP fp16 weight cast (2 B) count only for the parts the record's tasks
    touch (design U16 assumed lazily created slots; the GTX `resident_mb` values support all-slot).
  * Default-sampler int32 corner indices are host-resident (design U4); the lean sampler's int64
    indices live on the GPU and are modelled explicitly in its forward transient (plan NIT N2).
  * Hold-out correction (FEAT-004): under `--gradient-checkpointing full` phase I holds BOTH
    recomputed integrator blocks (exp(v) and exp(-v) are independent in the backward graph and
    are recomputed concurrently), not one block at a time as design 3.2 / 4.2 assumed.
  * One-sided hold-out gate (FEAT-004 deviation from design 4.3, orchestrator decision): the
    gate fails only on an under-prediction beyond -20 % (or a hold-out OOM predicted feasible);
    over-predictions are conservative and only counted. Margin = max(0.15, worst under-prediction).
    21 of the 40 local GTX hold-out points are over-predicted by > 20 % (small level-4 AMP / split
    points); the symmetric verdict is kept in `holdout_gate()['symmetric_passed']` and the report.

DALI term (`dali_memory`, tf-intensity-aug): the DALI pipelines hold device memory next to TF for
the whole run, linear in images x voxels. Calibrated on job 10517362 (RTX 8000, level 1, rigid N 8
+ affine N 8 + deform N 4 = 20 images, prefetch 1, exec_async / pipelined on): ~14.0 GiB with the
DALI intensity augmentation, ~4.5 GiB without (--intensity-aug-device tf or --intensity-aug false);
labels and the fixed pool overhead are absorbed in the per-image-voxel coefficient (two points,
prefetch 1 only; the scaling to other levels / group sizes is an assumption). `predict(...,
dali_gib=...)` subtracts it from the budgets; the default 0 keeps the TF-only calibration
records, the fit and the hold-out gate unchanged.
"""
import collections
import json
import math
import os
import sys

import numpy as np

from data_loading import params as DP
from models import width_presets as WP
from models.config import ModelConfig

GIB = float(2 ** 30)
MB = float(2 ** 20)
INT32_MAX = 2 ** 31 - 1
# Largest fp16 forward tensor observed OK: the level-2 linear AMP N = 16 build forward D2 concat
# [16, 80, 112, 96, 256] (design U1).
FP16_FWD_OBSERVED = 16 * 80 * 112 * 96 * 256
DEFAULT_LIMITS = {"fp32": {"fwd": INT32_MAX, "bwd": INT32_MAX},
                  "fp16": {"fwd": FP16_FWD_OBSERVED, "bwd": INT32_MAX}}
BUDGET_GIB = 44.0          # 48 GB cards (RTX 8000 / RTX A6000), design A1
BUDGET_A100_GIB = 36.0     # A100 40 GB
# DALI device memory per image voxel by intensity-aug device (module docstring; 'off' = 'tf').
_DALI_CAL_VOXELS = 20.0 * 160 * 224 * 192
DALI_BYTES_PER_IMAGE_VOXEL = {"dali": 14.0 * GIB / _DALI_CAL_VOXELS, "tf": 4.5 * GIB / _DALI_CAL_VOXELS}
DALI_BYTES_PER_IMAGE_VOXEL["off"] = DALI_BYTES_PER_IMAGE_VOXEL["tf"]
DALI_DEVICES = ("dali", "tf", "off")
DALI_CALIBRATION = ("job 10517362, RTX 8000, level 1, rigid N 8 + affine N 8 + deform N 4, prefetch 1: "
                    "14.0 GiB with DALI intensity augmentation, 4.5 GiB without")
DEFAULT_DALI_GROUPS = {"rigid": 8, "affine": 8, "deform": 4}
DEFAULT_MARGIN = 0.15
HOLDOUT_GATE = 0.20
FIT_TOLERANCE = 0.15
OOM_HINGE_FRACTION = 0.98

GPUS = ("rtx8000", "a6000", "a100", "gtx1080")
GPU_FAMILY = {"rtx8000": "cluster", "a6000": "cluster", "a100": "cluster", "gtx1080": "gtx1080"}
TASKS = ("rigid", "affine", "deform")
LINEAR_TASKS = ("rigid", "affine")
CHECKPOINTING = ("none", "integrator", "decoder", "both", "full")
CASCADES = ("none", "frozen4", "chain42")
PHASES = ("F", "Fw", "Fb", "I", "D", "E", "Z", "V")
TRAIN_PHASES = ("F", "Fw", "Fb", "I", "D", "E", "Z")
VAL_CHUNKS = (16, 8, 4, 2, 1)
KNOBS = ("lean_sampler", "lean_losses", "split_group_conv")
N_FROZEN = {"none": 0, "frozen4": 1, "chain42": 2}

# Coefficients (design 4.3): name -> (initial, lower, upper, expected, meaning).
COEFF_TABLE = collections.OrderedDict([
    ("c_net", (1.0, 0.6, 2.0, 1.03, "taped decoder and head tensors")),
    ("c_enc32", (1.0, 0.6, 1.6, 1.10, "encoder tensors, fp32 compute")),
    ("c_enc16", (1.0, 0.6, 1.6, 0.89, "encoder tensors, fp16 compute (AMP)")),
    ("c_loss", (1.0, 0.0, 1.0, 0.5, "share of the non-largest loss terms alive at F")),
    ("c_rec", (1.0, 0.5, 2.0, 1.0, "recomputed decoder block in its backward phase")),
    ("c_warp", (1.0, 0.6, 2.0, 1.02, "default-sampler tensors (absorbs U4)")),
    ("c_bwd", (2.0, 0.0, 4.0, 2.0, "backward transient, x C_in x 4 B of the largest block")),
    ("c0_cluster", (0.3, 0.0, 2.0, 0.3, "allocator / context constant (GiB), RTX 8000 / A6000 / A100")),
    ("c0_gtx1080", (0.1, 0.0, 1.0, 0.08, "allocator / context constant (GiB), GTX 1080 at 6000 MB")),
])
COEFF_NAMES = tuple(COEFF_TABLE)

# --------------------------------------------------------------------- inventory constants (2.1)
F_TRANSIENT = 150.0            # loss gradients at F
I_TRANSIENT_DEFAULT = 250.0    # one default composition's backward
LINEAR_BATCH = 10.0            # image + transform (linear tasks)
SMOOTH = 36.0                  # smoothness term
LNCC_DEFAULT = 168.0           # box filters and moments per LNCC call
LNCC_LEAN = 8.0                # LNCC under tf.recompute_grad (recompute inputs)
PDISP_DEFAULT_EXTRA = 24.0     # pair displacement output + add (default sampler)
FIELDS = 36.0                  # v, u, u_inv
CASCADE_FIELDS = 24.0          # U, U_inv at the level
LEVEL1_UP_FIELDS = 24.0        # upsampled U, U_inv at level 1 (chain)
VAL_FIELDS = 48.0              # u, u_inv, v, v_raw (validation)
COORDS = 12.0


def sampler_default(c):
    """Default sampler, taped bytes per output voxel (int32 indices host-resident, U4)."""
    return 84.0 + 32.0 * c


def lean_retained(field):
    """Lean sampler retained bytes: coordinates (+ the captured input for a field composition)."""
    return 24.0 if field else 12.0


def lean_fwd_moment(c, sequenced=False):
    """Lean forward moment T_fwd(C) + 4C = 252 + 36 C (84 + 32 C float + 168 B int64 + output).
    ``sequenced``: the backlogged forward corner sequencing (design 3.7), about 170 + 12 C."""
    return (170.0 + 12.0 * c) if sequenced else (252.0 + 36.0 * c)


def lean_bwd_transient(c, images_grad):
    """Sequenced lean backward transient T_bwd(C) = 88 + 8 C (+ 16 C with the input gradient)."""
    return 88.0 + 8.0 * c + (16.0 * c if images_grad else 0.0)


def lean_bwd_moment(c, images_grad):
    """dy (4 C) + T_bwd(C)."""
    return 4.0 * c + lean_bwd_transient(c, images_grad)


MIN_PAIRS = ("1", "n")      # --min-pairs (mirrors `data_loading.postprocess.MIN_PAIRS`)


def num_pairs(n, p_pair=0.1, min_pairs="1", max_pairs=0):
    """min(max(m, round(p n (n - 1))), n (n - 1)), m = 1 ('1') or n ('n'), like
    `data_loading.postprocess.num_pairs` (float32 product, half to even), then capped at
    ``max_pairs`` K when K > 0 (--max-pairs, retrain-prereqs design 16.3)."""
    if min_pairs not in MIN_PAIRS:
        raise ValueError("min_pairs must be one of %s, got %r" % (MIN_PAIRS, min_pairs))
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, (int, np.integer)) or max_pairs < 0:
        raise ValueError("max_pairs must be an int >= 0 (0 = no cap), got %r" % (max_pairs,))
    total = int(n) * (int(n) - 1)
    if total < 1:
        return 0
    k = int(np.round(np.float32(total) * np.float32(p_pair)))
    k = int(min(max(k, 1 if min_pairs == "1" else int(n)), total))
    return min(k, int(max_pairs)) if max_pairs > 0 else k


# --------------------------------------------------------------------- workload
def resolve_config(model_config):
    """ModelConfig from a ModelConfig, a kwargs dict or a preset name (preset -> three-task model,
    deform decoder on)."""
    if isinstance(model_config, ModelConfig):
        return model_config
    if model_config is None:
        return ModelConfig(deform_decoder=True)
    if isinstance(model_config, str):
        kw = WP.preset_kwargs(model_config)
        kw["deform_decoder"] = True
        return ModelConfig(**kw)
    return ModelConfig(**dict(model_config))


class Workload(object):
    """Inputs of one prediction (design 4.1). ``model_config``: ModelConfig, kwargs dict or preset
    name. ``integration_steps`` None = the config's T. ``build_n`` None = min(n, 2).
    ``tasks_run`` None = (task,) (single-task measurement record); a training workload passes
    ('rigid', 'affine', 'deform'). ``cascade``: 'none' | 'frozen4' (level 2) | 'chain42' (level 1).
    ``min_pairs``: '1' | 'n' (deform pair count, `num_pairs`; records without it read '1')."""

    FIELDS_ = ("level", "task", "n", "model_config", "checkpointing", "amp", "integration_steps",
               "lncc", "p_pdice", "n_classes", "lean_sampler", "lean_losses", "split_group_conv",
               "cascade", "val_pair_chunk", "build_n", "tasks_run", "gpu", "min_pairs")

    def __init__(self, level, task, n, model_config="default", checkpointing="none", amp=False,
                 integration_steps=None, lncc=True, p_pdice=0.1, n_classes=DP.N_TRAIN_LABELS,
                 lean_sampler=False, lean_losses=False, split_group_conv=False, cascade="none",
                 val_pair_chunk=16, build_n=None, tasks_run=None, gpu="rtx8000",
                 min_pairs="1"):
        self.level = int(level)
        if self.level not in DP.LEVEL_SHAPE:
            raise ValueError("level must be one of 4, 2, 1, got %r" % (level,))
        if task not in TASKS:
            raise ValueError("task must be one of %s, got %r" % (TASKS, task))
        self.task = task
        self.n = int(n)
        if self.n < 1:
            raise ValueError("n must be >= 1")
        self.model_config = model_config
        self.cfg = resolve_config(model_config)
        if self.cfg.block != "single":
            raise ValueError("memory model: only block='single' is modelled")
        if checkpointing not in CHECKPOINTING:
            raise ValueError("checkpointing must be one of %s, got %r" % (CHECKPOINTING, checkpointing))
        self.checkpointing = checkpointing
        self.amp = bool(amp)
        self.integration_steps = int(self.cfg.integration_steps if integration_steps is None
                                     else integration_steps)
        self.lncc = bool(lncc)
        self.p_pdice = float(p_pdice)
        self.n_classes = int(n_classes)
        self.lean_sampler = bool(lean_sampler)
        self.lean_losses = bool(lean_losses)
        self.split_group_conv = bool(split_group_conv)
        if cascade is True:
            cascade = {2: "frozen4", 1: "chain42"}.get(self.level, "none")
        elif cascade is False or cascade is None:
            cascade = "none"
        if cascade not in CASCADES:
            raise ValueError("cascade must be one of %s, got %r" % (CASCADES, cascade))
        self.cascade = cascade
        self.val_pair_chunk = int(val_pair_chunk)
        self.build_n = int(min(self.n, 2) if build_n is None else build_n)
        self.tasks_run = tuple(tasks_run) if tasks_run else (task,)
        if gpu not in GPUS:
            raise ValueError("gpu must be one of %s, got %r" % (GPUS, gpu))
        self.gpu = gpu
        if min_pairs not in MIN_PAIRS:
            raise ValueError("min_pairs must be one of %s, got %r" % (MIN_PAIRS, min_pairs))
        self.min_pairs = min_pairs
        if task == "deform" and not self.cfg.deform_decoder:
            raise ValueError("task 'deform' needs a model_config with deform_decoder=True")

    def replace(self, **changes):
        kw = {k: getattr(self, k) for k in self.FIELDS_}
        kw.update(changes)
        if "n" in changes and "build_n" not in changes:
            kw["build_n"] = None
        return Workload(**kw)

    def to_dict(self):
        d = {k: getattr(self, k) for k in self.FIELDS_ if k != "model_config"}
        d["tasks_run"] = list(self.tasks_run)
        d["model_config"] = (self.model_config if isinstance(self.model_config, str)
                             else {k: (list(v) if isinstance(v, tuple) else v)
                                   for k, v in self.cfg.to_dict().items()})
        return d

    @classmethod
    def from_record(cls, r):
        """Workload of a calibration / hold-out record."""
        kw = {k: r[k] for k in cls.FIELDS_ if k in r and k != "model_config"}
        kw["model_config"] = r.get("model_config", r.get("widths", "default"))
        return cls(**kw)


# --------------------------------------------------------------------- architecture (4.2)
def _frac(h):
    return 1.0 / (8.0 ** h)


def _halvings(level):
    """(h_l2, h_l3): halvings of the L2 and L3 inputs (pool2 at level 1, pool3 at levels 2/1)."""
    return {4: (0, 0), 2: (0, 1), 1: (1, 2)}[level]


def architecture(cfg, level):
    """Encoder / decoder stage list derived from `models.group_net` (no TF).

    Returns dict(encoder=[...], decoder=[...], head_c=w1); each stage
    dict(name, c_in, c_out, h_in, h_out, group, pre_act) (+ skip, skip_c, skip_h for decoder)."""
    f = cfg.filters
    g = cfg.group_config == "A"
    h2, h3 = _halvings(level)
    enc = [dict(name="stem", c_in=1, c_out=f[0], h_in=0, h_out=0, group=g, pre_act=False),
           dict(name="l2", c_in=f[0], c_out=f[1], h_in=h2, h_out=h2, group=g, pre_act=True),
           dict(name="l3", c_in=f[1], c_out=f[2], h_in=h3, h_out=h3, group=g, pre_act=True),
           dict(name="l4", c_in=f[2], c_out=f[3], h_in=h3, h_out=h3 + 1, group=g, pre_act=True),
           dict(name="l5", c_in=f[3], c_out=f[4], h_in=h3 + 1, h_out=h3 + 2, group=g, pre_act=True),
           dict(name="l6", c_in=f[4], c_out=f[5], h_in=h3 + 2, h_out=h3 + 3, group=g, pre_act=True)]
    dec = []
    if cfg.deform_decoder:
        w = cfg.resolved_decoder_widths()
        x = f[5]
        for name, wi, skip in zip(("d5", "d4", "d3", "d2", "d1"), w, ("l5", "l4", "l3", "l2", "stem")):
            s = [e for e in enc if e["name"] == skip][0]
            dec.append(dict(name=name, c_in=x + s["c_out"], c_out=wi, h_in=s["h_out"], h_out=s["h_out"],
                            group=True, pre_act=True, skip=skip, skip_c=s["c_out"], skip_h=s["h_out"]))
            x = wi
    return {"encoder": enc, "decoder": dec, "head_c": (dec[-1]["c_out"] if dec else 0)}


def _taped_channels(st, split, n):
    """Channels kept under the tape per image (design 2.1 block inventories)."""
    c = float(st["c_in"])
    if not st["pre_act"]:
        return 2.0 if st["group"] else 1.0          # stem: its (2-channel) concat
    if st["group"]:
        return (4.0 * c + c / n) if split else 5.0 * c
    return 4.0 * c


def _split_applies(st, split):
    return bool(split and st["group"] and st["pre_act"])


# --------------------------------------------------------------------- inventory
def _piece(**kw):
    return {k: float(v) for k, v in kw.items() if v}


def _add(*pieces):
    out = {}
    for p in pieces:
        for k, v in p.items():
            out[k] = out.get(k, 0.0) + v
    return out


def _scale(p, s):
    return {k: v * s for k, v in p.items()}


class _Inventory(object):
    """Per-image byte inventory of one workload (per voxel of the level); coefficient-free."""

    def __init__(self, w, fwd_sequenced=False, fp16_onehot=False):
        self.w = w
        cfg = w.cfg
        self.arch = architecture(cfg, w.level)
        self.b = 2.0 if w.amp else 4.0
        self.prec = "fp16" if w.amp else "fp32"
        n = float(w.n)
        b = self.b
        split = w.split_group_conv
        enc, dec = self.arch["encoder"], self.arch["decoder"]
        self.enc_bytes = sum(_taped_channels(st, _split_applies(st, split), n) * b * _frac(st["h_in"])
                             for st in enc)
        self.kept_skips = b + sum(st["c_out"] * b * _frac(st["h_out"]) for st in enc)
        self.enc_tr = max(st["c_in"] * 4.0 * _frac(st["h_in"]) for st in enc)
        self.dec_bytes = 0.0
        self.sub = []                       # decoder sub-phases fine -> coarse: (name, tr, skip_grad)
        if dec:
            for st in dec:
                self.dec_bytes += _taped_channels(st, _split_applies(st, split), n) * b * _frac(st["h_in"])
            hc = float(self.arch["head_c"])
            self.dec_bytes += 3.0 * hc * b + hc * 4.0       # head: x, y, z + fp32 conv input
            self.sub.append(("head", hc * 4.0, 0.0))
            for st in reversed(dec):
                self.sub.append((st["name"], st["c_in"] * 4.0 * _frac(st["h_in"]),
                                 st["skip_c"] * b * _frac(st["skip_h"])))
        self.skip_grads_all = sum(s[2] for s in self.sub)
        self.n_pairs = num_pairs(w.n, w.p_pdice, w.min_pairs)
        self.share = float(self.n_pairs) / n if w.n > 0 else 0.0
        self.fwd_sequenced = bool(fwd_sequenced)
        self.onehot_b = 2.0 if fp16_onehot else 4.0
        if w.task == "deform":
            self._deform()

    # ------------------------------------------------------------------ deform pieces
    def _deform(self):
        w, lean, ll = self.w, self.w.lean_sampler, self.w.lean_losses
        C = float(w.n_classes)
        T = w.integration_steps
        cas = w.cascade != "none"
        oh = self.onehot_b * C
        self.batch = _piece(other=8.0 + 4.0 + oh + FIELDS + (CASCADE_FIELDS if cas else 0.0)
                            + (LEVEL1_UP_FIELDS if (cas and w.level == 1) else 0.0))
        mode = w.checkpointing
        n_comp = 2 * T
        # Integration: taped (none / decoder) or recomputed block(s) in I. Hold-out correction
        # (FEAT-004, milestone-9 report): under `full` the two integrator blocks exp(v) / exp(-v)
        # are independent in the backward graph, so TF recomputes both concurrently and both
        # blocks' tapes (and transients) are alive in I (measured T6 - T7 = 2 compositions per
        # removed step; the design assumed one block at a time).
        nblk = 2.0 if mode == "full" else 1.0
        if lean:
            self.integ_taped = _piece(other=n_comp * lean_retained(True) + (COORDS if T else 0.0))
            blk = T if mode == "full" else n_comp
            mom = max(lean_fwd_moment(3.0, self.fwd_sequenced), lean_bwd_moment(3.0, True)) if blk else 0.0
            self.integ_block = _piece(other=(nblk * ((blk - 1) * lean_retained(True) + COORDS + mom))
                                      if blk else 0.0)
            self.integ_i0_tr = _piece(other=lean_bwd_moment(3.0, True) if T else 0.0)
        else:
            blk = T if mode == "full" else n_comp
            self.integ_taped = _piece(warp=n_comp * sampler_default(3.0))
            self.integ_block = _piece(warp=nblk * blk * sampler_default(3.0),
                                      other=nblk * I_TRANSIENT_DEFAULT if blk else 0.0)
            self.integ_i0_tr = _piece(other=I_TRANSIENT_DEFAULT if T else 0.0)
        if cas:
            self.cascade = (_piece(other=2 * lean_retained(True)) if lean
                            else _piece(warp=2 * sampler_default(3.0)))
        else:
            self.cascade = {}
        sh = self.share
        slices = 0.0 if ll else 8.0 * (C - 1.0)
        target = 4.0 * (C - 1.0)
        lncc = LNCC_LEAN if ll else LNCC_DEFAULT
        terms = []
        # (name, order, share, C, warp, other, after, coords)
        if lean:
            terms.append(dict(name="tdice", order=0, share=1.0, C=C, warp=0.0,
                              other=COORDS + 4.0 * C + slices + SMOOTH, after=4.0 * C + slices, coords=COORDS))
            terms.append(dict(name="pdice", order=2, share=sh, C=C, warp=0.0,
                              other=sh * (COORDS + 4.0 * C + slices + target), after=sh * (4.0 * C + slices),
                              coords=sh * COORDS))
            if w.lncc:
                for name, order, s in (("tncc", 3, 1.0), ("pncc", 4, sh)):
                    terms.append(dict(name=name, order=order, share=s, C=1.0, warp=0.0,
                                      other=s * (COORDS + 4.0 + lncc), after=s * (4.0 + lncc), coords=s * COORDS))
            self.pdisp = _piece(other=sh * (lean_retained(True) + 12.0))
        else:
            terms.append(dict(name="tdice", order=0, share=1.0, C=C, warp=sampler_default(C),
                              other=slices + SMOOTH, after=0.0, coords=0.0))
            terms.append(dict(name="pdice", order=2, share=sh, C=C, warp=sh * sampler_default(C),
                              other=sh * (slices + target), after=0.0, coords=0.0))
            if w.lncc:
                for name, order, s in (("tncc", 3, 1.0), ("pncc", 4, sh)):
                    terms.append(dict(name=name, order=order, share=s, C=1.0, warp=s * sampler_default(1.0),
                                      other=s * lncc, after=0.0, coords=0.0))
            self.pdisp = _piece(warp=sh * sampler_default(3.0), other=sh * PDISP_DEFAULT_EXTRA)
        self.terms = terms


# --------------------------------------------------------------------- coefficients
def default_coefficients():
    return collections.OrderedDict((k, v[0]) for k, v in COEFF_TABLE.items())


def full_coefficients(coeffs=None):
    co = default_coefficients()
    if coeffs:
        src = coeffs.get("coefficients", coeffs) if isinstance(coeffs, dict) else coeffs
        for k in COEFF_NAMES:
            if k in src:
                co[k] = float(src[k])
    return co


def load_coefficients(path):
    with open(path) as fh:
        return full_coefficients(json.load(fh))


def _val(piece, co, prec):
    s = 0.0
    for k, v in piece.items():
        if k == "enc":
            s += (co["c_enc16"] if prec == "fp16" else co["c_enc32"]) * v
        elif k == "net":
            s += co["c_net"] * v
        elif k == "rec":
            s += co["c_rec"] * v
        elif k == "warp":
            s += co["c_warp"] * v
        elif k == "tr":
            s += co["c_bwd"] * v
        else:
            s += v
    return s


def _agg(values, c_loss):
    """Loss terms alive at F: max_term + c_loss (sum - max) (design 2.1, U15)."""
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    m = max(vals)
    return m + c_loss * (sum(vals) - m)


def _term_value(t, co, drop_after=False, drop_coords=False):
    other = t["other"] - (t["after"] if drop_after else 0.0) - (t["coords"] if drop_coords else 0.0)
    return co["c_warp"] * t["warp"] + other


def _deform_phases(inv, co, skip_grad_bound, fwd_transient_bound):
    """Per-image bytes of the training phases of a deform workload -> (dict phase -> bytes, info)."""
    w, prec = inv.w, inv.prec
    mode = w.checkpointing
    full = mode == "full"
    V = lambda p: _val(p, co, prec)  # noqa: E731
    enc = {"enc": inv.enc_bytes}
    skips = {"enc": inv.kept_skips}
    dec_cls = "rec" if mode in ("decoder", "both", "full") else "net"
    dec = {dec_cls: inv.dec_bytes}
    dec_net_in_f = mode in ("none", "integrator")
    integ_in_f = mode in ("none", "decoder")
    base = _add(inv.batch, skips if full else enc, dec if dec_net_in_f else {},
                inv.integ_taped if integ_in_f else {}, inv.cascade)
    base_v = V(base)
    pdisp_v = V(inv.pdisp)
    terms = inv.terms
    tvals = [_term_value(t, co) for t in terms]
    loss = _agg(tvals, co["c_loss"])
    F = base_v + pdisp_v + loss + F_TRANSIENT
    info = {}
    ph = collections.OrderedDict((p, 0.0) for p in ("F", "Fw", "Fb", "I", "D", "E"))
    ph["F"] = F
    if w.lean_sampler:
        seq = inv.fwd_sequenced
        calls = []      # (name, value) Fw
        bcalls = []     # (name, value) Fb
        # Forward moments before the losses (cascade compositions, integrator forward).
        if inv.cascade or w.integration_steps:
            calls.append(("integ/cascade", base_v + lean_fwd_moment(3.0, seq)))
        loss_calls = [(t["order"], t["name"], t) for t in terms] + [(1, "pdisp", None)]
        for order, name, t in sorted(loss_calls, key=lambda x: x[0]):
            if t is None:
                s, c, grad = inv.share, 3.0, True
                after, coords = s * 12.0, s * COORDS
            else:
                s, c, grad = t["share"], t["C"], False
                after, coords = t["after"], t["coords"]
            mom = s * lean_fwd_moment(c, seq)
            if fwd_transient_bound:
                if t is None:
                    fw = F - F_TRANSIENT - after - coords + mom
                else:
                    tv = [_term_value(u, co, drop_after=(u is t), drop_coords=(u is t)) for u in terms]
                    fw = base_v + pdisp_v + _agg(tv, co["c_loss"]) + mom
            else:
                prior = [_term_value(u, co) for u in terms if u["order"] < order]
                fw = base_v + (pdisp_v if order > 1 else 0.0) + _agg(prior, co["c_loss"]) + mom
            calls.append((name, fw))
            if t is None:
                fb = F - F_TRANSIENT - after + s * lean_bwd_moment(c, grad)
            else:
                tv = [_term_value(u, co, drop_after=(u is t)) for u in terms]
                fb = base_v + pdisp_v + _agg(tv, co["c_loss"]) + s * lean_bwd_moment(c, grad)
            bcalls.append((name, fb))
        name, ph["Fw"] = max(calls, key=lambda x: x[1])
        info["Fw_call"] = name
        name, ph["Fb"] = max(bcalls, key=lambda x: x[1])
        info["Fb_call"] = name
    # I: recomputed integrator block, or the taped integration backward I0.
    if w.integration_steps:
        if mode in ("integrator", "both", "full"):
            alive = _add(inv.batch, skips if full else enc, dec if mode == "integrator" else {})
            ph["I"] = V(alive) + V(inv.integ_block)
        else:
            alive = _add(inv.batch, enc, dec if mode == "none" else {}, inv.integ_taped)
            ph["I"] = V(alive) + V(inv.integ_i0_tr)
    # D: decoder backward sub-phases (fine -> coarse), skip gradients of finer blocks.
    d_alive = V(_add(inv.batch, skips if full else enc)) + V(dec)
    best, done = None, 0.0
    for name, tr, sg in inv.sub:
        bsg = inv.skip_grads_all if skip_grad_bound else done
        val = d_alive + co["c_bwd"] * tr + bsg
        if best is None or val > best[1]:
            best = (name, val)
        done += sg
    if best is not None:
        ph["D"] = best[1]
        info["D_block"] = best[0]
    # E: encoder backward with every skip gradient.
    ph["E"] = V(_add(inv.batch, enc)) + inv.skip_grads_all + co["c_bwd"] * inv.enc_tr
    return ph, info


def _linear_phases(inv, co):
    ph = collections.OrderedDict((p, 0.0) for p in ("F", "Fw", "Fb", "I", "D", "E"))
    ph["E"] = LINEAR_BATCH + _val({"enc": inv.enc_bytes}, co, inv.prec) + co["c_bwd"] * inv.enc_tr
    return ph, {}


def _frozen_live(cfg, level, deform):
    """Largest live set (bytes per voxel of ``level``) of a float32 frozen inference forward: the
    largest unsplit group concat + its NCDHW copy + the block output, plus the skips."""
    arch = architecture(cfg, level)
    stages = arch["encoder"] + (arch["decoder"] if deform else [])
    skips = sum(st["c_out"] * 4.0 * _frac(st["h_out"]) for st in arch["encoder"])
    blk = max((2.0 * st["c_in"] * (2.0 if st["group"] else 1.0) * 4.0 * _frac(st["h_in"])
               + st["c_out"] * 4.0 * _frac(st["h_out"])) for st in stages)
    return skips + blk


def _z_bytes(inv):
    """Phase Z per image (bytes per voxel of the trainable level), 0 without a cascade."""
    w = inv.w
    if w.cascade == "none":
        return 0.0
    deform = w.task == "deform"
    cfg = w.cfg
    lv = [4] if w.cascade == "frozen4" else [2, 4]
    live = 0.0
    vl = float(np.prod(DP.LEVEL_SHAPE[w.level]))
    for fl in lv:
        ratio = float(np.prod(DP.LEVEL_SHAPE[fl])) / vl
        live = max(live, _frozen_live(cfg, fl, deform and cfg.deform_decoder) * ratio)
    pre = (8.0 + inv.onehot_b * w.n_classes) if deform else LINEAR_BATCH
    return pre + live + (CASCADE_FIELDS if deform else 0.0)


def _eval_live(inv):
    """Eval-forward live set of the trainable stage (no tape), bytes per image per level voxel."""
    b, w = inv.b, inv.w
    stages = inv.arch["encoder"] + inv.arch["decoder"]
    best = 0.0
    for st in stages:
        sp = _split_applies(st, w.split_group_conv)
        cat = (st["c_in"] + st["c_in"] / float(w.n)) if sp else (2.0 * st["c_in"] if st["group"] else st["c_in"])
        best = max(best, (st["c_in"] + cat) * b * _frac(st["h_in"]) + st["c_out"] * b * _frac(st["h_out"]))
    return inv.kept_skips + best


def val_pair_bytes(n_classes):
    """Per validation pair and voxel: one-hot gather + moved + sampler forward transient + coords."""
    c = float(n_classes)
    return 4.0 * c + 4.0 * c + 3.0 * (4.0 + 4.0 * c) + COORDS


# --------------------------------------------------------------------- resident
def resident_bytes(w):
    """Resident bytes without c_0: Adam slots + fp32 weight for every trainable parameter
    (`runtime.optimizers.build_optimizer`), gradient + AMP cast for touched parts, frozen stages."""
    parts = WP.count_params(w.cfg)
    total = parts["total"]
    touched = parts["encoder"]
    tasks = set(w.tasks_run)
    if tasks & set(LINEAR_TASKS):
        touched += parts["linear_decoder"]
    if "deform" in tasks:
        touched += parts["deform_decoder"]
    r = 12.0 * total + 4.0 * touched + (2.0 * touched if w.amp else 0.0)
    r += 4.0 * total * N_FROZEN[w.cascade]
    return r


# --------------------------------------------------------------------- tensor limits (2.4)
def _shape(level, h, n, c):
    s = DP.LEVEL_SHAPE[level]
    return (int(n),) + tuple(int(x) // (2 ** h) for x in s) + (int(c),)


def _tensor(desc, shape, dtype, direction):
    el = int(np.prod([int(x) for x in shape], dtype=np.int64))
    return {"desc": desc, "shape": list(shape), "dtype": dtype, "elements": el,
            "bytes": el * (2 if dtype == "fp16" else 4), "direction": direction}


def _block_tensors(arch, level, n, dtype, split, tag, direction, decoder=True):
    out = []
    stages = arch["encoder"] + (arch["decoder"] if decoder else [])
    for st in stages:
        nm = "%s/%s" % (tag, st["name"])
        if _split_applies(st, split):
            out.append(_tensor(nm + " conv_a input", _shape(level, st["h_in"], n, st["c_in"]), dtype, direction))
        else:
            ch = (2 if st["group"] else 1) * st["c_in"]
            out.append(_tensor(nm + " group concat" if st["group"] else nm + " input",
                               _shape(level, st["h_in"], n, ch), dtype, direction))
        out.append(_tensor(nm + " output", _shape(level, st["h_out"], n, st["c_out"]), dtype, direction))
    return out


def tensor_inventory(w):
    """(training-step tensors [bwd], forward-only tensors [frozen, validation], build tensors)."""
    arch = architecture(w.cfg, w.level)
    dt = "fp16" if w.amp else "fp32"
    deform = w.task == "deform"
    train = _block_tensors(arch, w.level, w.n, dt, w.split_group_conv, "train", "bwd", decoder=deform)
    if deform:
        hc = arch["head_c"]
        train.append(_tensor("train/svf head input", _shape(w.level, 0, w.n, hc), "fp32", "bwd"))
        train.append(_tensor("train/fields", _shape(w.level, 0, w.n, 3), "fp32", "bwd"))
        train.append(_tensor("train/one-hot", _shape(w.level, 0, w.n, w.n_classes), "fp32", "bwd"))
    fwd = []
    if w.cascade != "none":
        for fl in ([4] if w.cascade == "frozen4" else [2, 4]):
            fa = architecture(w.cfg, fl)
            fwd += _block_tensors(fa, fl, w.n, "fp32", False, "frozen%d" % fl, "fwd",
                                  decoder=deform and w.cfg.deform_decoder)
    if deform:
        fwd += _block_tensors(arch, w.level, w.n, dt, w.split_group_conv, "val", "fwd", decoder=True)
    build = _block_tensors(arch, w.level, w.build_n, dt, False, "build", "fwd", decoder=w.cfg.deform_decoder)
    return train, fwd, build


def load_tensor_limits(path=None):
    lim = json.loads(json.dumps(DEFAULT_LIMITS))
    if path:
        with open(path) as fh:
            d = json.load(fh)
        v = d.get("verified_max_elements", d)
        for dt in ("fp32", "fp16"):
            for direction in ("fwd", "bwd"):
                x = (v.get(dt) or {}).get(direction)
                if x:
                    lim[dt][direction] = int(x)
    return lim


def limit_flags(w, limits=None):
    """-> (largest training-step tensor dict, list of limit-flag strings)."""
    lim = limits or DEFAULT_LIMITS
    train, fwd, build = tensor_inventory(w)
    flags = []
    for t in train + fwd + build:
        bound = lim[t["dtype"]][t["direction"]]
        if t["elements"] > bound:
            flags.append("%s %s %s %d elements > %d (%s)" % (t["desc"], "x".join(str(s) for s in t["shape"]),
                                                             t["dtype"], t["elements"], bound, t["direction"]))
    largest = max(train, key=lambda t: t["elements"])
    return largest, flags


# --------------------------------------------------------------------- DALI term
def dali_memory(level, groups=None, aug_device="tf"):
    """DALI device memory of the training pipelines (module docstring, `DALI_CALIBRATION`).

    ``groups``: {task: group size} of the pipelines that run (default `DEFAULT_DALI_GROUPS`);
    ``aug_device``: 'dali' (augmentation in DALI), 'tf' or 'off'. Returns dict total_gib,
    per_pipeline {task: GiB}, bytes_per_image_voxel, calibration."""
    level = DP.check_level(level)
    if aug_device not in DALI_DEVICES:
        raise ValueError("aug_device must be one of %s, got %r" % (DALI_DEVICES, aug_device))
    groups = dict(DEFAULT_DALI_GROUPS if groups is None else groups)
    vox = float(np.prod(DP.LEVEL_SHAPE[level]))
    bpv = DALI_BYTES_PER_IMAGE_VOXEL[aug_device]
    per = collections.OrderedDict()
    for t, n in groups.items():
        if t not in TASKS:
            raise ValueError("unknown task %r (expected one of %s)" % (t, TASKS))
        if int(n) < 0:
            raise ValueError("group size of %s must be >= 0, got %r" % (t, n))
        per[t] = int(n) * vox * bpv / GIB
    return {"total_gib": float(sum(per.values())), "per_pipeline": per, "bytes_per_image_voxel": bpv,
            "calibration": DALI_CALIBRATION}


# --------------------------------------------------------------------- predict (4.1)
def predict(workload, coeffs=None, margin=0.0, skip_grad_bound=False, fwd_transient_bound=False,
            tensor_limits=None, time_model=None, fwd_sequenced=False, fp16_onehot=False, dali_gib=0.0):
    """Predicted memory of one workload (design 4.1).

    ``dali_gib``: DALI device memory resident next to TF for the whole run (`dali_memory`); the
    feasibility flags use the effective budget ``budget - dali_gib`` for the peak and the
    validation chunks. 0 (default) = the calibrated TF-only model, unchanged.

    ``skip_grad_bound`` and ``fwd_transient_bound`` default to False (the timed rules of 2.1);
    setting either to True selects its conservative variant: every skip gradient counted at every
    decoder sub-phase, resp. the order-independent estimate (one overlapping sampler forward) for
    the lean sampler's forward moment (plan NIT N1, N3). ``fwd_sequenced`` / ``fp16_onehot``
    evaluate backlog items (design 3.7) for the R3.7 verdict only.

    Returns dict: peak_gib, peak_margin_gib, resident_gib (incl. c_0), phases {F, Fw, Fb, I, D, E,
    Z, V: GiB totals, 0 when absent}, peak_phase, components, largest_tensor, limit_flags,
    feasible_44, feasible_36 (at ``margin``), val_peak_gib (at ``workload.val_pair_chunk``),
    val_by_chunk, step_s, step_s_range, per_image_gib.
    """
    w = workload
    co = full_coefficients(coeffs)
    inv = _Inventory(w, fwd_sequenced=fwd_sequenced, fp16_onehot=fp16_onehot)
    if w.task == "deform":
        ph, info = _deform_phases(inv, co, skip_grad_bound, fwd_transient_bound)
    else:
        ph, info = _linear_phases(inv, co)
    vox = float(np.prod(DP.LEVEL_SHAPE[w.level]))
    scale = w.n * vox / GIB
    c0 = co["c0_gtx1080"] if GPU_FAMILY[w.gpu] == "gtx1080" else co["c0_cluster"]
    res = resident_bytes(w) / GIB + c0
    phases = collections.OrderedDict()
    for p, v in ph.items():
        phases[p] = (res + scale * v) if v > 0 else 0.0
    zb = _z_bytes(inv)
    phases["Z"] = (res + scale * zb) if zb > 0 else 0.0
    val_by_chunk = collections.OrderedDict()
    if w.task == "deform":
        pairs = w.n * (w.n - 1)
        fixed = w.n * (8.0 + VAL_FIELDS + (CASCADE_FIELDS if w.cascade != "none" else 0.0))
        live = w.n * _eval_live(inv)
        for ch in sorted(set(VAL_CHUNKS + (w.val_pair_chunk,)), reverse=True):
            pv = min(ch, max(pairs, 1)) * val_pair_bytes(w.n_classes)
            val_by_chunk[ch] = res + vox * (fixed + max(live, pv)) / GIB
        phases["V"] = val_by_chunk[w.val_pair_chunk]
    else:
        phases["V"] = 0.0
    peak_phase = max(TRAIN_PHASES, key=lambda p: phases[p])
    peak = phases[peak_phase]
    largest, flags = limit_flags(w, tensor_limits)
    m = float(margin)
    val = phases["V"]
    ok_val = (lambda b: val * (1 + m) <= b) if w.task == "deform" else (lambda b: True)
    dali = float(dali_gib)
    b44, b36 = BUDGET_GIB - dali, BUDGET_A100_GIB - dali
    out = collections.OrderedDict()
    out["peak_gib"] = peak
    out["peak_margin_gib"] = peak * (1.0 + m)
    out["resident_gib"] = res
    out["phases"] = phases
    out["peak_phase"] = peak_phase
    out["phase_info"] = info
    out["per_image_gib"] = collections.OrderedDict((p, v * vox / GIB) for p, v in ph.items())
    out["components"] = _components(inv, w)
    out["largest_tensor"] = largest
    out["limit_flags"] = flags
    out["feasible_44"] = bool(peak * (1 + m) <= b44 and ok_val(b44) and not flags)
    out["feasible_36"] = bool(peak * (1 + m) <= b36 and ok_val(b36) and not flags)
    out["dali_gib"] = dali
    out["total_gib"] = peak + dali
    out["val_peak_gib"] = val
    out["val_by_chunk"] = val_by_chunk
    st = time_model.predict(w) if time_model is not None else None
    out["step_s"] = st
    out["step_s_range"] = [0.5 * st, 1.5 * st] if st is not None else None
    return out


def _components(inv, w):
    """Readable component list (name, phases, bytes per image per level voxel, class)."""
    mode = w.checkpointing
    full = mode == "full"
    comps = []

    def add(name, phases, nbytes, cls):
        if nbytes:
            comps.append({"name": name, "phases": phases, "bytes": round(float(nbytes), 2), "class": cls})
    add("encoder tape", "E" if full else "F,I,D,E", inv.enc_bytes, "net")
    if w.task != "deform":
        add("batch", "E", LINEAR_BATCH, "other")
        add("encoder backward transient (c_bwd units)", "E", inv.enc_tr, "transient")
        return comps
    add("batch and fields", "all", sum(inv.batch.values()), "other")
    if full:
        add("kept skips + input", "F,Fw,I,D", inv.kept_skips, "net")
    add("decoder + head", "D" if mode in ("decoder", "both", "full") else "F,D",
        inv.dec_bytes, "rec" if mode in ("decoder", "both", "full") else "net")
    if mode in ("none", "decoder"):
        add("integration (taped)", "F,I", sum(inv.integ_taped.values()), "warp" if not w.lean_sampler else "other")
    else:
        add("integration block (recomputed)", "I", sum(inv.integ_block.values()),
            "warp" if not w.lean_sampler else "other")
    add("cascade compositions", "F", sum(inv.cascade.values()), "warp" if not w.lean_sampler else "other")
    add("pair displacement", "F", sum(inv.pdisp.values()), "other")
    for t in inv.terms:
        add("loss term " + t["name"], "F", t["warp"] + t["other"], "warp+other")
    add("F transient", "F", F_TRANSIENT, "other")
    for name, tr, sg in inv.sub:
        add("D transient at %s (c_bwd units)" % name, "D", tr, "transient")
        add("skip gradient of %s" % name, "D,E", sg, "other")
    add("encoder backward transient (c_bwd units)", "E", inv.enc_tr, "transient")
    return comps


# --------------------------------------------------------------------- calibration and fit (4.3)
REQUIRED_FIELDS = ("id", "source", "gpu", "cap_gib", "status", "measure", "lncc", "cascade", "build_n",
                   "level", "task", "n", "checkpointing", "amp", "integration_steps")
STATUSES = ("ok", "oom", "failed_d2", "range", "holdout")


def load_calibration(path):
    with open(path) as fh:
        d = json.load(fh)
    validate_calibration(d)
    return d


def validate_calibration(d):
    """Schema check of the calibration file (ValueError naming the record and field)."""
    if "records" not in d:
        raise ValueError("calibration: missing 'records'")
    seen = set()
    for r in d["records"]:
        rid = r.get("id", "?")
        for k in REQUIRED_FIELDS:
            if k not in r:
                raise ValueError("calibration record %s: missing field %r" % (rid, k))
        if rid in seen:
            raise ValueError("calibration record %s: duplicate id" % rid)
        seen.add(rid)
        if r["status"] not in STATUSES:
            raise ValueError("calibration record %s: status %r not in %s" % (rid, r["status"], STATUSES))
        if r["measure"] != "peak_mb":
            raise ValueError("calibration record %s: measure must be 'peak_mb'" % rid)
        if "model_config" not in r and "widths" not in r:
            raise ValueError("calibration record %s: needs model_config or widths" % rid)
        if r["status"] == "ok" and "peak_mb" not in r:
            raise ValueError("calibration record %s: ok without peak_mb" % rid)
        if r["status"] == "range" and len(r.get("range_gib", ())) != 2:
            raise ValueError("calibration record %s: range needs range_gib [lo, hi]" % rid)
        if r["status"] == "holdout":
            ms = r.get("measured_status")
            if ms not in ("ok", "oom"):
                raise ValueError("calibration record %s: holdout needs measured_status ok|oom" % rid)
            if ms == "ok" and "peak_mb" not in r:
                raise ValueError("calibration record %s: holdout ok without peak_mb" % rid)
        Workload.from_record(r)
    return True


def _pred_gib(r, co, limits=None):
    p = predict(Workload.from_record(r), co, tensor_limits=limits)
    return p


def evaluate_records(records, co, limits=None):
    """Rows (id, status, measured GiB, predicted GiB, err, ok_flag, note) for every record."""
    rows = []
    for r in records:
        p = _pred_gib(r, co, limits)
        pred = p["peak_gib"]
        st = r["status"] if r["status"] != "holdout" else "holdout_" + r["measured_status"]
        meas = r["peak_mb"] / 1024.0 if "peak_mb" in r else None
        err = (pred - meas) / meas if meas else None
        cap = float(r["cap_gib"])
        if st in ("ok",):
            good = abs(err) <= FIT_TOLERANCE
            note = "" if good else "outside +-15 %"
        elif st == "holdout_ok":
            # One-sided gate (FEAT-004 deviation from design 4.3, orchestrator decision): only an
            # under-prediction beyond -20 % fails; over-predictions are conservative and are only
            # flagged (the symmetric numbers stay in the report as an open model-form question).
            good = err >= -HOLDOUT_GATE
            note = ("" if abs(err) <= HOLDOUT_GATE else
                    "over-prediction > +20 % (conservative, not gated)" if good else
                    "under-prediction < -20 % (hold-out gate)")
        elif st in ("oom", "holdout_oom"):
            good = pred > cap or bool(p["limit_flags"])
            note = "infeasible" if good else "OOM predicted feasible (%.2f <= cap %.2f)" % (pred, cap)
        elif st == "failed_d2":
            good = bool(p["limit_flags"]) or pred > cap
            note = ("limit flag: " + p["limit_flags"][0]) if p["limit_flags"] else (
                "infeasible" if good else "failed_d2 without a limit flag")
        else:  # range
            lo, hi = r["range_gib"]
            good = pred <= hi
            note = "range %.1f-%.1f, upper containment %s" % (lo, hi, "ok" if good else "FAILED")
        rows.append({"id": r["id"], "status": st, "meas_gib": meas, "pred_gib": pred, "err": err,
                     "good": bool(good), "note": note, "phase": p["peak_phase"], "cap_gib": cap,
                     "group": r.get("group")})
    return rows


def fast_peak_gib(inv, co):
    """Training peak (incl. Z) of a prebuilt inventory; the coefficient-dependent part of `predict`."""
    w = inv.w
    if w.task == "deform":
        ph, _ = _deform_phases(inv, co, False, False)
    else:
        ph, _ = _linear_phases(inv, co)
    vox = float(np.prod(DP.LEVEL_SHAPE[w.level]))
    c0 = co["c0_gtx1080"] if GPU_FAMILY[w.gpu] == "gtx1080" else co["c0_cluster"]
    s = max(list(ph.values()) + [_z_bytes(inv)])
    return resident_bytes(w) / GIB + c0 + w.n * vox * s / GIB


def _residuals(x, recs, invs):
    co = full_coefficients(dict(zip(COEFF_NAMES, x)))
    res = []
    for r, inv in zip(recs, invs):
        pred = fast_peak_gib(inv, co)
        if r["status"] == "ok":
            meas = r["peak_mb"] / 1024.0
            res.append((pred - meas) / meas)
        elif r["status"] == "oom":
            cap = float(r["cap_gib"])
            res.append(oom_hinge(pred, cap))
    return np.asarray(res, np.float64)


def oom_hinge(pred_gib, cap_gib):
    """OOM hinge residual max(0, 0.98 cap - pred) / cap (0 when already above 0.98 cap)."""
    return max(0.0, OOM_HINGE_FRACTION * cap_gib - pred_gib) / cap_gib


def fit_coefficients(records, limits=None, x0=None):
    """least_squares (soft_l1, f_scale 0.05) on the ok residuals + OOM hinges -> OrderedDict."""
    from scipy.optimize import least_squares
    recs = [r for r in records if r["status"] in ("ok", "oom")]
    lo = np.array([COEFF_TABLE[k][1] for k in COEFF_NAMES])
    hi = np.array([COEFF_TABLE[k][2] for k in COEFF_NAMES])
    if x0 is None:
        x0 = np.array([COEFF_TABLE[k][0] for k in COEFF_NAMES], np.float64)
    x0 = np.clip(np.asarray(x0, np.float64), lo, hi)
    invs = [_Inventory(Workload.from_record(r)) for r in recs]
    sol = least_squares(_residuals, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.05,
                        args=(recs, invs))
    return collections.OrderedDict(zip(COEFF_NAMES, [float(v) for v in sol.x]))


def at_bound(co, rel=1e-4):
    hits = []
    for k in COEFF_NAMES:
        lo, hi = COEFF_TABLE[k][1], COEFF_TABLE[k][2]
        tol = rel * max(1.0, hi - lo)
        if co[k] <= lo + tol:
            hits.append("%s at lower bound %g" % (k, lo))
        elif co[k] >= hi - tol:
            hits.append("%s at upper bound %g" % (k, hi))
    return hits


LOMO_GROUPS = ("fp32_none", "fp32_integrator", "fp32_both", "amp_none", "amp_integrator", "amp_both")


def leave_one_mode_out(records, limits=None, co_full=None):
    """Refit without each level-2 deform group (design 4.3) -> {group: [(id, err), ...]}."""
    out = collections.OrderedDict()
    for g in LOMO_GROUPS:
        keep = [r for r in records if r.get("group") != g]
        held = [r for r in records if r.get("group") == g and r["status"] == "ok"]
        if not held:
            continue
        x0 = [co_full[k] for k in COEFF_NAMES] if co_full else None
        co = fit_coefficients(keep, limits, x0=x0)
        rows = evaluate_records(held, co, limits)
        out[g] = [(row["id"], row["err"]) for row in rows]
    return out


def warmup_excess_gib(records):
    ex = [(r["warmup_peak_mb"] - r["peak_mb"]) / 1024.0 for r in records
          if r.get("warmup_peak_mb") is not None and r.get("peak_mb") is not None]
    return max(ex) if ex else None


def holdout_gate(records, co, limits=None):
    """-> dict(evaluated, passed, max_under_err, max_abs_err, n_over, symmetric_passed, margin, rows).

    One-sided gate (FEAT-004 deviation from design 4.3, orchestrator decision): it passes iff no
    ``ok`` hold-out point is under-predicted by more than 20 % and every hold-out OOM is predicted
    infeasible. Over-predictions are conservative; they are counted (``n_over``: points above
    +20 %) and the symmetric verdict is kept as ``symmetric_passed`` for the report. Margin
    m = max(0.15, worst under-prediction magnitude ``max_under_err``)."""
    hold = [r for r in records if r["status"] == "holdout"]
    rows = evaluate_records(hold, co, limits) if hold else []
    errs = [row["err"] for row in rows if row["status"] == "holdout_ok" and row["err"] is not None]
    mx = max([abs(e) for e in errs]) if errs else 0.0
    under = max([-e for e in errs if e < 0] or [0.0])
    n_over = sum(1 for e in errs if e > HOLDOUT_GATE)
    return {"evaluated": bool(hold), "passed": all(row["good"] for row in rows), "max_under_err": under,
            "max_abs_err": mx, "n_over": n_over, "symmetric_passed": all(row["good"] for row in rows) and mx <= HOLDOUT_GATE,
            "margin": max(DEFAULT_MARGIN, under), "rows": rows}


def _print_rows(rows, out, title):
    out.write("%s\n" % title)
    out.write("  %-44s %-12s %9s %9s %8s  %-5s %s\n" % ("point", "status", "meas GiB", "pred GiB", "err %",
                                                       "phase", "check"))
    for r in rows:
        out.write("  %-44s %-12s %9s %9.2f %8s  %-5s %s%s\n" % (
            r["id"], r["status"], "%.2f" % r["meas_gib"] if r["meas_gib"] is not None else "-",
            r["pred_gib"], "%+.1f" % (100 * r["err"]) if r["err"] is not None else "-",
            r["phase"], "ok" if r["good"] else "FAIL", (" (" + r["note"] + ")") if r["note"] else ""))


def fit(calibration, limits=None, out=None, lomo=True):
    """Fit, report and check (design 4.3). Returns (result dict, failures list)."""
    out = out or sys.stdout
    recs = calibration["records"]
    pre = default_coefficients()
    co = fit_coefficients(recs, limits)
    rows = evaluate_records([r for r in recs if r["status"] != "holdout"], co, limits)
    pre_rows = evaluate_records([r for r in recs if r["status"] in ("ok",)], pre, limits)
    out.write("Fitted coefficients (initial -> fitted, bounds):\n")
    for k in COEFF_NAMES:
        ini, lo, hi, exp, meaning = COEFF_TABLE[k]
        out.write("  %-11s %6.3f -> %7.4f  [%g, %g]  expected %.2f  %s\n" % (k, ini, co[k], lo, hi, exp, meaning))
    hits = at_bound(co)
    for h in hits:
        out.write("  NOTE: %s (model-form check: a coefficient at a bound is reported, design 4.3)\n" % h)
    _print_rows(rows, out, "Accuracy (fitted):")
    ok_errs = [r["err"] for r in rows if r["status"] == "ok"]
    pre_errs = [r["err"] for r in pre_rows]
    out.write("ok points: %d, fitted error range %+.1f ... %+.1f %% (pre-fit %+.1f ... %+.1f %%)\n" % (
        len(ok_errs), 100 * min(ok_errs), 100 * max(ok_errs), 100 * min(pre_errs), 100 * max(pre_errs)))
    lomo_res = {}
    if lomo:
        lomo_res = leave_one_mode_out(recs, limits, co)
        out.write("Leave-one-mode-out (refit without the group, its errors):\n")
        for g, errs in lomo_res.items():
            out.write("  %-16s %s\n" % (g, ", ".join("%s %+.1f %%" % (i, 100 * e) for i, e in errs)))
    gate = holdout_gate(recs, co, limits)
    if gate["evaluated"]:
        _print_rows(gate["rows"], out, "Hold-out (never fitted; one-sided gate err >= -20 %, OOM infeasible):")
        out.write("hold-out gate (one-sided) %s, worst under-prediction %.1f %%, margin m = %.3f; "
                  "symmetric +-20 %% %s (max |err| %.1f %%, %d points over-predicted by > 20 %%)\n" % (
                      "PASSED" if gate["passed"] else "FAILED", 100 * gate["max_under_err"], gate["margin"],
                      "passed" if gate["symmetric_passed"] else "FAILED", 100 * gate["max_abs_err"],
                      gate["n_over"]))
    else:
        out.write("Hold-out: no hold-out records yet (FEAT-004 adds them); margin m = 0.15\n")
    wex = warmup_excess_gib(recs)
    out.write("warmup_excess_gib = %s (accepted, not budgeted; design 4.1, U12)\n" % (
        "%.3f" % wex if wex is not None else "n/a"))
    failures = [r for r in rows if not r["good"]]
    for r in failures:
        out.write("FIT FAILURE: %s %s: %s\n" % (r["id"], r["status"], r["note"] or "outside tolerance"))
    result = collections.OrderedDict()
    result["version"] = 1
    result["coefficients"] = co
    result["initial"] = default_coefficients()
    result["bounds"] = {k: [COEFF_TABLE[k][1], COEFF_TABLE[k][2]] for k in COEFF_NAMES}
    result["at_bound"] = hits
    result["fit_tolerance"] = FIT_TOLERANCE
    result["n_ok"] = len(ok_errs)
    result["n_oom"] = sum(1 for r in recs if r["status"] == "oom")
    result["err_range"] = [min(ok_errs), max(ok_errs)]
    result["prefit_err_range"] = [min(pre_errs), max(pre_errs)]
    result["leave_one_mode_out"] = {g: dict(e) for g, e in lomo_res.items()}
    result["holdout"] = {"evaluated": gate["evaluated"], "passed": gate["passed"], "one_sided": True,
                         "max_under_err": gate["max_under_err"], "max_abs_err": gate["max_abs_err"],
                         "n_over": gate["n_over"], "symmetric_passed": gate["symmetric_passed"],
                         "margin": gate["margin"]}
    result["warmup_excess_gib"] = wex
    result["failures"] = [r["id"] for r in failures]
    return result, failures


# --------------------------------------------------------------------- step-time model (4.5)
class StepTimeModel(object):
    """t = a_gpu F_conv + b_gpu W_warp + t0_gpu (design 4.5). RTX 8000 / A6000 coefficients come
    from one linear and one deform point (a from linear, b exactly from the deform point, so there
    is no residual); the A100 is the RTX 8000 time x r_task (U17). Every estimate carries
    x0.5 ... x1.5 (U9)."""

    def __init__(self, spec):
        self.spec = spec
        self.t0 = float(spec.get("t0_s", 0.1))
        self.r_task = dict(spec.get("r_task", {"linear": 0.63, "deform": 0.70}))
        lin = spec["rtx8000"]["linear"]
        dfm = spec["rtx8000"]["deform"]
        wl = Workload.from_record(lin["workload"])
        wd = Workload.from_record(dfm["workload"])
        fl, _ = self.work(wl)
        fd, wwd = self.work(wd)
        self.a = max(float(lin["seconds"]) - self.t0, 1e-6) / fl
        self.b = max(float(dfm["seconds"]) - self.t0 - self.a * fd, 0.0) / max(wwd, 1.0)
        self.level4_scalar = None       # set by the m7 consistency check (A100 level 4)

    @staticmethod
    def work(w):
        """(F_conv FLOPs, W_warp voxel-channel passes) per step."""
        vox = float(np.prod(DP.LEVEL_SHAPE[w.level]))
        n = float(w.n)
        arch = architecture(w.cfg, w.level)
        k3 = float(w.cfg.kernel_size) ** 3

        def conv_flops(stages, split):
            tot = 0.0
            for st in stages:
                if not st["pre_act"]:
                    cin = 2.0 if st["group"] else 1.0
                elif _split_applies(st, split):
                    cin = st["c_in"] * (1.0 + 1.0 / n)
                else:
                    cin = st["c_in"] * (2.0 if st["group"] else 1.0)
                tot += 2.0 * k3 * cin * st["c_out"] * _frac(st["h_out"])
            return tot * vox * n
        fp = 1.0 if w.amp else 2.2
        deform = w.task == "deform"
        mode = w.checkpointing
        enc_f = conv_flops(arch["encoder"], w.split_group_conv)
        F = enc_f * (3.0 + (1.0 if (deform and mode == "full") else 0.0)) * fp
        if deform:
            dec_f = conv_flops(arch["decoder"], w.split_group_conv) + 2.0 * k3 * arch["head_c"] * 3 * vox * n
            F += dec_f * (3.0 + (1.0 if mode in ("decoder", "both", "full") else 0.0)) * fp
        if w.cascade != "none":
            for fl in ([4] if w.cascade == "frozen4" else [2, 4]):
                fa = architecture(w.cfg, fl)
                ratio = float(np.prod(DP.LEVEL_SHAPE[fl])) / vox
                st = fa["encoder"] + (fa["decoder"] if (deform and w.cfg.deform_decoder) else [])
                F += conv_flops(st, False) * ratio * 2.2
        W = 0.0
        if deform:
            bwd = 4.0 if w.lean_sampler else 3.0
            p = float(num_pairs(w.n, w.p_pdice, w.min_pairs))
            rec = 1.0 if mode in ("integrator", "both", "full") else 0.0
            T = w.integration_steps
            W += 2 * T * n * vox * 3.0 * (1.0 + bwd + rec)
            if w.cascade != "none":
                W += 2 * n * vox * 3.0 * (1.0 + bwd)
            C = float(w.n_classes)
            calls = n * C + p * C + p * 3.0 + ((n + p) if w.lncc else 0.0)
            W += calls * vox * (1.0 + bwd)
        return F, W

    def predict(self, w, gpu=None):
        gpu = gpu or w.gpu
        if gpu == "gtx1080":
            return None
        F, W = self.work(w)
        t = self.a * F + self.b * W + self.t0
        if gpu == "a100":
            if w.level == 4 and self.level4_scalar is not None:
                return t * self.level4_scalar
            return t * self.r_task["deform" if w.task == "deform" else "linear"]
        return t


def m7_consistency(t_rtx, r_task, measured_mixed_s, tol=0.2):
    """Level-4 A100 check (design 4.5): predicted (t_r + t_a + 2 t_d) / 4 from the RTX 8000 times x
    r_task vs the measured m7 median. Within +-tol keep r_task, else one scalar s = measured / RTX
    mixed mean replaces the level-4 ratios. -> dict(mode, ratio, s, a100)."""
    a100 = {k: t_rtx[k] * r_task["deform" if k == "deform" else "linear"] for k in ("rigid", "affine", "deform")}
    pred = (a100["rigid"] + a100["affine"] + 2.0 * a100["deform"]) / 4.0
    ratio = pred / float(measured_mixed_s)
    if abs(ratio - 1.0) <= tol:
        return {"mode": "r_task", "ratio": ratio, "s": None, "a100": a100}
    rtx_mixed = (t_rtx["rigid"] + t_rtx["affine"] + 2.0 * t_rtx["deform"]) / 4.0
    s = float(measured_mixed_s) / rtx_mixed
    return {"mode": "scalar", "ratio": ratio, "s": s, "a100": {k: v * s for k, v in t_rtx.items()}}


# --------------------------------------------------------------------- level-1 grid and recommendation (4.4)
def knob_subsets():
    out = []
    for ls in (False, True):
        for ll in (False, True):
            for sp in (False, True):
                out.append({"lean_sampler": ls, "lean_losses": ll, "split_group_conv": sp})
    return out


def knob_label(k):
    """Short knob label: '-', L (lean sampler), LL (lean losses), S (split), LS, LLS (all three), ..."""
    key = (bool(k["lean_sampler"]), bool(k["lean_losses"]), bool(k["split_group_conv"]))
    return {(False, False, False): "-", (True, False, False): "L", (False, True, False): "LL",
            (False, False, True): "S", (True, False, True): "LS", (True, True, False): "L+LL",
            (False, True, True): "LL+S", (True, True, True): "LLS"}[key]


def n_knobs(k):
    return sum(1 for x in KNOBS if k[x])


def level1_workload(widths, n, T, knobs, gpu="rtx8000", task="deform", amp=True, checkpointing="full"):
    return Workload(1, task, n, model_config=widths, checkpointing=checkpointing, amp=amp, integration_steps=T,
                    lncc=True, cascade="chain42", tasks_run=TASKS, gpu=gpu, val_pair_chunk=16, **knobs)


def feasible(p, margin, budget):
    """Steps 2(a)-(d): peak (incl. Z), a validation chunk, no limit flag -> (bool, chunk).
    The budget is reduced by the prediction's resident DALI memory (``p['dali_gib']``, 0 if absent)."""
    budget = float(budget) - float(p.get("dali_gib", 0.0))
    if p["limit_flags"] or p["peak_gib"] * (1 + margin) > budget:
        return False, None
    if p["phases"]["V"] == 0.0 and not p["val_by_chunk"]:
        return True, None
    for ch in VAL_CHUNKS:
        if p["val_by_chunk"].get(ch, float("inf")) * (1 + margin) <= budget:
            return True, ch
    return False, None


def choose_t(f7, p7, f6, p6, t6_ok, margin, budget=BUDGET_GIB):
    """T rule (design 4.4 step 4) for equal widths and knobs -> 7, 6 or None (infeasible)."""
    if t6_ok and f6 and not f7:
        return 6
    if t6_ok and f6 and f7 and p7 * (1 + margin) > 0.9 * budget and p6 * (1 + margin) <= 0.9 * budget:
        return 6
    if f7:
        return 7
    return None


def read_t6_verdict(path):
    """-> (t6_ok, warning or None). Missing file / missing or false flag -> T = 6 excluded."""
    if not path:
        return False, "no --t6-verdict: T = 6 excluded (fail-safe, design 4.4 step 1)"
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (IOError, OSError, ValueError) as exc:
        return False, "--t6-verdict %s unreadable (%s): T = 6 excluded" % (path, exc)
    if d.get("t6_accuracy_ok") is True:
        return True, None
    return False, "t6_accuracy_ok is not true in %s: T = 6 excluded" % path


def _params(widths):
    return WP.count_params(resolve_config(widths))["total"]


def candidate_table(co, margin, t6_ok, limits=None, time_model=None, ns=range(2, 9), widths=None,
                    rule="timed", budget=BUDGET_GIB, dali_gib=0.0):
    """All level-1 candidates (AMP, full) -> list of dicts after the T rule, per (widths, knobs, N)."""
    widths = list(widths or WP.PRESETS)
    kw = {"skip_grad_bound": rule == "skip", "fwd_transient_bound": rule == "fwd", "dali_gib": dali_gib}
    rows = []
    for wd in widths:
        params = _params(wd)
        for k in knob_subsets():
            for n in ns:
                preds = {}
                for T in ((7, 6) if t6_ok else (7,)):
                    wl = level1_workload(wd, n, T, k)
                    p = predict(wl, co, margin, tensor_limits=limits, time_model=time_model, **kw)
                    ok, ch = feasible(p, margin, budget)
                    preds[T] = (p, ok, ch)
                p7, f7, c7 = preds[7]
                if 6 in preds:
                    p6, f6, c6 = preds[6]
                    t = choose_t(f7, p7["peak_gib"], f6, p6["peak_gib"], t6_ok, margin, budget)
                else:
                    t = 7 if f7 else None
                p, ok, ch = preds[t] if t else preds[7]
                rows.append({"widths": wd, "knobs": k, "n": n, "T": t or 7, "feasible": bool(t), "params": params,
                             "peak_gib": p["peak_gib"], "peak_margin_gib": p["peak_gib"] * (1 + margin),
                             "phase": p["peak_phase"], "phases": p["phases"], "val_pair_chunk": ch,
                             "limit_flags": p["limit_flags"], "step_s": p["step_s"],
                             "feasible_a100": feasible(p, margin, BUDGET_A100_GIB)[0],
                             "feasible_a100_no_margin": feasible(p, 0.0, BUDGET_A100_GIB)[0]})
    return rows


def _rank_key(r):
    return (-r["params"], n_knobs(r["knobs"]), r["step_s"] if r["step_s"] is not None else 0.0)


def per_n_table(rows, ns=range(2, 9)):
    """N -> the widest feasible row (params > T rule [resolved per widths/knobs] > fewer knobs > time)."""
    out = []
    for n in ns:
        cand = sorted([r for r in rows if r["n"] == n and r["feasible"]], key=_rank_key)
        out.append(cand[0] if cand else {"n": n, "feasible": False})
    return out


def _row_brief(r):
    if not r.get("feasible"):
        return {"n": r["n"], "widths": None}
    return {"n": r["n"], "widths": r["widths"], "knobs": r["knobs"], "integration_steps": r["T"],
            "peak_gib": round(r["peak_gib"], 3), "peak_margin_gib": round(r["peak_margin_gib"], 3),
            "phase": r["phase"], "params": r["params"], "a100": bool(r["feasible_a100"]),
            "a100_no_margin": bool(r["feasible_a100_no_margin"]), "val_pair_chunk": r["val_pair_chunk"]}


def print_per_n(table, out, title):
    out.write("%s\n" % title)
    out.write("  %-3s %-8s %-6s %-3s %8s %10s %-5s %8s %s\n" % ("N", "widths", "knobs", "T", "peak", "w/ margin",
                                                             "phase", "params", "A100 (margin / none)"))
    for r in table:
        if not r.get("feasible"):
            out.write("  %-3d none feasible\n" % r["n"])
            continue
        out.write("  %-3d %-8s %-6s %-3d %8.2f %10.2f %-5s %7.2fM %s / %s\n" % (
            r["n"], r["widths"], knob_label(r["knobs"]), r["T"], r["peak_gib"], r["peak_margin_gib"], r["phase"],
            r["params"] / 1e6, "yes" if r["feasible_a100"] else "no", "yes" if r["feasible_a100_no_margin"] else "no"))


def grid(co, margin=DEFAULT_MARGIN, limits=None, out=None, level=1, dali_gib=0.0):
    """The full R3.4 grid (design 4.4): widths x checkpointing x AMP x T x knobs -> largest feasible
    deform N (2..8) and its peak; linear N 4..16 per widths x task x AMP; frozen-chain cost per N."""
    out = out or sys.stdout
    cascade = {1: "chain42", 2: "frozen4", 4: "none"}[level]
    res = {"deform": [], "linear": [], "frozen": []}
    out.write("Level-%d deform grid (margin %.2f, budget %.0f GiB; largest feasible N in 2..8):\n" % (
        level, margin, BUDGET_GIB))
    out.write("  %-8s %-10s %-4s %-2s %-6s %4s %8s %-5s\n" % ("widths", "ckpt", "amp", "T", "knobs", "N", "peak",
                                                           "phase"))
    for wd in WP.PRESETS:
        for ck in ("none", "both", "full"):
            for amp in (False, True):
                for T in (6, 7):
                    for k in knob_subsets():
                        best = None
                        for n in range(2, 9):
                            wl = Workload(level, "deform", n, wd, ck, amp, T, lncc=True, cascade=cascade,
                                          tasks_run=TASKS, **k)
                            p = predict(wl, co, margin, tensor_limits=limits, dali_gib=dali_gib)
                            if feasible(p, margin, BUDGET_GIB)[0]:
                                best = (n, p)
                        row = {"widths": wd, "checkpointing": ck, "amp": amp, "T": T, "knobs": k,
                               "largest_n": best[0] if best else None,
                               "peak_gib": best[1]["peak_gib"] if best else None,
                               "phase": best[1]["peak_phase"] if best else None}
                        res["deform"].append(row)
                        out.write("  %-8s %-10s %-4s %-2d %-6s %4s %8s %-5s\n" % (
                            wd, ck, "y" if amp else "n", T, knob_label(k), row["largest_n"] or "-",
                            "%.2f" % row["peak_gib"] if best else "-", row["phase"] or "-"))
    out.write("Level-%d linear grid (largest feasible N in 4..16):\n" % level)
    for wd in WP.PRESETS:
        for task in LINEAR_TASKS:
            for amp in (False, True):
                for sp in (False, True):
                    best = None
                    for n in range(4, 17):
                        wl = Workload(level, task, n, wd, "none", amp, cascade=cascade, tasks_run=TASKS,
                                      split_group_conv=sp)
                        p = predict(wl, co, margin, tensor_limits=limits, dali_gib=dali_gib)
                        if feasible(p, margin, BUDGET_GIB)[0]:
                            best = (n, p)
                    row = {"widths": wd, "task": task, "amp": amp, "split_group_conv": sp,
                           "largest_n": best[0] if best else None,
                           "peak_gib": best[1]["peak_gib"] if best else None}
                    res["linear"].append(row)
                    out.write("  %-8s %-6s amp %-1s split %-1s  N %4s  peak %s\n" % (
                        wd, task, "y" if amp else "n", "y" if sp else "n", row["largest_n"] or "-",
                        "%.2f" % row["peak_gib"] if best else "-"))
    if cascade != "none":
        out.write("Frozen-chain cost per N (phase Z total GiB / frozen weights GiB / persistent U,U^-1 GiB):\n")
        vox = float(np.prod(DP.LEVEL_SHAPE[level]))
        for wd in WP.PRESETS:
            cells = []
            for n in range(2, 9):
                wl = level1_workload(wd, n, 7, {"lean_sampler": True, "lean_losses": False,
                                                "split_group_conv": True}) if level == 1 else Workload(
                    level, "deform", n, wd, "full", True, cascade=cascade, tasks_run=TASKS)
                p = predict(wl, co, 0.0, tensor_limits=limits, dali_gib=dali_gib)
                fw = N_FROZEN[cascade] * 4.0 * _params(wd) / GIB
                uu = n * vox * CASCADE_FIELDS / GIB
                cells.append({"n": n, "z_gib": p["phases"]["Z"], "frozen_weights_gib": fw, "fields_gib": uu})
            res["frozen"].append({"widths": wd, "per_n": cells})
            out.write("  %-8s %s\n" % (wd, "  ".join("N%d %.1f/%.2f/%.2f" % (c["n"], c["z_gib"], c["frozen_weights_gib"],
                                                                           c["fields_gib"]) for c in cells)))
    return res


def _largest_n(fn, ns):
    best = None
    for n in ns:
        if fn(n):
            best = n
    return best


def _recipe(level, widths, T, task, n, co, limits, time_model, knobs=None, checkpointing=None, amp=None):
    if level == 2:
        ck = checkpointing or "both"
        amp = True if amp is None else amp
        cas = "frozen4"
    else:
        ck = checkpointing or "none"
        amp = False if amp is None else amp
        cas = "none"
    k = knobs or {"lean_sampler": False, "lean_losses": False, "split_group_conv": False}
    if task != "deform":
        k = {"lean_sampler": False, "lean_losses": False, "split_group_conv": k["split_group_conv"]}
        ck = "none"
    wl = Workload(level, task, n, widths, ck, amp, T, lncc=True, cascade=cas, tasks_run=TASKS, **k)
    return predict(wl, co, 0.0, tensor_limits=limits, time_model=time_model)


def recommend(co, calibration, n_target=4, margin=None, t6_verdict=None, limits=None, time_model=None,
              out=None, coefficients_path="models/memory_coefficients.json", dali_gib=0.0):
    """Level-1 recommendation (design 4.4 steps 1-6). Returns (recommendation dict, exit code)."""
    out = out or sys.stdout
    recs = calibration["records"]
    warnings = []
    gate = holdout_gate(recs, co, limits)
    if gate["evaluated"] and not gate["passed"]:
        _print_rows(gate["rows"], out, "Hold-out gate FAILED:")
        out.write("EXIT 4: hold-out gate bypass (under-prediction < -20 % or a hold-out OOM predicted feasible); "
                  "stop for Jay (design 4.3)\n")
        return {"holdout_gate": "failed"}, 4
    if not gate["evaluated"]:
        warnings.append("no hold-out records: margin = 0.15 (gate not evaluated yet)")
    m = gate["margin"] if margin is None else float(margin)
    t6_ok, wmsg = read_t6_verdict(t6_verdict)
    if wmsg:
        warnings.append(wmsg)
    for wmsg in warnings:
        out.write("WARNING: %s\n" % wmsg)
    tables = {}
    rows_by_rule = {}
    for rule in ("timed", "skip", "fwd"):
        rows = candidate_table(co, m, t6_ok, limits, time_model, rule=rule, dali_gib=dali_gib)
        rows_by_rule[rule] = rows
        tables[rule] = per_n_table(rows)
    # Grid summary (AMP, full): largest feasible N per widths (any knobs, timed rule).
    out.write("Level-1 grid summary (AMP, full, chain42, LNCC 1/1, margin %.2f; timed rule):\n" % m)
    for wd in WP.PRESETS:
        fe = [r for r in rows_by_rule["timed"] if r["widths"] == wd and r["feasible"]]
        if fe:
            best = sorted(fe, key=lambda r: (-r["n"], n_knobs(r["knobs"])))[0]
            out.write("  %-8s %6.2fM  largest N %d (%s, T %d, %.2f GiB, %s)\n" % (
                wd, _params(wd) / 1e6, best["n"], knob_label(best["knobs"]), best["T"], best["peak_gib"],
                best["phase"]))
        else:
            out.write("  %-8s %6.2fM  none feasible at N >= 2\n" % (wd, _params(wd) / 1e6))
    print_per_n(tables["timed"], out, "Per-N table (timed rule; selection):")
    print_per_n(tables["skip"], out, "Per-N table (skip-gradient bound):")
    print_per_n(tables["fwd"], out, "Per-N table (order-independent estimate, one overlapping sampler forward):")
    row = [r for r in tables["timed"] if r["n"] == n_target]
    row = row[0] if row else {"feasible": False}
    if not row.get("feasible"):
        verdict = r37_verdict(co, m, n_target, limits, out)
        return {"verdict": verdict, "n_target": n_target, "margin": m}, 3
    wd, k, T = row["widths"], row["knobs"], row["T"]
    cfg_rows = sorted([r for r in rows_by_rule["timed"] if r["widths"] == wd and r["knobs"] == k and r["feasible"]
                       and r["T"] == T], key=lambda r: r["n"])
    n_star = max(r["n"] for r in cfg_rows)
    star = [r for r in cfg_rows if r["n"] == n_star][0]
    def P(n, mg=None, **kw):
        return predict(level1_workload(wd, n, T, k), co, m if mg is None else mg, tensor_limits=limits,
                       time_model=time_model, dali_gib=dali_gib, **kw)
    p_star = P(n_star)
    p_fwd = P(n_star, fwd_transient_bound=True)
    # Linear at level 1 (rigid and affine have the same inventory).
    lin = lambda n, mg, budget: feasible(predict(level1_workload(wd, n, T, k, task="rigid"), co, mg,  # noqa: E731
                                                 tensor_limits=limits, dali_gib=dali_gib), mg, budget)[0]
    linear_n = _largest_n(lambda n: lin(n, m, BUDGET_GIB), range(4, 17))
    linear_n_a100 = _largest_n(lambda n: lin(n, m, BUDGET_A100_GIB), range(4, 17))
    p_lin = predict(level1_workload(wd, linear_n or 4, T, k, task="rigid"), co, m, tensor_limits=limits,
                    time_model=time_model, dali_gib=dali_gib)
    a100_margin = _largest_n(lambda n: feasible(P(n), m, BUDGET_A100_GIB)[0], range(2, 9))
    a100_nomargin = _largest_n(lambda n: feasible(P(n, mg=0.0), 0.0, BUDGET_A100_GIB)[0], range(2, 9))
    a100_nomargin_fwd = _largest_n(lambda n: feasible(P(n, mg=0.0, fwd_transient_bound=True), 0.0,
                                                      BUDGET_A100_GIB)[0], range(2, 9))
    frow = [r for r in tables["fwd"] if r["n"] == n_star]
    alt = {}
    if frow and frow[0].get("feasible"):
        if frow[0]["widths"] != wd:
            alt["widths"] = frow[0]["widths"]
        for kk in KNOBS:
            if frow[0]["knobs"][kk] != k[kk]:
                alt[kk] = frow[0]["knobs"][kk]
    else:
        alt = {"feasible": False}
    # Pinned retraining recipes (design 6.4) under config*.
    l2d = _recipe(2, wd, T, "deform", 8, co, limits, time_model)
    l2l = _recipe(2, wd, T, "rigid", 16, co, limits, time_model)
    l4d = _recipe(4, wd, T, "deform", 16, co, limits, time_model)
    l4l = _recipe(4, wd, T, "rigid", 32, co, limits, time_model)
    l2_both = _largest_n(lambda n: feasible(_recipe(2, wd, T, "deform", n, co, limits, None), 0.0,
                                            BUDGET_A100_GIB)[0], range(2, 33))
    l2_full = _largest_n(lambda n: feasible(_recipe(2, wd, T, "deform", n, co, limits, None, knobs=k,
                                                    checkpointing="full"), 0.0, BUDGET_A100_GIB)[0], range(2, 33))
    l4_none = _largest_n(lambda n: feasible(_recipe(4, wd, T, "deform", n, co, limits, None), 0.0,
                                            BUDGET_A100_GIB)[0], range(2, 49))
    flags = []
    for name, p in (("level2 deform N8 (m8: AMP, both)", l2d), ("level2 linear N16", l2l),
                    ("level4 deform N16 (m7: fp32, none)", l4d), ("level4 linear N32", l4l)):
        if p["peak_gib"] > BUDGET_A100_GIB or p["limit_flags"]:
            flags.append("%s: %.2f GiB > 36 GiB or limit flag" % (name, p["peak_gib"]))
    # Step times.
    step = {}
    long_run = {}
    m7 = None
    if time_model is not None:
        wl4 = {t: Workload(4, t, 16 if t == "deform" else 32, wd, "none", False, T, cascade="none", tasks_run=TASKS)
               for t in TASKS}
        m7_meas = time_model.spec.get("m7_mixed_a100_s")
        if m7_meas:
            wd7 = {t: Workload(4, t, 16 if t == "deform" else 32, "default", "none", False, 7, cascade="none",
                               tasks_run=TASKS) for t in TASKS}
            m7 = m7_consistency({t: time_model.predict(wd7[t], "rtx8000") for t in TASKS}, time_model.r_task,
                                m7_meas)
            if m7["mode"] == "scalar":
                time_model.level4_scalar = m7["s"]
        for gpu in ("rtx8000", "a100"):
            def rng(t):
                return [round(0.5 * t, 2), round(1.5 * t, 2)] if t is not None else None
            l1d = time_model.predict(level1_workload(wd, n_star, T, k), gpu)
            l1l = time_model.predict(level1_workload(wd, linear_n or 4, T, k, task="rigid"), gpu)
            step[gpu] = {"l4_deform": rng(time_model.predict(wl4["deform"], gpu)),
                         "l4_linear": rng(time_model.predict(wl4["rigid"], gpu)),
                         "l2_deform": rng(time_model.predict(Workload(2, "deform", 8, wd, "both", True, T,
                                                                      cascade="frozen4", tasks_run=TASKS), gpu)),
                         "l2_linear": rng(time_model.predict(Workload(2, "rigid", 16, wd, "none", True, T,
                                                                      cascade="frozen4", tasks_run=TASKS), gpu)),
                         "l1_deform": rng(l1d), "l1_linear": rng(l1l), "source": "model (pre-measurement)"}
            hours = (4000 * l1d + 4000 * l1l) / 3600.0 + 16 * 0.05 + 10.0 / 60.0
            long_run[gpu] = [round(0.5 * hours, 1), round(1.5 * hours, 1)]
    rec = collections.OrderedDict()
    rec["provisional"] = True
    rec["margin"] = m
    rec["n_target"] = n_target
    rec["coefficients"] = coefficients_path
    rec["widths"] = wd
    cfg = resolve_config(wd)
    rec["filters"] = list(cfg.filters)
    rec["decoder_widths"] = list(cfg.resolved_decoder_widths())
    rec["integration_steps"] = T
    rec["knobs"] = collections.OrderedDict([("amp", True), ("gradient_checkpointing", "full")]
                                           + [(kk, bool(k[kk])) for kk in KNOBS])
    rec["level1"] = collections.OrderedDict([
        ("deform_n", n_star), ("n_target", n_target), ("linear_n", linear_n), ("val_pair_chunk", star["val_pair_chunk"]),
        ("deform_peak_gib", round(p_star["peak_gib"], 3)),
        ("deform_peak_margin_gib", round(p_star["peak_gib"] * (1 + m), 3)),
        ("deform_peak_phase", p_star["peak_phase"]),
        ("linear_peak_gib", round(p_lin["peak_gib"], 3)),
        ("deform_peak_fwd_bound_gib", round(p_fwd["peak_gib"], 3)),
        ("deform_peak_fwd_bound_margin_gib", round(p_fwd["peak_gib"] * (1 + m), 3)),
        ("bound_alt_knobs", alt),
        ("n_no_margin", _largest_n(lambda n: feasible(P(n, mg=0.0), 0.0, BUDGET_GIB)[0], range(2, 9))),
        ("a100", collections.OrderedDict([("deform_n_margin", a100_margin), ("deform_n_no_margin", a100_nomargin),
                                          ("linear_n", linear_n_a100),
                                          ("deform_n_no_margin_fwd_bound", a100_nomargin_fwd)]))])
    rec["level2"] = collections.OrderedDict([
        ("recipe", "m8"), ("deform_n", 8), ("linear_n", 16), ("amp", True), ("gradient_checkpointing", "both"),
        ("deform_peak_gib", round(l2d["peak_gib"], 3)), ("linear_peak_gib", round(l2l["peak_gib"], 3)),
        ("largest_deform_n", {"both": l2_both, "full": l2_full})])
    rec["level4"] = collections.OrderedDict([
        ("recipe", "m7"), ("deform_n", 16), ("linear_n", 32), ("amp", False), ("gradient_checkpointing", "none"),
        ("deform_peak_gib", round(l4d["peak_gib"], 3)), ("linear_peak_gib", round(l4l["peak_gib"], 3)),
        ("largest_deform_n", {"none": l4_none})])
    rec["recipe_flags"] = flags
    rec["step_s"] = step
    rec["level1_long_run_h"] = long_run
    rec["m7_consistency"] = m7
    rec["per_n"] = [_row_brief(r) for r in tables["timed"]]
    rec["per_n_skip_bound"] = [_row_brief(r) for r in tables["skip"]]
    rec["per_n_fwd_bound"] = [_row_brief(r) for r in tables["fwd"]]
    rec["holdout_gate"] = {"evaluated": gate["evaluated"], "passed": gate["passed"], "one_sided": True,
                           "max_under_err": gate["max_under_err"], "max_abs_err": gate["max_abs_err"],
                           "n_over": gate["n_over"], "symmetric_passed": gate["symmetric_passed"]}
    rec["dali_gib"] = float(dali_gib)
    rec["t6_allowed"] = t6_ok
    rec["warnings"] = warnings
    _print_recommendation(rec, out)
    return rec, 0


def _print_recommendation(rec, out):
    l1 = rec["level1"]
    k = rec["knobs"]
    if rec.get("dali_gib", 0.0) > 0:
        out.write("  (level-1 feasibility includes %.2f GiB of resident DALI memory)\n" % rec["dali_gib"])
    out.write("Recommendation: widths %s (filters %s, decoder %s), AMP, full, lean sampler %s, lean losses %s, "
              "split %s, T %d\n" % (rec["widths"], rec["filters"], rec["decoder_widths"], k["lean_sampler"],
                                    k["lean_losses"], k["split_group_conv"], rec["integration_steps"]))
    out.write("  level 1: deform N* = %d (n_target %d; %d without the margin), peak %.2f GiB (%.2f with margin "
              "%.2f), phase %s; linear N %s at %.2f GiB; val_pair_chunk %s\n" % (
                  l1["deform_n"], l1["n_target"], l1["n_no_margin"] or 0, l1["deform_peak_gib"],
                  l1["deform_peak_margin_gib"], rec["margin"], l1["deform_peak_phase"], l1["linear_n"],
                  l1["linear_peak_gib"], l1["val_pair_chunk"]))
    out.write("  order-independent estimate (one overlapping sampler forward): %.2f GiB (%.2f with margin); "
              "bound alternative %s\n" % (l1["deform_peak_fwd_bound_gib"], l1["deform_peak_fwd_bound_margin_gib"],
                                          l1["bound_alt_knobs"]))
    a = l1["a100"]
    out.write("  A100 (<= 36 GiB): deform N %s with margin, %s without (%s under the bound), linear N %s\n" % (
        a["deform_n_margin"], a["deform_n_no_margin"], a["deform_n_no_margin_fwd_bound"], a["linear_n"]))
    out.write("  level 2 (m8 recipe): deform N8 %.2f GiB, linear N16 %.2f GiB, largest deform N on A100 %s\n" % (
        rec["level2"]["deform_peak_gib"], rec["level2"]["linear_peak_gib"], rec["level2"]["largest_deform_n"]))
    out.write("  level 4 (m7 recipe): deform N16 %.2f GiB, linear N32 %.2f GiB, largest deform N on A100 %s\n" % (
        rec["level4"]["deform_peak_gib"], rec["level4"]["linear_peak_gib"], rec["level4"]["largest_deform_n"]))
    for f in rec["recipe_flags"]:
        out.write("  FLAG pinned recipe infeasible: %s\n" % f)
    for gpu, st in rec["step_s"].items():
        out.write("  step times %s (s, estimate x0.5..x1.5, %s): %s\n" % (
            gpu, st.get("source"), ", ".join("%s %s-%s" % (kk, v[0], v[1]) for kk, v in st.items()
                                             if isinstance(v, list))))
    if rec.get("m7_consistency"):
        mc = rec["m7_consistency"]
        out.write("  m7 consistency (A100 level 4): predicted / measured %.2f -> %s%s\n" % (
            mc["ratio"], mc["mode"], " (s = %.3f replaces the level-4 r_task)" % mc["s"] if mc["s"] else ""))
    for gpu, h in rec["level1_long_run_h"].items():
        fl = []
        if h[1] > 7 * 24:
            fl.append("> 7 days (partition maximum)")
        if h[1] > 72:
            fl.append("> 72 h (needs a resume chain)")
        out.write("  level-1 long run %s: %s-%s h%s\n" % (gpu, h[0], h[1], (" FLAG " + ", ".join(fl)) if fl else ""))
    out.write("PROVISIONAL until slurm/m9_memory.sh measures it (N %d with margin %.2f, %s without)\n" % (
        l1["deform_n"], rec["margin"], l1["n_no_margin"]))


def print_env(rec, out=None):
    """Shell assignments for the SLURM scripts (design 4.4 step 6, 5.5)."""
    out = out or sys.stdout
    k = rec["knobs"]
    knobs = "--lean-sampler %s --lean-losses %s --split-group-conv %s" % tuple(
        "true" if k[x] else "false" for x in KNOBS)
    order = sorted(WP.PRESETS, key=lambda p: -_params(p))
    i = order.index(rec["widths"])
    neigh = [rec["widths"]] + ([order[i - 1]] if i > 0 else []) + ([order[i + 1]] if i + 1 < len(order) else [])
    extra = "--filters %s --decoder-widths %s --integration-steps %d --amp true --gradient-checkpointing full %s" % (
        ",".join(str(x) for x in rec["filters"]), ",".join(str(x) for x in rec["decoder_widths"]),
        rec["integration_steps"], knobs)
    out.write('EXTRA_ARGS="%s" N=%s DEFORM_N=%s\n' % (extra, rec["level1"]["linear_n"], rec["level1"]["deform_n"]))
    out.write('WIDTHS="%s"\n' % ",".join(neigh))
    out.write('KNOBS="%s"\n' % knobs)


def r37_verdict(co, m, n_target, limits, out):
    """R3.7: no configuration at n_target -> the smallest backlog change (design 3.7) that reaches it."""
    out.write("R3.7 VERDICT: no width preset x knob subset reaches deform N = %d within 44 GiB (margin %.2f)\n" % (
        n_target, m))
    options = (("forward corner sequencing in the lean sampler (design 3.7)", {"fwd_sequenced": True}),
               ("fp16 one-hot (design 3.7)", {"fp16_onehot": True}))
    found = []
    for label, kw in options:
        best = None
        for wd in WP.PRESETS:
            for k in knob_subsets():
                wl = level1_workload(wd, n_target, 7, k)
                base = predict(wl, co, m, tensor_limits=limits)
                p = predict(wl, co, m, tensor_limits=limits, **kw)
                if feasible(p, m, BUDGET_GIB)[0]:
                    cand = (-_params(wd), n_knobs(k), wd, k, base["peak_gib"] - p["peak_gib"], p["peak_gib"])
                    if best is None or cand < best:
                        best = cand
        if best:
            found.append({"change": label, "widths": best[2], "knobs": best[3], "saving_gib": best[4],
                          "peak_gib": best[5]})
    if found:
        found.sort(key=lambda f: f["saving_gib"])
        f = found[0]
        out.write("  smallest extra change: %s -> %s %s at %.2f GiB (saves %.2f GiB)\n" % (
            f["change"], f["widths"], knob_label(f["knobs"]), f["peak_gib"], f["saving_gib"]))
    else:
        out.write("  no backlog item reaches it: architecture change needed (e.g. per-image chunking of D1)\n")
    out.write("EXIT 3: stop for Jay's decision (R3.7)\n")
    return found
