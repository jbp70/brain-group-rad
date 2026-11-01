"""Checkpoint loading for prediction (predict-script design 3.2, 3.3 precision rules).

`load_model(path, branches, precision, device)` resolves a run checkpoint, reads the run's
config.json, builds the trainable stage under the right dtype policy plus the frozen cascade
stages (always float32), restores the weights, checks the frozen-stage fingerprints and returns a
`LoadedModel`. Every failure the user can fix leaves as `LoaderError` (a ValueError subclass; the
CLI maps it to exit 2).

Precision (design 3.2 step 4, 3.3): the trainable stage is built under ``mixed_float16`` iff the
run used AMP, ``precision == 'checkpoint'`` and ``device == 'gpu'``; otherwise under float32
(``device == 'cpu'`` on an AMP checkpoint logs a WARNING: ``precision.trainable = 'float32
(cpu)'``). ``deform_decoder_dtype 'float16'`` would raise under a float32 build, so it is mapped to
``'policy'`` (INFO, recorded). The lean sampler is always used at inference (same forward values,
GPU-resident indices); ``knobs.lean_sampler`` records ``{'checkpoint': <value>, 'used': True}``.

`unchunked_n_max` / `check_unchunked` (design 3.2 step 4b) bound the group size of the unchunked
mode by the int32 per-tensor element limit, from the config alone (no model build).
"""
import json
import logging
import os

from data_loading import params as P
from inference import streaming as ST
from models.config import ModelConfig

logger = logging.getLogger(__name__)

PRECISIONS = ("checkpoint", "float32")
DEVICES = ("gpu", "cpu")
CASCADE_LEVELS = (4, 2, 1)
# Runtime knobs read from config.json args, with the defaults of runs that predate them.
KNOB_DEFAULTS = (("amp", False), ("split_group_conv", False), ("deform_decoder_dtype", "policy"),
                 ("fp16_act_clip", 0.0), ("fp16_act_clip_scope", "deform_decoder"),
                 ("bn_epsilon_override", 0.0))
# Checkpoint keys of the frozen stages per trainable level (main.py / FrozenStage / FrozenChain).
FROZEN_KEYS = {4: (), 2: ("frozen_stage",), 1: ("frozen_stage_4", "frozen_stage_2")}


class LoaderError(ValueError):
    """A checkpoint / config problem the user can fix (CLI exit 2)."""


def frozen_levels_of(level):
    """Levels of the frozen stages in front of a trainable stage at ``level`` (coarse first)."""
    level = P.check_level(level)
    return tuple(lv for lv in CASCADE_LEVELS if lv > level)


class LoadedModel(object):
    """A restored cascade: the trainable stage ``model`` at ``level`` and its frozen stages.

    Attributes: level, model, frozen (None | cascade.FrozenStage | cascade.FrozenChain),
    model_cfg, knobs (dict), step, prefix, run_dir, config (the config.json dict),
    training_group_sizes {group_size, deform_group_size}, precision (dict). Tests construct it
    directly from built models; absent fields default from the model.
    """

    def __init__(self, level, model, frozen=None, model_cfg=None, knobs=None, step=None,
                 prefix=None, run_dir=None, config=None, training_group_sizes=None,
                 precision=None):
        self.level = P.check_level(level)
        if model.level != self.level:
            raise ValueError("LoadedModel: model is level %d, expected %d" % (model.level, self.level))
        expected = {4: 0, 2: 1, 1: 2}[self.level]
        got = 0 if frozen is None else int(getattr(frozen, "n_stages", 1))
        if got != expected:
            raise ValueError("LoadedModel: a level-%d stage needs %d frozen stage(s), got %d"
                             % (self.level, expected, got))
        self.model = model
        self.frozen = frozen
        self.model_cfg = model_cfg if model_cfg is not None else model.config
        if knobs is None:
            knobs = {"amp": model.compute_dtype == "float16",
                     "split_group_conv": bool(model.split_group_conv),
                     "deform_decoder_dtype": model.deform_decoder_dtype,
                     "fp16_act_clip": float(model.fp16_act_clip),
                     "fp16_act_clip_scope": model.fp16_act_clip_scope,
                     "bn_epsilon_override": float(model.bn_epsilon_override),
                     "lean_sampler": {"checkpoint": bool(model.lean_sampler), "used": True}}
        self.knobs = knobs
        self.step = step
        self.prefix = prefix
        self.run_dir = run_dir
        self.config = config
        self.training_group_sizes = training_group_sizes
        self.precision = precision

    def stage_models(self):
        """{level: GroupRegLevelModel} of every cascade stage (frozen ones float32)."""
        if self.level == 4:
            return {4: self.model}
        if self.level == 2:
            return {4: self.frozen.model, 2: self.model}
        return {4: self.frozen.stage4.model, 2: self.frozen.model2, 1: self.model}

    @property
    def levels(self):
        """Cascade levels, coarse to fine (the last one is the trainable stage)."""
        return frozen_levels_of(self.level) + (self.level,)


# ============================================================================ unchunked bound
def _tasks_of(branches):
    """Model tasks of a ``--branches`` value (or an iterable of tasks)."""
    if isinstance(branches, str):
        from inference.runner import BRANCHES      # local import: runner imports this module
        if branches not in BRANCHES:
            raise LoaderError("branches must be one of %s, got %r" % (sorted(BRANCHES), branches))
        return tuple(t for t in BRANCHES[branches] if t is not None)
    return tuple(branches)


def _unchunked_bound(level, model_cfg, tasks, split_group_conv, frozen_levels=None):
    """(n_max, stage level, E_max, tensor) minimising floor((2^31 - 1) / E_max) over the tasks and
    the cascade stages (frozen stages at their own level, unsplit)."""
    level = P.check_level(level)
    tasks = _tasks_of(tasks)
    if frozen_levels is None:
        frozen_levels = frozen_levels_of(level)
    stages = [(int(lv), False) for lv in frozen_levels] + [(level, bool(split_group_conv))]
    best = None
    for lv, split in stages:
        for task in tasks:
            e_max, tensor = ST.max_elements_per_image(lv, model_cfg, (task,), split)
            n_max = int((ST.INT32_LIMIT - 1) // e_max)
            if best is None or n_max < best[0]:
                best = (n_max, lv, e_max, tensor)
    return best


def unchunked_n_max(level, model_cfg, tasks, split_group_conv, frozen_levels=None):
    """Largest group of the unchunked mode: min over ``tasks`` and all cascade stages of
    floor((2^31 - 1) / E_max) (design 3.2 step 4b). Default widths at level 1, split on: deform 4,
    linear 9. ``tasks``: a ``--branches`` value or an iterable of model tasks."""
    return _unchunked_bound(level, model_cfg, tasks, split_group_conv, frozen_levels)[0]


def check_unchunked(n, level, model_cfg, tasks, split_group_conv, frozen_levels=None):
    """LoaderError if ``n`` exceeds `unchunked_n_max`. Returns n_max."""
    n_max, lv, e_max, tensor = _unchunked_bound(level, model_cfg, tasks, split_group_conv,
                                                frozen_levels)
    if int(n) > n_max:
        where = tensor if lv == int(level) else "%s of the frozen level-%d stage" % (tensor, lv)
        raise LoaderError("unchunked mode at level %d supports at most %d subjects (largest tensor "
                          "%s, %d elements per image); use --mode streamed or fewer --n-subjects"
                          % (int(level), n_max, where, e_max))
    return n_max


# ============================================================================ loading
def _read_config(prefix):
    from runtime import checkpoint as C
    cfg_path = C.source_config_path(prefix)
    if cfg_path is None:
        raise LoaderError("no config.json next to the checkpoint %s (expected <run>/config.json "
                          "for <run>/ckpt/<kind>/<name>)" % prefix)
    try:
        with open(cfg_path) as f:
            config = json.load(f)
    except (OSError, ValueError) as e:
        raise LoaderError("cannot read %s: %s" % (cfg_path, e))
    return cfg_path, config


def _knobs(args):
    knobs = {k: d if args.get(k) is None else args[k] for k, d in KNOB_DEFAULTS}
    knobs["amp"] = bool(knobs["amp"])
    knobs["split_group_conv"] = bool(knobs["split_group_conv"])
    knobs["fp16_act_clip"] = float(knobs["fp16_act_clip"])
    knobs["bn_epsilon_override"] = float(knobs["bn_epsilon_override"])
    knobs["lowres_method"] = args.get("lowres_method")          # recorded only (design 2.1)
    knobs["lean_sampler"] = {"checkpoint": bool(args.get("lean_sampler", False)), "used": True}
    return knobs


def _precision_plan(knobs, precision, device):
    """(global policy of the trainable build, precision record, deform_decoder_dtype used)."""
    amp = knobs["amp"]
    dd = knobs["deform_decoder_dtype"]
    if amp and precision == "checkpoint" and device == "gpu":
        policy, trainable = "mixed_float16", "mixed_float16"
    else:
        policy = "float32"
        trainable = "float32"
        if amp and precision == "checkpoint" and device == "cpu":
            logger.warning("--device cpu on an AMP checkpoint: the trainable stage runs in float32 "
                           "(differs from the GPU AMP run by fp16 rounding only)")
            trainable = "float32 (cpu)"
    used = dd
    if dd == "float16" and policy == "float32":
        used = "policy"
        logger.info("deform_decoder_dtype 'float16' under a float32 build -> 'policy'")
    record = {"requested": precision, "device": device, "trainable": trainable, "frozen": "float32",
              "deform_decoder_dtype": {"checkpoint": dd,
                                       "used": "policy (float32 build)" if used != dd else dd}}
    return policy, record, used


def _restore(prefix, model, extra):
    import tensorflow as tf
    try:
        status = tf.train.Checkpoint(model=model, **extra).restore(prefix)
        status.expect_partial()
        status.assert_existing_objects_matched()
    except (AssertionError, ValueError, tf.errors.NotFoundError,
            tf.errors.InvalidArgumentError) as e:
        raise LoaderError("restore of %s failed: %s: %s" % (prefix, type(e).__name__, e))


def _check_fingerprints(level, config, frozen, cfg_path):
    from runtime import checkpoint as C
    rec = config.get("cascade") or {}
    if level == 2:
        saved = rec.get("fingerprint")
        if saved is None:
            raise LoaderError("%s: a level-2 run needs cascade.fingerprint (frozen level 4)" % cfg_path)
        if not C.fingerprints_match(saved, frozen.fingerprint):
            raise LoaderError("%s: frozen level-4 weights %s do not match cascade.fingerprint %s"
                              % (cfg_path, frozen.fingerprint, saved))
        return
    stages = rec.get("stages") or []
    current = frozen.fingerprints
    found = sorted(int(s.get("level", -1)) for s in stages)
    if found != sorted(current):
        raise LoaderError("%s: a level-1 run needs cascade.stages for levels %s, got %s"
                          % (cfg_path, sorted(current), found))
    for s in stages:
        lv = int(s["level"])
        if not C.fingerprints_match(s.get("fingerprint"), current[lv]):
            raise LoaderError("%s: frozen level-%d weights %s do not match cascade.stages "
                              "fingerprint %s" % (cfg_path, lv, current[lv], s.get("fingerprint")))


def load_model(checkpoint_path, branches, precision="checkpoint", device="gpu"):
    """Restore a run checkpoint for prediction -> `LoadedModel` (design 3.2 steps 1-9).

    checkpoint_path: a checkpoint prefix or a checkpoint directory (its latest checkpoint).
    branches: a ``--branches`` value (rigid, affine, deform, affine+deform).
    precision: 'checkpoint' (the run's policy) or 'float32'. device: 'gpu' or 'cpu' (only the
    precision rule; device placement is the caller's).
    """
    import tensorflow as tf
    from models import cascade as CS
    from models.group_net import GroupRegLevelModel
    from models.precision import policy_scope
    from runtime import checkpoint as C

    if precision not in PRECISIONS:
        raise LoaderError("precision must be one of %s, got %r" % (PRECISIONS, precision))
    if device not in DEVICES:
        raise LoaderError("device must be one of %s, got %r" % (DEVICES, device))
    tasks = _tasks_of(branches)
    try:
        prefix = os.path.abspath(C.resolve_checkpoint(checkpoint_path))
    except ValueError as e:
        raise LoaderError("checkpoint %s: %s" % (checkpoint_path, e))
    cfg_path, config = _read_config(prefix)
    try:
        model_cfg = ModelConfig.from_dict(config["model_config"])
    except (KeyError, TypeError, ValueError) as e:
        raise LoaderError("%s: bad model_config: %s" % (cfg_path, e))
    args = config.get("args") or {}
    if args.get("level") is None:
        raise LoaderError("%s: args.level missing" % cfg_path)
    level = int(args["level"])
    if level not in CASCADE_LEVELS:
        raise LoaderError("%s: args.level %r is not one of %s" % (cfg_path, level, CASCADE_LEVELS))
    knobs = _knobs(args)
    if "deform" in tasks and not model_cfg.deform_decoder:
        raise LoaderError("branches %r need a deform decoder, but the model of %s has none "
                          "(model_config.deform_decoder false)" % (branches, cfg_path))
    policy, prec, dd_used = _precision_plan(knobs, precision, device)
    try:
        with policy_scope(policy):
            model = GroupRegLevelModel(
                level, model_cfg, gradient_checkpointing="none", lean_sampler=True,
                split_group_conv=knobs["split_group_conv"], fp16_act_clip=knobs["fp16_act_clip"],
                fp16_act_clip_scope=knobs["fp16_act_clip_scope"], deform_decoder_dtype=dd_used,
                bn_epsilon_override=knobs["bn_epsilon_override"]).build_model(1)
    except ValueError as e:
        raise LoaderError("%s: cannot build the level-%d model: %s" % (cfg_path, level, e))
    frozen = None
    if level == 2:
        frozen = CS.FrozenStage.from_config(model_cfg)
    elif level == 1:
        frozen = CS.FrozenChain.from_config(model_cfg)
    extra = {} if frozen is None else frozen.checkpoint_extra()
    _restore(prefix, model, extra)
    if frozen is not None:
        _check_fingerprints(level, config, frozen, cfg_path)
    try:
        step = int(tf.train.load_variable(prefix, C.STEP_KEY))
    except (tf.errors.NotFoundError, ValueError) as e:
        raise LoaderError("checkpoint %s holds no global step (%s)" % (prefix, e))
    sizes = {"group_size": args.get("group_size"), "deform_group_size": args.get("deform_group_size")}
    logger.info("loaded level-%d checkpoint %s (step %d), trainable %s, frozen levels %s",
                level, prefix, step, prec["trainable"], list(frozen_levels_of(level)))
    return LoadedModel(level, model, frozen, model_cfg=model_cfg, knobs=knobs, step=step,
                       prefix=prefix, run_dir=os.path.dirname(cfg_path), config=config,
                       training_group_sizes=sizes, precision=prec)
