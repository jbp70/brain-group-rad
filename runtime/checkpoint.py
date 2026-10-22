"""Checkpoints ('last' and 'best') and the run config JSON.

One `tf.train.Checkpoint` holds the model, the (single, shared) optimizer, the global step, the
best validation value and extra trackables (the post-processing RNGs, so a resumed run continues
the same flip / pair sequence). Two `tf.train.CheckpointManager`s write it:
* ``ckpt/last``: every ``ckpt_every`` steps and at the end, keep 2; resume restores the latest.
* ``ckpt/best``: when the validation metric improves (lower is better), keep 1.
* optional extra best files (``extra_best={name: dir}``, milestone 6: ``ckpt/best_deform`` in
  runs that train deform with a linear task): each has its own ``best_metric_<name>`` variable
  in the checkpoint and its own manager (``checkpoint_name='best_<name>'``, keep 1), saved by
  ``maybe_save_best(value, name)``. Without extra_best the checkpoint holds exactly the
  milestone-5 objects, so old checkpoints restore unchanged.

Build the model and the optimizer (on all trainable variables) before restoring, so every
variable exists and restores eagerly (`assert_existing_objects_matched`).

Warm start (--init-weights, `warm_start_model`; milestone 7): model variables only. Same
ModelConfig: strict whole-model restore. Configs differing only in the deform-decoder fields
(`DECODER_ONLY_FIELDS`): the encoder, linear decoder + heads and centering are restored through
a temporary model built with the source config, the deform decoder stays freshly initialised.
Milestone 8: the source level is read from the source config.json ``args.level`` (absent = the
model's level). A source of another level (e.g. a level-4 run into a level-2 model, the no-cascade
baseline) is restored into a temporary model of the SOURCE level and copied variable by variable
(`GroupRegLevelModel.copy_weights_from`, or the shared parts for a decoder-only difference);
those records add ``source_level``, ``source_step`` and ``cross_level``. Same-level records are
unchanged.

Level cascade (--cascade-from, milestone 8, design §2.2 / §9): `load_cascade_source` resolves and
checks the level-4 source, `check_cascade_config` requires identical ModelConfigs (seed aside),
`cascade_init_record` is the run's ``init_weights`` record and `weight_fingerprint` identifies
the frozen stage's weights in config.json (checked after every resume).
"""
import json
import os
import platform
import subprocess
import sys

import numpy as np
import tensorflow as tf

from models.config import ModelConfig


class RunCheckpoint:
    def __init__(self, ckpt_dir_last, ckpt_dir_best, model, optimizer, step, extra=None,
                 keep_last=2, extra_best=None):
        self.step = step
        self.best_metric = tf.Variable(np.inf, dtype=tf.float64, trainable=False, name="best_metric")
        objs = dict(model=model, optimizer=optimizer, step=step, best_metric=self.best_metric)
        self.extra_best_metric = {}
        for name in (extra_best or {}):
            key = "best_metric_%s" % name
            self.extra_best_metric[name] = tf.Variable(np.inf, dtype=tf.float64, trainable=False, name=key)
            objs[key] = self.extra_best_metric[name]
        for k, v in (extra or {}).items():
            if k in objs:
                raise ValueError("extra key %r collides with a checkpoint field" % k)
            objs[k] = v
        self.checkpoint = tf.train.Checkpoint(**objs)
        self.last = tf.train.CheckpointManager(self.checkpoint, ckpt_dir_last, max_to_keep=int(keep_last),
                                               checkpoint_name="last")
        self.best = tf.train.CheckpointManager(self.checkpoint, ckpt_dir_best, max_to_keep=1,
                                               checkpoint_name="best")
        self.extra_best = {name: tf.train.CheckpointManager(self.checkpoint, d, max_to_keep=1,
                                                            checkpoint_name="best_%s" % name)
                           for name, d in (extra_best or {}).items()}

    @property
    def latest_last(self):
        return self.last.latest_checkpoint

    @property
    def latest_best(self):
        return self.best.latest_checkpoint

    def save_last(self):
        return self.last.save(checkpoint_number=int(self.step.numpy()))

    def latest_best_named(self, name):
        """Latest checkpoint of the extra best file `name` (None if never saved)."""
        return self.extra_best[name].latest_checkpoint

    def maybe_save_best(self, value, name=None):
        """Save to 'best' (or the extra best file `name`) if `value` is finite and lower than that
        file's best so far. Returns True if saved."""
        var = self.best_metric if name is None else self.extra_best_metric[name]
        manager = self.best if name is None else self.extra_best[name]
        value = float(value)
        if not np.isfinite(value) or value >= float(var.numpy()):
            return False
        var.assign(value)
        manager.save(checkpoint_number=int(self.step.numpy()))
        return True

    def _restore(self, path):
        if not path:
            return False
        self.checkpoint.restore(path).assert_existing_objects_matched()
        return True

    def restore_last(self):
        """Restore the latest 'last' checkpoint. Returns True if one existed."""
        return self._restore(self.latest_last)

    def restore_best(self):
        return self._restore(self.latest_best)


STEP_KEY = "step/.ATTRIBUTES/VARIABLE_VALUE"


def peek_last_step(ckpt_dir_last):
    """Global step stored in the latest 'last' checkpoint of `ckpt_dir_last`, read from the
    checkpoint file without building the model (0 if there is none). main.py uses it to write
    the resumed step's control values before the DALI pipelines start."""
    path = tf.train.latest_checkpoint(ckpt_dir_last) if os.path.isdir(ckpt_dir_last) else None
    if not path:
        return 0
    return int(tf.train.load_variable(path, STEP_KEY))


def resolve_checkpoint(path):
    """A checkpoint prefix (``.../ckpt/best/best-3``) or a directory holding checkpoints (a run's
    ``ckpt/best`` or ``ckpt/last``: its latest one) -> the checkpoint prefix. ValueError if none."""
    path = str(path)
    if os.path.isdir(path):
        found = tf.train.latest_checkpoint(path)
        if not found:
            raise ValueError("no checkpoint in directory %s" % path)
        return found
    if os.path.exists(path + ".index"):
        return path
    raise ValueError("--init-weights %s is neither a checkpoint prefix (no %s.index) nor a "
                     "checkpoint directory" % (path, path))


def source_config_path(ckpt_prefix):
    """config.json of the run that wrote `ckpt_prefix` (<run>/ckpt/<kind>/<name>), or None."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ckpt_prefix))))
    path = os.path.join(root, "config.json")
    return path if os.path.exists(path) else None


# ModelConfig fields that change nothing outside the deform decoder / SVF integrator
# (models/group_net.py: DeformDecoder and GroupRegLevelModel._call_deform). A warm start whose
# configs differ only in these restores the shared parts and keeps the deform decoder fresh.
DECODER_ONLY_FIELDS = ("deform_decoder", "decoder_widths", "integration_steps")
# Model parts restored by a partial warm start (GroupRegLevelModel attributes).
SHARED_PARTS = ("encoder", "linear_decoder", "centering")
DEFORM_PART = "deform_decoder"


def _strict_restore(model, prefix):
    status = tf.train.Checkpoint(model=model).restore(prefix)
    status.expect_partial()                       # the optimizer, step, ... are not loaded
    status.assert_existing_objects_matched()


def _checkpoint_step(prefix):
    """Global step stored in the checkpoint `prefix` (ValueError if it has none)."""
    try:
        return int(tf.train.load_variable(prefix, STEP_KEY))
    except (tf.errors.NotFoundError, ValueError) as e:
        raise ValueError("checkpoint %s holds no global step (%s)" % (prefix, e))


CASCADE_SOURCE_LEVEL = 4


def load_cascade_source(path):
    """--cascade-from: (prefix, src_cfg ModelConfig, info dict).

    `path` is a checkpoint prefix or a checkpoint directory (`resolve_checkpoint`). Raises
    ValueError when the path does not resolve, the source run's config.json is missing
    (`source_config_path`), or its ``args.level`` is not 4. info: ``checkpoint`` (absolute
    prefix), ``source_config`` (absolute path), ``source_level`` 4, ``source_step`` (the global
    step stored in the checkpoint)."""
    try:
        prefix = resolve_checkpoint(path)
    except ValueError as e:
        raise ValueError("--cascade-from %s: %s" % (path, e))
    prefix = os.path.abspath(prefix)
    cfg_path = source_config_path(prefix)
    if cfg_path is None:
        raise ValueError("--cascade-from %s: no config.json of the source run next to the checkpoint "
                         "(expected <run>/config.json for <run>/ckpt/<kind>/<name>)" % path)
    with open(cfg_path) as f:
        saved = json.load(f)
    level = (saved.get("args") or {}).get("level")
    if level is None or int(level) != CASCADE_SOURCE_LEVEL:
        raise ValueError("--cascade-from %s: the source run %s is level %s; the cascade needs a level-%d "
                         "source" % (path, cfg_path, level, CASCADE_SOURCE_LEVEL))
    src_cfg = ModelConfig.from_dict(saved["model_config"])
    info = {"checkpoint": prefix, "source_config": os.path.abspath(cfg_path),
            "source_level": CASCADE_SOURCE_LEVEL, "source_step": _checkpoint_step(prefix)}
    return prefix, src_cfg, info


def check_cascade_config(model_config, src_cfg):
    """ValueError naming every ModelConfig field (seed excepted) in which this run's config
    differs from the cascade source's; no field may differ (`DECODER_ONLY_FIELDS` included: the
    frozen stage and the copied trainable stage need the source's deform decoder)."""
    mine = {k: v for k, v in model_config.to_dict().items() if k != "seed"}
    theirs = {k: v for k, v in src_cfg.to_dict().items() if k != "seed"}
    diff = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
    if diff:
        raise ValueError("--cascade-from: model config differs from the source run in %s (%s); the "
                         "cascade needs identical model flags"
                         % (diff, ", ".join("%s: source %r, this run %r" % (k, theirs.get(k), mine.get(k))
                                            for k in diff)))


def weight_fingerprint(model):
    """{'n_vars': int, 'abs_sum': float64 sum of |w|, 'sq_sum': float64 sum of w^2} over
    ``model.weights`` (in variable order), e.g. to identify the frozen stage after a resume."""
    abs_sum = 0.0
    sq_sum = 0.0
    for v in model.weights:
        w = np.asarray(v.numpy(), np.float64)
        abs_sum += float(np.abs(w).sum())
        sq_sum += float(np.square(w).sum())
    return {"n_vars": len(model.weights), "abs_sum": abs_sum, "sq_sum": sq_sum}


def _part_counts(model):
    parts = SHARED_PARTS + ((DEFORM_PART,) if getattr(model, DEFORM_PART, None) is not None else ())
    return {p: len(getattr(model, p).weights) for p in parts}


def cascade_init_record(model, info):
    """The ``init_weights`` record of a cascade run (design §2.2): `model` is the trainable
    (level-2) model after ``copy_weights_from(frozen.model)``, `info` the dict of
    `load_cascade_source`."""
    return {"mode": "full", "via": "cascade", "checkpoint": info["checkpoint"],
            "source_config": info["source_config"], "source_level": int(info["source_level"]),
            "source_step": int(info["source_step"]), "cross_level": [int(info["source_level"]), model.level],
            "restored_parts": _part_counts(model), "fresh_parts": {}, "ignored_source_parts": [],
            "n_restored_vars": len(model.weights), "n_fresh_vars": 0, "config_diff": {}}


def warm_start_model(model, model_config, path):
    """Warm start: restore ONLY model variables of another run's checkpoint into `model` (built).
    Optimizer, step, best value and RNG states are never read (the caller builds a fresh
    optimizer).

    * No source config.json next to the checkpoint, or the same ModelConfig (init seed excluded):
      strict restore of the whole model (every model variable must be in the checkpoint),
      ``mode='full'``.
    * Configs differing only in `DECODER_ONLY_FIELDS` (e.g. a linear-only source into a model
      with the deform decoder, or the reverse): ``mode='partial'``. A model with the source
      config is built and restored strictly, then the `SHARED_PARTS` (encoder, linear decoder +
      heads, centering; trainable variables and BN / LayerNorm state) are copied after checking
      count, shape and scope-local name of every variable. The target's deform decoder keeps
      its fresh initialisation; a source deform decoder is ignored.
    * Any other difference: ValueError naming the fields.
    * Source of another level (config.json ``args.level`` != ``model.level``; absent = same
      level; milestone 8, design §2.3): the source model is built at the SOURCE level and
      restored strictly, then copied: identical configs -> `copy_weights_from` (``mode='full'``),
      decoder-only difference -> the shared-parts copy above (``mode='partial'``). These records
      add exactly ``source_level``, ``source_step`` (the checkpoint's global step) and
      ``cross_level`` = [source level, model level]; same-level records are unchanged.

    Returns an info dict: checkpoint, source_config, mode, config_diff {field: [source, target]},
    restored_parts {part: n_vars}, fresh_parts {part: n_vars}, ignored_source_parts [part],
    n_restored_vars, n_fresh_vars (plus the three cross-level keys)."""
    prefix = resolve_checkpoint(path)
    cfg_path = source_config_path(prefix)
    info = {"checkpoint": prefix, "source_config": cfg_path, "mode": "full", "config_diff": {},
            "restored_parts": {}, "fresh_parts": {}, "ignored_source_parts": [],
            "n_restored_vars": 0, "n_fresh_vars": 0}
    src_cfg = None
    src_level = model.level
    diff = []
    if cfg_path is not None:
        with open(cfg_path) as f:
            saved = json.load(f)
        src_cfg = ModelConfig.from_dict(saved["model_config"])
        saved_level = (saved.get("args") or {}).get("level")
        if saved_level is not None:
            src_level = int(saved_level)
        theirs = {k: v for k, v in src_cfg.to_dict().items() if k != "seed"}
        mine = {k: v for k, v in model_config.to_dict().items() if k != "seed"}
        diff = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
        info["config_diff"] = {k: [theirs.get(k), mine.get(k)] for k in diff}
        other = [k for k in diff if k not in DECODER_ONLY_FIELDS]
        if other:
            raise ValueError("--init-weights: model config differs from the source run %s in %s "
                             "(only %s may differ); build the model with the same flags"
                             % (cfg_path, other, list(DECODER_ONLY_FIELDS)))
    cross = src_level != model.level
    if not diff and not cross:
        _strict_restore(model, prefix)
        parts = SHARED_PARTS + ((DEFORM_PART,) if getattr(model, DEFORM_PART, None) is not None else ())
        info["restored_parts"] = {p: len(getattr(model, p).weights) for p in parts}
        info["n_restored_vars"] = len(model.weights)
        return info

    from models.group_net import GroupRegLevelModel   # local import: no module-level cycle
    if cross:
        src_step = _checkpoint_step(prefix)
    if not diff:                                      # cross-level, identical configs
        src = GroupRegLevelModel(src_level, src_cfg).build_model(2)
        try:
            _strict_restore(src, prefix)
            model.copy_weights_from(src)
        finally:
            del src
        info["restored_parts"] = _part_counts(model)
        info["n_restored_vars"] = len(model.weights)
        info.update(source_level=src_level, source_step=src_step, cross_level=[src_level, model.level])
        return info

    info["mode"] = "partial"
    src = GroupRegLevelModel(src_level, src_cfg).build_model(2)
    try:
        _strict_restore(src, prefix)
        pairs = []
        for part in SHARED_PARTS:
            s_w, d_w = getattr(src, part).weights, getattr(model, part).weights
            if len(s_w) != len(d_w):
                raise ValueError("--init-weights: %s variable count differs: %d (source) vs %d"
                                 % (part, len(s_w), len(d_w)))
            for s, d in zip(s_w, d_w):
                sn, dn = GroupRegLevelModel._local_name(s), GroupRegLevelModel._local_name(d)
                if tuple(s.shape) != tuple(d.shape) or sn != dn:
                    raise ValueError("--init-weights: %s variable mismatch: %s %s vs %s %s"
                                     % (part, sn, tuple(s.shape), dn, tuple(d.shape)))
                pairs.append((s, d))
            info["restored_parts"][part] = len(d_w)
        # Coverage: every target variable is either restored or in the (fresh) deform decoder.
        dfm = getattr(model, DEFORM_PART, None)
        fresh = list(dfm.weights) if dfm is not None else []
        covered = {d.ref() for _, d in pairs} | {v.ref() for v in fresh}
        unhandled = [v.name for v in model.weights if v.ref() not in covered]
        if unhandled or len(covered) != len(model.weights):
            raise RuntimeError("--init-weights: unhandled model part variables %s" % unhandled[:5])
        for s, d in pairs:
            d.assign(s)
        if dfm is not None:
            info["fresh_parts"][DEFORM_PART] = len(fresh)
        if getattr(src, DEFORM_PART, None) is not None:
            info["ignored_source_parts"].append(DEFORM_PART)
        info["n_restored_vars"] = len(pairs)
        info["n_fresh_vars"] = len(fresh)
    finally:
        del src
    if cross:
        info.update(source_level=src_level, source_step=src_step, cross_level=[src_level, model.level])
    return info


def load_model_weights(model, model_config, path):
    """`warm_start_model` returning (checkpoint prefix, source config path or None)."""
    info = warm_start_model(model, model_config, path)
    return info["checkpoint"], info["source_config"]


def _git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, timeout=10, check=False)
        return out.stdout.decode().strip() or None
    except Exception:  # noqa: BLE001  (no git, no repo, timeout)
        return None


def write_config(path, model_config, args_dict, extra=None):
    """Write config.json: model config, run args, versions and the git commit (if any)."""
    cfg = {"model_config": model_config.to_dict(), "args": args_dict,
           "versions": {"python": sys.version.split()[0], "tensorflow": tf.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
           "git_commit": _git_commit()}
    if extra:
        cfg.update(extra)
    from runtime.logging import to_jsonable
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(to_jsonable(cfg), f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return cfg


def check_config_compatible(path, model_config):
    """Raise ValueError naming the ModelConfig fields that differ from the saved config.json."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        saved = json.load(f)
    old = ModelConfig.from_dict(saved["model_config"]).to_dict()
    new = model_config.to_dict()
    diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    if diff:
        raise ValueError("model config differs from %s in %s; use a new --run-name or matching flags"
                         % (path, diff))
