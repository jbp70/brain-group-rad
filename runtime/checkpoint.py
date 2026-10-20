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


def load_model_weights(model, model_config, path):
    """Warm start: restore ONLY the model variables of another run's checkpoint into `model`
    (built, same ModelConfig). Optimizer, step, best value and RNG states are not read.

    The source run's config.json (if present) must have the same ModelConfig except the init
    seed, as in `GroupRegLevelModel.copy_weights_from`; otherwise ValueError names the fields.
    Every model variable must be matched in the checkpoint. Returns (checkpoint prefix,
    source config path or None)."""
    prefix = resolve_checkpoint(path)
    cfg_path = source_config_path(prefix)
    if cfg_path is not None:
        with open(cfg_path) as f:
            saved = json.load(f)
        theirs = {k: v for k, v in ModelConfig.from_dict(saved["model_config"]).to_dict().items()
                  if k != "seed"}
        mine = {k: v for k, v in model_config.to_dict().items() if k != "seed"}
        diff = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
        if diff:
            raise ValueError("--init-weights: model config differs from the source run %s in %s; "
                             "build the model with the same flags" % (cfg_path, diff))
    status = tf.train.Checkpoint(model=model).restore(prefix)
    status.expect_partial()                       # the optimizer, step, ... are not loaded
    status.assert_existing_objects_matched()
    return prefix, cfg_path


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
