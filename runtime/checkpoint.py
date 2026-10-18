"""Checkpoints ('last' and 'best') and the run config JSON.

One `tf.train.Checkpoint` holds the model, the (single, shared) optimizer, the global step, the
best validation value and extra trackables (the post-processing RNGs, so a resumed run continues
the same flip / pair sequence). Two `tf.train.CheckpointManager`s write it:
* ``ckpt/last``: every ``ckpt_every`` steps and at the end, keep 2; resume restores the latest.
* ``ckpt/best``: when the validation metric improves (lower is better), keep 1.

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
                 keep_last=2):
        self.step = step
        self.best_metric = tf.Variable(np.inf, dtype=tf.float64, trainable=False, name="best_metric")
        objs = dict(model=model, optimizer=optimizer, step=step, best_metric=self.best_metric)
        for k, v in (extra or {}).items():
            if k in objs:
                raise ValueError("extra key %r collides with a checkpoint field" % k)
            objs[k] = v
        self.checkpoint = tf.train.Checkpoint(**objs)
        self.last = tf.train.CheckpointManager(self.checkpoint, ckpt_dir_last, max_to_keep=int(keep_last),
                                               checkpoint_name="last")
        self.best = tf.train.CheckpointManager(self.checkpoint, ckpt_dir_best, max_to_keep=1,
                                               checkpoint_name="best")

    @property
    def latest_last(self):
        return self.last.latest_checkpoint

    @property
    def latest_best(self):
        return self.best.latest_checkpoint

    def save_last(self):
        return self.last.save(checkpoint_number=int(self.step.numpy()))

    def maybe_save_best(self, value):
        """Save to 'best' if `value` is finite and lower than the best so far. Returns True if saved."""
        value = float(value)
        if not np.isfinite(value) or value >= float(self.best_metric.numpy()):
            return False
        self.best_metric.assign(value)
        self.best.save(checkpoint_number=int(self.step.numpy()))
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
