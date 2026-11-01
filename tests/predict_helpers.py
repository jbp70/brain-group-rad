"""Run directories for the prediction tests (predict-script design 10.1; not a test module).

Each helper writes what main.py writes for a run: `runtime.logging.make_run_dir` paths, a
`RunCheckpoint` 'last' checkpoint (model, optimizer, step, plus the frozen stages under their
milestone-8 / milestone-9 keys) and config.json via `runtime.checkpoint.write_config` with the
args / cascade record keys of main.py (322-354). Import `tests.tf_setup` before this module.
"""
from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory limit)
from tests.test_cascade import randomize
import tensorflow as tf

from models.group_net import GroupRegLevelModel
from runtime import checkpoint as C
from runtime import optimizers as O
from runtime.logging import make_run_dir


def default_args(level, amp=False, **kw):
    """config.json ``args`` subset the loader reads (main.py `A.args_to_dict` keys)."""
    d = {"level": int(level), "amp": bool(amp), "split_group_conv": True, "lean_sampler": True,
         "deform_decoder_dtype": "policy", "fp16_act_clip": 0.0,
         "fp16_act_clip_scope": "deform_decoder", "bn_epsilon_override": 0.0,
         "lowres_method": "nn", "group_size": 4, "deform_group_size": 2}
    d.update(kw)
    return d


def _save(root, name, model, args, extra=None, cascade=None, step=7):
    paths = make_run_dir(root, name)
    opt = O.make_optimizer("adam", lr=1e-3)
    O.build_optimizer(opt, model.trainable_variables)
    st = tf.Variable(int(step), dtype=tf.int64)
    ck = C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], model, opt, st, extra=extra)
    prefix = ck.save_last()
    C.write_config(paths["config"], model.config, args,
                   extra=None if cascade is None else {"cascade": cascade})
    return prefix


def save_level4_run(root, name, cfg, seed, amp=False, extra_args=None, step=7):
    """A randomised float32 level-4 model of `cfg` saved as <root>/<name> (args.amp = `amp`; the
    variables are float32 under every policy). Returns (checkpoint prefix, model)."""
    model = GroupRegLevelModel(4, cfg).build_model(2)
    randomize(model, seed, 0.02)
    args = default_args(4, amp=amp, **dict(extra_args or {}))
    return _save(root, name, model, args, step=step), model


def save_cascade_run(root, name, m2, frozen_stage, args=None, step=5):
    """Level-2 cascade run: model = `m2`, key frozen_stage = `frozen_stage.model`, config.json
    cascade record with the frozen level-4 fingerprint. Returns the checkpoint prefix."""
    cascade = {"source_checkpoint": None, "source_config": None, "source_level": 4,
               "source_step": 0, "fingerprint": frozen_stage.fingerprint,
               "S_rel_diff_vs_level4": None, "skip_prob": 0.0}
    return _save(root, name, m2, default_args(2, **dict(args or {})),
                 extra=frozen_stage.checkpoint_extra(), cascade=cascade, step=step)


def save_chain_run(root, name, m1, chain, args=None, step=5):
    """Level-1 cascade run: model = `m1`, keys frozen_stage_4 / frozen_stage_2 = the chain's
    models, config.json cascade record with ``stages`` (levels 4 and 2). Returns the prefix."""
    fps = chain.fingerprints
    cascade = {"source_checkpoint": None, "source_config": None, "source_level": 2,
               "source_step": 0, "fingerprint": chain.fingerprint,
               "S_rel_diff_vs_level4": None, "S_rel_diff_vs_level2": None, "skip_prob": 0.0,
               "skip_stage": 2,
               "stages": [{"level": 4, "checkpoint_key": "frozen_stage", "fingerprint": fps[4],
                           "origin_checkpoint": None, "origin_step": None},
                          {"level": 2, "checkpoint_key": "model", "fingerprint": fps[2],
                           "origin_checkpoint": None, "origin_step": 0}]}
    return _save(root, name, m1, default_args(1, **dict(args or {})),
                 extra=chain.checkpoint_extra(), cascade=cascade, step=step)
