"""Frozen-chain stand-in for level-1 memory measurements (milestone 9 phase 1, R4.2).

Design: `.agents/tasks/milestone-9/design-m9-phase1.md` section 5.2. The real implementation is
`models.cascade.FrozenChain` (milestone 9 phase 2, `.agents/tasks/milestone-9-phase2/
design-m9-phase2.md` section 2); this stand-in is kept for
`tests/gpu/measure_level_memory.py --level 1 --cascade true` (records schema-compatible with
job 10516237; since phase 2 ``val_peak_mb`` includes the ``_frozen4`` fields U4 / U4_inv / V4,
~+1 GB at level-1 N 4, so it is not like-for-like with that job) and for test T-C2, which asserts both give identical outputs. ``n_stages`` = 2, so the
Trainer runs its chain branches with it (``linear_stages`` / ``deform_stages``: the same
formulas as FrozenChain). It lives in tests/gpu (not models/) and is imported only by the
measurement tool and tests, which call `configure_tf_precision` first.

* `FrozenChainStandIn(stage4, model2)`: ``level`` = 2 (the trainable level-1 stage's previous
  level); ``stage4`` = a `models.cascade.FrozenStage` of a random level-4 model; ``model2`` = a
  float32 level-2 `GroupRegLevelModel` (gradient checkpointing 'none') built under
  `models.cascade._float32_policy()`. Same interface as `FrozenStage`: ``linear(images, task)``
  -> A_prev [N, 4, 4] and ``deform(images)`` -> dict U, U_inv, V at level 1, all float32 and
  stop_gradient'ed; inference only (``training=False``). Weights are random (memory and time).
* `build_standin(model_config, build_n=2)`: both models built at ``build_n`` (the R1.3 rule:
  every model builds at min(N, --build-n)).
* `patched_prev_level()`: context manager setting ``models.cascade.PREV_LEVEL = {2: 4, 1: 2}``
  (so `Trainer(cascade=standin)` accepts a level-1 model) and restoring the original dict in
  ``finally``. Since phase 2 the module constant is already that dict, so this is a no-op kept
  for the measurement tool and old scripts. ``FROZEN_LEVEL`` is a literal 4, unaffected.
"""
import contextlib

import numpy as np
import tensorflow as tf

from data_loading import params as P
from models import cascade as CS
from models import diffeo
from models.group_net import DEFORM_TASK, GroupRegLevelModel


class FrozenChainStandIn(object):
    """Frozen level 4 -> level 2 chain in front of a trainable level-1 stage (random weights)."""

    level = 2
    n_stages = 2                     # Trainer dispatch: the chain branches (phase 2, design 2.1)

    def __init__(self, stage4, model2):
        if stage4.level != CS.FROZEN_LEVEL:
            raise ValueError("stage4 must be a level-%d FrozenStage" % CS.FROZEN_LEVEL)
        if model2.level != 2:
            raise ValueError("model2 must be a level-2 model, got level %d" % model2.level)
        self.stage4 = stage4
        self.model2 = model2

    @property
    def models(self):
        return [self.stage4.model, self.model2]

    @property
    def n_params(self):
        """Parameter count of both frozen models (all weights)."""
        return int(sum(int(np.prod(v.shape)) for m in self.models for v in m.weights))

    def deform(self, images_l1):
        """dict U, U_inv, V (level-1 voxels): composed level-4 + level-2 fields, upsampled."""
        x2 = CS.pool_to_coarser(images_l1)
        p4 = self.stage4.deform(x2)                       # level-4 fields upsampled to level 2
        out = self.model2(CS.prewarp_deform(x2, p4["U"]), task=DEFORM_TASK, training=False)
        u2, u2i = diffeo.compose_cascade(p4["U"], p4["U_inv"], out["u"], out["u_inv"])
        up = diffeo.upsample_displacement
        return {"U": tf.stop_gradient(up(u2)), "U_inv": tf.stop_gradient(up(u2i)),
                "V": tf.stop_gradient(up(out["v"]))}

    def linear(self, images_l1, task):
        """A_prev [N, 4, 4] float32 (image i -> level-2 central frame)."""
        x2 = CS.pool_to_coarser(images_l1)
        A4 = self.stage4.linear(x2, task)
        out = self.model2(CS.prewarp_linear(x2, A4, P.LEVEL_SPACING[2]), task=task, training=False)
        return tf.stop_gradient(CS.compose_linear(out["A_raw"], A4, self.model2.centering))

    def linear_stages(self, images_l1, task):
        """{'A': `linear` result, 'A4': the level-4 stage alone} (same formulas as FrozenChain)."""
        x2 = CS.pool_to_coarser(images_l1)
        A4 = self.stage4.linear(x2, task)
        out = self.model2(CS.prewarp_linear(x2, A4, P.LEVEL_SPACING[2]), task=task, training=False)
        return {"A": tf.stop_gradient(CS.compose_linear(out["A_raw"], A4, self.model2.centering)),
                "A4": tf.stop_gradient(A4)}

    def deform_stages(self, images_l1):
        """`deform` plus U4, U4_inv, V4 (the level-4 stage alone, upsampled to level 1)."""
        x2 = CS.pool_to_coarser(images_l1)
        p4 = self.stage4.deform(x2)
        out = self.model2(CS.prewarp_deform(x2, p4["U"]), task=DEFORM_TASK, training=False)
        u2, u2i = diffeo.compose_cascade(p4["U"], p4["U_inv"], out["u"], out["u_inv"])
        up = diffeo.upsample_displacement
        return {"U": tf.stop_gradient(up(u2)), "U_inv": tf.stop_gradient(up(u2i)),
                "V": tf.stop_gradient(up(out["v"])), "U4": tf.stop_gradient(up(p4["U"])),
                "U4_inv": tf.stop_gradient(up(p4["U_inv"])), "V4": tf.stop_gradient(up(p4["V"]))}


def build_standin(model_config, build_n=2):
    """Stand-in with random float32 level-4 and level-2 models of ``model_config``, both built
    with `build_model(build_n)`."""
    build_n = int(build_n)
    if build_n < 1:
        raise ValueError("build_n must be >= 1")
    with CS._float32_policy():
        m4 = GroupRegLevelModel(CS.FROZEN_LEVEL, model_config, gradient_checkpointing="none").build_model(build_n)
        m2 = GroupRegLevelModel(2, model_config, gradient_checkpointing="none").build_model(build_n)
    return FrozenChainStandIn(CS.FrozenStage(m4), m2)


@contextlib.contextmanager
def patched_prev_level():
    """models.cascade.PREV_LEVEL = {2: 4, 1: 2} inside the block (restored in ``finally``)."""
    saved = CS.PREV_LEVEL
    CS.PREV_LEVEL = {2: 4, 1: 2}
    try:
        yield
    finally:
        CS.PREV_LEVEL = saved
