"""Two-stage level cascade (milestone 8, `.agents/tasks/milestone-8/design.md` §3-§6).

Level 2 is trained as the second stage of a cascade whose first stage is a FROZEN level-4
`GroupRegLevelModel` (`FrozenStage`). Every step pools the (flipped, augmented) level-2 group to
level 4, runs the frozen stage on it, pre-warps the level-2 images with the frozen estimate and
lets the trainable level-2 stage predict a residual that is composed with the frozen estimate.

Conventions (binding; design 2 / 4.6 / 4.7 of `.agents/tasks/network-design/design.md`):

| Item | Convention |
|---|---|
| Levels | `PREV_LEVEL` = {2: 4}: milestone 8 only cascades level 4 -> level 2 (level 1 = milestone 9). |
| Pooling | `pool_to_coarser`: 2x2x2 average pooling ('VALID', float32). Level-2 voxel centers (p - 39.5) * 2 mm; the block {2q, 2q+1} averages to (q - 19.5) * 4 mm = the level-4 center, so block averaging preserves world points. All level-2 axis lengths are even, so a flip commutes with the pooling. |
| Linear | A_prev_i (image i -> level-4 central frame, centered mm) from the frozen stage; pre-warp x_i = I_i o A_prev_i^-1 (`prewarp_linear`, pull with A^-1, zero fill); composed A_i = Center(R_raw_i A_prev_i) with ONE GroupCentering (`compose_linear`, the trainable model's `centering`). |
| Deform | (U, U_inv, V) = `diffeo.upsample_displacement` of the frozen (u, u_inv, v) (level-2 voxels); pre-warp x_i = warp(I_i, U_i) (`prewarp_deform`); composed (u, u_inv) = `diffeo.compose_cascade(U, U_inv, u_res, u_inv_res)`. |
| Skip | `skip_linear` / `skip_fields` (bool scalar tensor): a skipped stage passes its input on (A_prev = I, U = U_inv = 0). Their `tf.where` ops live under the name scope ``cascade_skip`` (tests check that p_skip = 0 graphs contain none). |
| Float32 | Every `FrozenStage` model is constructed AND built inside `_float32_policy()` (Keras layers capture the dtype policy in ``__init__``), with ``gradient_checkpointing='none'``, so the frozen outputs never depend on --amp. |
| Inference only | `FrozenStage.linear` / `.deform` always call the model with ``training=False`` (BN moving statistics = the level-4 values, never recalibrated) and wrap the outputs in `tf.stop_gradient`; callers run them outside the GradientTape and never give the frozen variables to the optimizer. |
"""
import contextlib

import tensorflow as tf

from models import diffeo
from models import layers as L
from models import transforms as tfm
from models.group_net import DEFORM_TASK, GroupRegLevelModel

PREV_LEVEL = {2: 4}                       # milestone 8: level 2 only (level 1 = milestone 9)
FROZEN_LEVEL = PREV_LEVEL[2]
SKIP_SCOPE = "cascade_skip"


def pool_to_coarser(images):
    """[N, X, Y, Z, C] -> float32 [N, X/2, Y/2, Z/2, C] by 2x2x2 block averaging (world points kept)."""
    return tf.nn.avg_pool3d(tf.cast(images, tf.float32), 2, 2, "VALID")


@contextlib.contextmanager
def _float32_policy():
    """Temporarily set the global Keras policy to 'float32' (restored in ``finally``)."""
    mp = tf.keras.mixed_precision
    saved = mp.global_policy()
    mp.set_global_policy("float32")
    try:
        yield
    finally:
        mp.set_global_policy(saved)


def _build_frozen_model(cfg):
    with _float32_policy():
        return GroupRegLevelModel(FROZEN_LEVEL, cfg, gradient_checkpointing="none").build_model(2)


class FrozenStage:
    """The previous cascade level, inference only.

    Attributes: ``model`` = `GroupRegLevelModel(4, cfg)` (float32 policy, gradient checkpointing
    'none'); ``level`` = 4; ``fingerprint`` = `runtime.checkpoint.weight_fingerprint(model)`.
    Build it with one of the factories:
    * `from_checkpoint(prefix, cfg)`: a fresh run's level-4 source (strict model restore);
    * `from_model(model)`: a float32 copy of another model's weights (tests, memory script);
    * `from_config(cfg)`: random init, restored afterwards by `RunCheckpoint` (resume).
    """

    def __init__(self, model):
        if model.level != FROZEN_LEVEL:
            raise ValueError("the frozen stage is a level-%d model, got level %d" % (FROZEN_LEVEL, model.level))
        self.model = model
        self.level = FROZEN_LEVEL

    @classmethod
    def from_checkpoint(cls, prefix, cfg):
        """Level-4 model of `cfg`, restored from the checkpoint `prefix` (model variables only:
        `runtime.checkpoint._strict_restore`, every model variable must be in the checkpoint)."""
        from runtime import checkpoint as C        # local import: models/ stays runtime-free at import
        model = _build_frozen_model(cfg)
        C._strict_restore(model, prefix)
        return cls(model)

    @classmethod
    def from_model(cls, model):
        """Fresh float32 level-4 model of ``model.config`` with ``model``'s weights copied
        (`copy_weights_from`; `model` may be any level and may have been built under another
        policy: it is never wrapped)."""
        frozen = _build_frozen_model(model.config)
        frozen.copy_weights_from(model)
        return cls(frozen)

    @classmethod
    def from_config(cls, cfg):
        """Randomly initialised level-4 model of `cfg` (its weights come from the run checkpoint)."""
        return cls(_build_frozen_model(cfg))

    @property
    def fingerprint(self):
        from runtime import checkpoint as C
        return C.weight_fingerprint(self.model)

    def linear(self, images_fine, task):
        """A_prev [N, 4, 4] float32 (image i -> level-4 central frame), no gradient."""
        out = self.model(pool_to_coarser(images_fine), task=task, training=False)
        return tf.stop_gradient(tf.cast(out["A"], tf.float32))

    def deform(self, images_fine):
        """dict U, U_inv, V: the frozen u, u_inv, v upsampled to the fine level (level-2 voxels),
        no gradient."""
        out = self.model(pool_to_coarser(images_fine), task=DEFORM_TASK, training=False)
        up = diffeo.upsample_displacement
        return {k: tf.stop_gradient(up(out[s])) for k, s in (("U", "u"), ("U_inv", "u_inv"), ("V", "v"))}


def prewarp_linear(images, A_prev, spacing):
    """I_i o A_prev_i^-1 (pull warp into the previous stage's central frame, zero fill)."""
    return L.transform_image(images, tfm.affine_inverse(A_prev), spacing)


def compose_linear(R_raw, A_prev, centering):
    """Center(R_raw_i @ A_prev_i), in the dtype of A_prev.

    Train / validation steps: A_prev float32 (`FrozenStage.linear`), centering = the trainable
    model's `centering` (float32). float64 work: float64 inputs and
    ``tfm.GroupCentering(iters=cfg.centering_iters, dtype='float64')``."""
    return centering(tf.matmul(tf.cast(R_raw, A_prev.dtype), A_prev))


def prewarp_deform(images, U):
    """warp(I_i, U_i) (linear, zero fill), float32."""
    return diffeo.warp(tf.cast(images, tf.float32), U)


def skip_linear(skip, A_prev):
    """A_prev, or the identity for every image when `skip` (bool scalar tensor) is true."""
    with tf.name_scope(SKIP_SCOPE):
        eye = tf.zeros_like(A_prev) + tf.eye(4, dtype=A_prev.dtype)
        return tf.where(skip, eye, A_prev)


def skip_fields(skip, fields):
    """[each field, or zeros when `skip` (bool scalar tensor) is true] for a list of fields."""
    with tf.name_scope(SKIP_SCOPE):
        return [tf.where(skip, tf.zeros_like(f), f) for f in fields]
