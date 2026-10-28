"""Level cascade (milestone 8, `.agents/tasks/milestone-8/design.md` §3-§6; 3-level chain:
milestone 9 phase 2, `.agents/tasks/milestone-9-phase2/design-m9-phase2.md` §2).

Level 2 is trained as the second stage of a cascade whose first stage is a FROZEN level-4
`GroupRegLevelModel` (`FrozenStage`). Every step pools the (flipped, augmented) level-2 group to
level 4, runs the frozen stage on it, pre-warps the level-2 images with the frozen estimate and
lets the trainable level-2 stage predict a residual that is composed with the frozen estimate.

Conventions (binding; design 2 / 4.6 / 4.7 of `.agents/tasks/network-design/design.md`):

| Item | Convention |
|---|---|
| Levels | `PREV_LEVEL` = {2: 4, 1: 2}: level 2 follows a frozen level-4 `FrozenStage` (milestone 8); level 1 follows the frozen level 4 -> level 2 `FrozenChain` (milestone 9). `FROZEN_LEVEL` = 4 is the level of `FrozenStage`. |
| Chain | `FrozenChain(stage4, model2)` (``level`` 2, ``n_stages`` 2): x2 = pool(x1), A4 / U4 from the level-4 stage on x2, level-2 residual on the pre-warped x2, composed at level 2 (`compose_linear` with model2's centering; `diffeo.compose_cascade`), fields upsampled to level 1. Restored from ONE milestone-8 level-2 cascade checkpoint (keys ``model`` + ``frozen_stage``, level-4 fingerprint checked); run-checkpoint keys ``frozen_stage_4`` / ``frozen_stage_2``. `FrozenStage.n_stages` = 1 (Trainer dispatch). Skip at level 1 drops the last frozen stage (`skip_to`: A4 / U4 instead of the chain output). |
| Pooling | `pool_to_coarser`: 2x2x2 average pooling ('VALID', float32). Level-2 voxel centers (p - 39.5) * 2 mm; the block {2q, 2q+1} averages to (q - 19.5) * 4 mm = the level-4 center, so block averaging preserves world points. All level-2 axis lengths are even, so a flip commutes with the pooling. |
| Linear | A_prev_i (image i -> level-4 central frame, centered mm) from the frozen stage; pre-warp x_i = I_i o A_prev_i^-1 (`prewarp_linear`, pull with A^-1, zero fill); composed A_i = Center(R_raw_i A_prev_i) with ONE GroupCentering (`compose_linear`, the trainable model's `centering`). |
| Deform | (U, U_inv, V) = `diffeo.upsample_displacement` of the frozen (u, u_inv, v) (level-2 voxels); pre-warp x_i = warp(I_i, U_i) (`prewarp_deform`); composed (u, u_inv) = `diffeo.compose_cascade(U, U_inv, u_res, u_inv_res)`. |
| Skip | `skip_linear` / `skip_fields` (bool scalar tensor): a skipped stage passes its input on (A_prev = I, U = U_inv = 0). Their `tf.where` ops live under the name scope ``cascade_skip`` (tests check that p_skip = 0 graphs contain none). Per-stage skip at level 1 (retrain-prereqs design 4.4): `FrozenChain.linear_skip` / `deform_skip` (skip4: level 4 = identity, level 2 still runs; skip2: level 2 dropped; both: identity). |
| Float32 | Every `FrozenStage` model is constructed AND built inside `_float32_policy()` (Keras layers capture the dtype policy in ``__init__``), with ``gradient_checkpointing='none'``, so the frozen outputs never depend on --amp. |
| Inference only | `FrozenStage.linear` / `.deform` always call the model with ``training=False`` (BN moving statistics = the level-4 values, never recalibrated) and wrap the outputs in `tf.stop_gradient`; callers run them outside the GradientTape and never give the frozen variables to the optimizer. |
"""
import tensorflow as tf

from data_loading import params as P
from models import diffeo
from models import layers as L
from models import precision
from models import transforms as tfm
from models.group_net import DEFORM_TASK, GroupRegLevelModel

PREV_LEVEL = {2: 4, 1: 2}                 # trainable level -> previous (frozen) level
FROZEN_LEVEL = 4                          # level of `FrozenStage` (and of the chain's first stage)
CHAIN_LEVEL = 2                           # level of the chain's second frozen stage (= PREV_LEVEL[1])
SKIP_SCOPE = "cascade_skip"


def pool_to_coarser(images):
    """[N, X, Y, Z, C] -> float32 [N, X/2, Y/2, Z/2, C] by 2x2x2 block averaging (world points kept)."""
    return tf.nn.avg_pool3d(tf.cast(images, tf.float32), 2, 2, "VALID")


def _float32_policy():
    """Temporarily set the global Keras policy to 'float32' (restored in ``finally``)."""
    return precision.policy_scope("float32")


def _build_frozen_model(cfg, level=FROZEN_LEVEL):
    """Float32 `GroupRegLevelModel(level, cfg)` for a frozen stage (constructed AND built inside
    `_float32_policy()`, gradient checkpointing 'none', default sampler / group conv)."""
    with _float32_policy():
        return GroupRegLevelModel(level, cfg, gradient_checkpointing="none").build_model(2)


class FrozenStage:
    """The previous cascade level, inference only.

    Attributes: ``model`` = `GroupRegLevelModel(4, cfg)` (float32 policy, gradient checkpointing
    'none'); ``level`` = 4; ``fingerprint`` = `runtime.checkpoint.weight_fingerprint(model)`.
    Build it with one of the factories:
    * `from_checkpoint(prefix, cfg)`: a fresh run's level-4 source (strict model restore);
    * `from_model(model)`: a float32 copy of another model's weights (tests, memory script);
    * `from_config(cfg)`: random init, restored afterwards by `RunCheckpoint` (resume).
    """

    n_stages = 1                           # Trainer dispatch (milestone 9 phase 2, design 2.1)

    def __init__(self, model):
        if model.level != FROZEN_LEVEL:
            raise ValueError("the frozen stage is a level-%d model, got level %d" % (FROZEN_LEVEL, model.level))
        self.model = model
        self.level = FROZEN_LEVEL

    @property
    def source_model(self):
        """The model the trainable stage copies (`copy_weights_from`) on a fresh run."""
        return self.model

    @property
    def fingerprints(self):
        """{4: fingerprint} (same shape as `FrozenChain.fingerprints`)."""
        return {FROZEN_LEVEL: self.fingerprint}

    def checkpoint_extra(self):
        """Run-checkpoint objects of the frozen stage: {'frozen_stage': model} (milestone-8 key)."""
        return {"frozen_stage": self.model}

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


def _config_without_seed(cfg):
    return {k: v for k, v in cfg.to_dict().items() if k != "seed"}


class FrozenChain:
    """Frozen level 4 -> level 2 chain in front of a trainable level-1 stage (inference only).

    Milestone 9 phase 2, design section 2.2. ``stage4`` = a `FrozenStage` (level 4); ``model2``
    = a float32 level-2 `GroupRegLevelModel` built by `_build_frozen_model(cfg, level=2)`; both
    have the same ModelConfig (seed aside) and are only called with ``training=False``. The math
    is that of the phase-1 stand-in (`tests/gpu/frozen_chain_standin.py`) verbatim:

    * linear: x2 = pool(x1); A4 = stage4.linear(x2); A2 = Center2(R2_raw A4) with model2's
      centering (the level-2 run's composed output);
    * deform: level-4 fields upsampled to level 2 (stage4.deform(x2)), level-2 residual on
      warp(x2, U4_2), composed by `diffeo.compose_cascade` at level 2 and upsampled to level 1.

    ``level`` = 2 (PREV_LEVEL[1]); ``n_stages`` = 2. Factories: `from_checkpoint` (fresh level-1
    run, ONE milestone-8 level-2 cascade checkpoint: keys ``model`` + ``frozen_stage``),
    `from_config` (resume; restored by `RunCheckpoint` under the keys of `checkpoint_extra`),
    `from_models` (tests / tools: float32 copies).
    """

    level = CHAIN_LEVEL
    n_stages = 2

    def __init__(self, stage4, model2):
        if getattr(stage4, "level", None) != FROZEN_LEVEL or not hasattr(stage4, "model"):
            raise ValueError("FrozenChain: stage4 must be a level-%d FrozenStage" % FROZEN_LEVEL)
        if model2.level != CHAIN_LEVEL:
            raise ValueError("FrozenChain: model2 must be a level-%d model, got level %d"
                             % (CHAIN_LEVEL, model2.level))
        c4, c2 = _config_without_seed(stage4.model.config), _config_without_seed(model2.config)
        if c4 != c2:
            diff = sorted(k for k in set(c4) | set(c2) if c4.get(k) != c2.get(k))
            raise ValueError("FrozenChain: the level-4 and level-2 model configs differ in %s" % diff)
        self.stage4 = stage4
        self.model2 = model2

    @classmethod
    def from_checkpoint(cls, prefix, cfg, expected_fingerprint4):
        """Chain of `cfg` restored from a milestone-8 level-2 cascade checkpoint `prefix`
        (``model`` = level 2, ``frozen_stage`` = level 4; optimizer, step, RNGs not read). The
        restored level-4 weights must match `expected_fingerprint4` (the source config.json
        ``cascade.fingerprint``). Every failure leaves as ValueError (design 2.2)."""
        from runtime import checkpoint as C        # local import: models/ stays runtime-free at import
        model4 = _build_frozen_model(cfg, level=FROZEN_LEVEL)
        model2 = _build_frozen_model(cfg, level=CHAIN_LEVEL)
        try:
            status = tf.train.Checkpoint(model=model2, frozen_stage=model4).restore(prefix)
            status.expect_partial()
            status.assert_existing_objects_matched()
        except (AssertionError, ValueError, tf.errors.NotFoundError, tf.errors.InvalidArgumentError) as e:
            raise ValueError("FrozenChain.from_checkpoint(%s): %s: %s" % (prefix, type(e).__name__, e)) from e
        fp4 = C.weight_fingerprint(model4)
        if not C.fingerprints_match(expected_fingerprint4, fp4):
            raise ValueError("FrozenChain.from_checkpoint(%s): the frozen level-4 weights %s do not match the "
                             "source config.json cascade.fingerprint %s" % (prefix, fp4, expected_fingerprint4))
        return cls(FrozenStage(model4), model2)

    @classmethod
    def from_config(cls, cfg):
        """Randomly initialised chain of `cfg` (its weights come from the run checkpoint)."""
        return cls(FrozenStage(_build_frozen_model(cfg, level=FROZEN_LEVEL)),
                   _build_frozen_model(cfg, level=CHAIN_LEVEL))

    @classmethod
    def from_models(cls, model4, model2):
        """Fresh float32 models with the weights of `model4` (level 4) and `model2` (level 2)
        copied (`copy_weights_from`; the inputs are never wrapped)."""
        if model2.level != CHAIN_LEVEL:
            raise ValueError("FrozenChain.from_models: model2 must be a level-%d model, got level %d"
                             % (CHAIN_LEVEL, model2.level))
        m2 = _build_frozen_model(model2.config, level=CHAIN_LEVEL)
        m2.copy_weights_from(model2)
        return cls(FrozenStage.from_model(model4), m2)

    @property
    def models(self):
        return [self.stage4.model, self.model2]

    @property
    def source_model(self):
        """The level-2 model (the trainable level-1 stage copies it on a fresh run)."""
        return self.model2

    @property
    def fingerprints(self):
        from runtime import checkpoint as C
        return {FROZEN_LEVEL: C.weight_fingerprint(self.stage4.model),
                CHAIN_LEVEL: C.weight_fingerprint(self.model2)}

    @property
    def fingerprint(self):
        """Fingerprint of the stage directly before the trainable one (level 2; m8 meaning)."""
        from runtime import checkpoint as C
        return C.weight_fingerprint(self.model2)

    def checkpoint_extra(self):
        """Run-checkpoint objects of the chain: {'frozen_stage_4', 'frozen_stage_2'}."""
        return {"frozen_stage_4": self.stage4.model, "frozen_stage_2": self.model2}

    def _linear(self, images_l1, task):
        x2 = pool_to_coarser(images_l1)
        A4 = self.stage4.linear(x2, task)
        out = self.model2(prewarp_linear(x2, A4, P.LEVEL_SPACING[CHAIN_LEVEL]), task=task, training=False)
        return tf.stop_gradient(compose_linear(out["A_raw"], A4, self.model2.centering)), A4

    def linear(self, images_l1, task):
        """A_prev [N, 4, 4] float32 (image i -> level-2 central frame), no gradient."""
        return self._linear(images_l1, task)[0]

    def linear_stages(self, images_l1, task):
        """{'A': `linear` result, 'A4': the level-4 stage alone} (validation / skip)."""
        A, A4 = self._linear(images_l1, task)
        return {"A": A, "A4": tf.stop_gradient(A4)}

    def _deform(self, images_l1):
        x2 = pool_to_coarser(images_l1)
        p4 = self.stage4.deform(x2)                       # level-4 fields upsampled to level 2
        out = self.model2(prewarp_deform(x2, p4["U"]), task=DEFORM_TASK, training=False)
        u2, u2i = diffeo.compose_cascade(p4["U"], p4["U_inv"], out["u"], out["u_inv"])
        up = diffeo.upsample_displacement
        return {"U": tf.stop_gradient(up(u2)), "U_inv": tf.stop_gradient(up(u2i)),
                "V": tf.stop_gradient(up(out["v"]))}, p4

    def deform(self, images_l1):
        """dict U, U_inv, V (level-1 voxels): composed level-4 + level-2 fields, upsampled, no gradient."""
        return self._deform(images_l1)[0]

    def deform_stages(self, images_l1):
        """`deform` plus U4, U4_inv, V4: the level-4 stage alone, upsampled to level 1."""
        res, p4 = self._deform(images_l1)
        up = diffeo.upsample_displacement
        res = dict(res)
        res.update({k: tf.stop_gradient(up(p4[s])) for k, s in (("U4", "U"), ("U4_inv", "U_inv"), ("V4", "V"))})
        return res

    def linear_skip(self, images_l1, task, skip4=None, skip2=None):
        """A_prev [N, 4, 4] with per-stage skips (retrain-prereqs design 4.1 / 4.4; bool scalar
        tensors, None = not built). skip4: A4 = I and the level-2 stage still runs on the
        identity-pre-warped x2 (A = Center2(R2_raw)); skip2: A = A4; both: A = I. No gradient."""
        x2 = pool_to_coarser(images_l1)
        A4 = self.stage4.linear(x2, task)
        if skip4 is not None:
            A4 = skip_linear(skip4, A4)
        out = self.model2(prewarp_linear(x2, A4, P.LEVEL_SPACING[CHAIN_LEVEL]), task=task, training=False)
        A = compose_linear(out["A_raw"], A4, self.model2.centering)
        if skip2 is not None:
            A = skip_to(skip2, A4, A)
        return tf.stop_gradient(A)

    def deform_skip(self, images_l1, skip4=None, skip2=None):
        """dict U, U_inv (level-1 voxels) with per-stage skips (retrain-prereqs design 4.1 / 4.4).
        skip4: u4 = u4_inv = 0 and the level-2 stage runs on x2 warped by the zero field; skip2:
        U = up(U4); both: U = U_inv = 0. No gradient."""
        x2 = pool_to_coarser(images_l1)
        p4 = self.stage4.deform(x2)                       # level-4 fields upsampled to level 2
        U4, U4i = p4["U"], p4["U_inv"]
        if skip4 is not None:
            U4, U4i = skip_fields(skip4, [U4, U4i])
        out = self.model2(prewarp_deform(x2, U4), task=DEFORM_TASK, training=False)
        u2, u2i = diffeo.compose_cascade(U4, U4i, out["u"], out["u_inv"])
        up = diffeo.upsample_displacement
        U, Ui = up(u2), up(u2i)
        if skip2 is not None:
            U = skip_to(skip2, up(U4), U)
            Ui = skip_to(skip2, up(U4i), Ui)
        return {"U": tf.stop_gradient(U), "U_inv": tf.stop_gradient(Ui)}


def skip_to(skip, coarse, fine):
    """`coarse` when `skip` (bool scalar tensor) is true, else `fine` (A [N, 4, 4] or a field):
    the 3-level cascade's skip of the last frozen stage (name scope ``cascade_skip``)."""
    with tf.name_scope(SKIP_SCOPE):
        return tf.where(skip, coarse, fine)


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
