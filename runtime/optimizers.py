"""Optimizers: the rigid project's SGDW (ported) and a factory for SGDW / SGD / Adam / AdamW.

Defaults (rigid project, design 5.2): SGDW, lr 0.01, momentum 0.99, dampening 0, Nesterov,
coupled weight decay 3e-5, ``global_clipnorm`` 12, no weight decay on BatchNorm affine
parameters and biases (variable names matching ``gamma``, ``beta`` or ``bias``; the reference
list ``['normalization', 'bias']`` does not match this model's names). ``jit_compile`` is off
unless XLA is requested.

Port notes: the reference ``_use_weight_decay`` tested ``variable in var_list``, which compares
tf.Variables elementwise; this port compares by identity. One optimizer is shared by all tasks:
build it on ALL trainable variables (`build_optimizer`) before the first step, otherwise a
rigid-first run has no slots for the scale/shear head and a later affine step fails the
variable-key lookup.

Weight decay per optimizer (retrain-prereqs design 2): ``adam`` ignores ``weight_decay``, so
every m4-m9 Adam run trained with zero weight decay whatever its config.json said. ``adamw`` is
Keras 2.10 ``experimental.AdamW``: decoupled decay ``w <- w - lr_t * wd * w`` before the
unchanged Adam update, with lr_t = the optimizer's current (scheduled) ``learning_rate``;
excluded names gamma / beta / bias (BatchNorm and LayerNorm affine parameters and every bias).
At wd 0 AdamW equals Adam op for op. A non-finite step is skipped together with its decay
(the decay lives in ``update_step``): in fp32 by the Trainer's ``tf.cond(finite, apply, skip)``
around ``apply_gradients``; under AMP the Trainer gates only on input finiteness and passes
non-finite gradients on, so the LossScaleOptimizer's own check is the gradient guard (it skips
the update and lowers its scale).
"""
import re

import tensorflow as tf

DEFAULT_WD_EXCLUDE = ("gamma", "beta", "bias")
OPTIMIZERS = ("sgdw", "sgd", "adam", "adamw")
WEIGHT_DECAY_OPTIMIZERS = ("sgdw", "adamw")  # the optimizers that apply --weight-decay
_Base = tf.keras.optimizers.experimental.Optimizer


class SGDW(_Base):
    """SGD with momentum (PyTorch-style dampening), Nesterov and coupled or decoupled weight decay.

    Update (dense gradients g, variable w, lr, wd, momentum mu, dampening d):
      b = g + wd w (coupled, if w uses weight decay) else g
      m <- mu m + (1 - d) b;  b <- b + mu m (Nesterov) or m
      w <- w - lr b;  then w <- w - lr wd w if decoupled (and w uses weight decay)
    """

    def __init__(self, learning_rate=0.01, weight_decay=0.00003, momentum=0.99, dampening=0.0,
                 nesterov=True, decoupled=False, clipnorm=None, clipvalue=None,
                 global_clipnorm=None, use_ema=False, ema_momentum=0.999,
                 ema_overwrite_frequency=None, jit_compile=False, name="sgdw", **kwargs):
        super().__init__(name=name, clipnorm=clipnorm, clipvalue=clipvalue,
                         global_clipnorm=global_clipnorm, use_ema=use_ema,
                         ema_momentum=ema_momentum, ema_overwrite_frequency=ema_overwrite_frequency,
                         jit_compile=jit_compile, **kwargs)
        if isinstance(learning_rate, (int, float)) and learning_rate < 0:
            raise ValueError("`learning_rate` must be between [0, inf].")
        if isinstance(momentum, (int, float)) and not 0 <= momentum <= 1:
            raise ValueError("`momentum` must be between [0, 1].")
        if isinstance(dampening, (int, float)) and not 0 <= dampening <= 1:
            raise ValueError("`dampening` must be between [0, 1].")
        if isinstance(weight_decay, (int, float)) and weight_decay < 0:
            raise ValueError("`weight_decay` must be between [0, inf].")
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.decoupled = decoupled

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.momentums = []
        if self.momentum != 0:
            for var in var_list:
                self.momentums.append(self.add_variable_from_reference(model_variable=var, variable_name="m"))
        self._built = True

    def _use_weight_decay(self, variable):
        if not self.weight_decay:
            return False
        for v in getattr(self, "_exclude_from_weight_decay", []):
            if v is variable:
                return False
        for name in getattr(self, "_exclude_from_weight_decay_names", []):
            if re.search(name, variable.name) is not None:
                return False
        return True

    def update_step(self, gradient, variable):
        lr = tf.cast(self.learning_rate, variable.dtype)
        wd = tf.cast(self.weight_decay, variable.dtype)
        damp = tf.cast(self.dampening, variable.dtype)
        use_wd = self._use_weight_decay(variable)
        m = None
        if self.momentum != 0:
            momentum = tf.cast(self.momentum, variable.dtype)
            m = self.momentums[self._index_dict[self._var_key(variable)]]
        buf = gradient + variable * wd if (not self.decoupled and use_wd) else gradient
        if m is not None:
            m.assign(m * momentum + buf * (1.0 - damp))
            buf = buf + m * momentum if self.nesterov else m
        variable.assign_sub(buf * lr)
        if self.decoupled and use_wd:
            variable.assign_sub(variable * wd * lr)

    def get_config(self):
        config = super().get_config()
        config.update({"learning_rate": self._serialize_hyperparameter(self._learning_rate),
                       "weight_decay": self.weight_decay, "momentum": self.momentum,
                       "dampening": self.dampening, "nesterov": self.nesterov,
                       "decoupled": self.decoupled})
        return config

    def exclude_from_weight_decay(self, var_list=None, var_names=None):
        """Exclude variables (by identity) or name regexes from weight decay. Call before build."""
        if getattr(self, "_built", False):
            raise ValueError("`exclude_from_weight_decay()` can only be configured before the optimizer is built.")
        self._exclude_from_weight_decay = list(var_list or [])
        self._exclude_from_weight_decay_names = list(var_names or [])


def make_optimizer(name="sgdw", lr=0.01, weight_decay=3e-5, momentum=0.99, dampening=0.0,
                   nesterov=True, decoupled=False, clipnorm=12.0, exclude_wd=True, amp=False,
                   jit_compile=False):
    """Build one optimizer (wrapped in a LossScaleOptimizer when ``amp``).

    ``clipnorm`` is the global clip norm (None or <= 0 disables). ``weight_decay`` is used by
    `WEIGHT_DECAY_OPTIMIZERS` only (``adam`` and ``sgd`` ignore it); ``adamw`` decays
    ``w <- w - lr_t * wd * w`` before the Adam update (lr_t = current learning_rate). The
    weight-decay exclusion (`DEFAULT_WD_EXCLUDE` names) applies to sgdw and adamw.
    """
    clip = float(clipnorm) if clipnorm is not None and clipnorm > 0 else None
    exp = tf.keras.optimizers.experimental
    common = dict(global_clipnorm=clip, jit_compile=bool(jit_compile))
    if name == "sgdw":
        opt = SGDW(learning_rate=lr, weight_decay=weight_decay, momentum=momentum,
                   dampening=dampening, nesterov=nesterov, decoupled=decoupled, **common)
    elif name == "sgd":
        opt = exp.SGD(learning_rate=lr, momentum=momentum, nesterov=nesterov, **common)
    elif name == "adam":
        opt = exp.Adam(learning_rate=lr, **common)
    elif name == "adamw":
        opt = exp.AdamW(learning_rate=lr, weight_decay=weight_decay, **common)
    else:
        raise ValueError("optimizer must be one of {}, got {!r}".format(OPTIMIZERS, name))
    if exclude_wd and name in ("sgdw", "adamw"):
        opt.exclude_from_weight_decay(var_names=list(DEFAULT_WD_EXCLUDE))
    if amp:
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    return opt


def inner_optimizer(opt):
    """The wrapped optimizer of a LossScaleOptimizer, else `opt` itself."""
    return getattr(opt, "inner_optimizer", opt)


def build_optimizer(opt, variables):
    """Create every slot for `variables` now (call with ALL trainable variables)."""
    inner_optimizer(opt).build(list(variables))
    return opt


def set_learning_rate(opt, value):
    inner_optimizer(opt).learning_rate.assign(float(value))


def get_learning_rate(opt):
    return float(inner_optimizer(opt).learning_rate.numpy())


def uses_weight_decay(opt, variable):
    """True if `variable` gets weight decay from `opt`."""
    inner = inner_optimizer(opt)
    if isinstance(inner, SGDW) or hasattr(inner, "_use_weight_decay"):
        if not getattr(inner, "weight_decay", 0):
            return False
        return bool(inner._use_weight_decay(variable))
    return False
