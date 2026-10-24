"""Gradient checkpointing with BatchNorm moving statistics updated exactly once per step.

Milestone 9, `.agents/tasks/milestone-9/design-m9-phase1.md` section 3.2 (``--gradient-
checkpointing full``). `tf.recompute_grad` runs its function again inside every gradient call;
in training mode that re-runs the BatchNorm moving-average updates (the documented double update
of the 'decoder' / 'both' modes). `recompute` counts the calls of the wrapped function: call 1 is
the forward pass and runs as is; every later call (the recompute inside a gradient, a second
``tape.gradient`` of a persistent tape, a diagnostic gradient) snapshots the moving mean /
variance of the given BN layers, runs the function and restores the snapshot under a control
dependency on its outputs. The recompute normalises with the batch statistics (training mode),
which the restore does not touch, so the gradients equal the non-checkpointed ones.

Ordering: the forward's moving-average assigns, the snapshot reads, the recompute's assigns and
the restores touch the same resource variables and are created in that order, so tf.function's
automatic control dependencies serialise them (design U8, checked by tests/test_checkpointing.py).
"""
import tensorflow as tf

from models import layers as L


def _bn_stats(bn_layers):
    """moving_mean / moving_variance variables of the `BatchNormWithSeparateAffine` layers."""
    out = []
    for layer in bn_layers:
        out.extend([layer.bn.moving_mean, layer.bn.moving_variance])
    return out


def _restores_needed(bn_layers, training):
    """Trace-time rule: no snapshot / restore ops when the call cannot update statistics
    (``training`` Python False / None, or every layer has ``stats_frozen``)."""
    if training is None or training is False:
        return False
    layers = [m for m in bn_layers if isinstance(m, L.BatchNormWithSeparateAffine)]
    return any(not m.stats_frozen for m in layers)


def recompute(fn, bn_layers=(), training=True):
    """tf.recompute_grad(fn) whose recompute passes leave the BN moving statistics as the forward
    pass left them (exactly one update per step). fn(*tensors) -> tensor or flat tuple / list.

    Create the returned function once per model call (its call counter belongs to that call).
    ``training`` is the training argument ``fn`` passes to its layers (read at trace time: Python
    False / None, or all ``bn_layers`` frozen, builds no snapshot or restore op). With
    ``bn_layers=()`` nothing is snapshotted (the integrator's case).
    """
    layers = list(bn_layers)
    restore = _restores_needed(layers, training)
    calls = [0]

    def wrapped(*args):
        calls[0] += 1
        if calls[0] == 1 or not restore:
            return fn(*args)
        stats = _bn_stats(layers)
        saved = [tf.identity(v) for v in stats]
        out = fn(*args)
        with tf.control_dependencies(tf.nest.flatten(out)):
            for v, s in zip(stats, saved):
                v.assign(s)
        return out

    return tf.recompute_grad(wrapped)
