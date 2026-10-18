"""Time the parts of one rigid train step (level 4): postprocess, forward+backward, full step.

Usage (tf_gpu, repo root): python .agents/tasks/milestone-4/profile_step.py <N> <narrow|full>
"""
import os
import sys
import time

sys.path.insert(0, os.getcwd())
from tests import tf_setup  # noqa: F401,E402
from tests.runtime_helpers import narrow_config, synthetic_group  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from data_loading.postprocess import LinearPostprocess  # noqa: E402
from models.config import ModelConfig  # noqa: E402
from models.group_net import GroupRegLevelModel  # noqa: E402
from runtime import fixed_data as fd  # noqa: E402
from runtime import losses, optimizers as O  # noqa: E402
from runtime.run import Trainer, TrainConfig  # noqa: E402
from runtime.schedulers import ScheduleConfig  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
width = sys.argv[2] if len(sys.argv) > 2 else "narrow"
cfg = narrow_config(dropout=0.0) if width == "narrow" else ModelConfig(dropout=0.0)
g = synthetic_group(4, n, "rigid", seed=0)
batch = g["batch"]
pp = LinearPostprocess("rigid", 4, flip_prob=0.5)
pp_fn = tf.function(lambda a, b, c: pp(a, b, c))
model = GroupRegLevelModel(4, cfg).build_model(n)
S = tf.constant(g["S"], tf.float32)


@tf.function
def fwd_bwd(x, a_star, t_target, pi, pj, p_star):
    with tf.GradientTape() as tape:
        out = model(x, task="rigid", training=True)
        loss = losses.linear_loss(out["A"], {"A_star": a_star, "T_target": t_target, "pair_i": pi,
                                             "pair_j": pj, "P_star": p_star}, S)["loss"]
    return tape.gradient(loss, model.trainable_variables)


@tf.function
def enc_only(x):
    with tf.GradientTape() as tape:
        b, _ = model.encoder(x, training=True)
        loss = tf.reduce_sum(tf.cast(b, tf.float32))
    return tape.gradient(loss, model.encoder.trainable_variables)


T_START = time.perf_counter()


def timeit(fn, k=10):
    t = time.perf_counter()
    fn()
    print("  [t=%.0f s] first call (trace + autotune) %.1f s, peak %.0f MB" % (
        time.perf_counter() - T_START, time.perf_counter() - t,
        tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2 ** 20), flush=True)
    t = time.perf_counter()
    r = fn()
    tf.nest.map_structure(lambda t: t.numpy() if hasattr(t, "numpy") else t, r)
    print("  second call %.1f s" % (time.perf_counter() - t), flush=True)
    t0 = time.perf_counter()
    for _ in range(k):
        r = fn()
    tf.nest.map_structure(lambda t: t.numpy() if hasattr(t, "numpy") else t, r)
    return 1e3 * (time.perf_counter() - t0) / k


b = pp_fn(*batch)
print("N=%d width=%s" % (n, width))
print("postprocess      %.1f ms" % timeit(lambda: pp_fn(*batch)))
print("encoder fwd+bwd  %.1f ms" % timeit(lambda: enc_only(b["images"])))
print("model fwd+bwd    %.1f ms" % timeit(lambda: fwd_bwd(b["images"], b["A_star"], b["T_target"], b["pair_i"],
                                                         b["pair_j"], b["P_star"])))
tr = Trainer(model, O.make_optimizer("sgdw"), {"rigid": fd.FixedGroupSource(batch)}, g["S"],
             TrainConfig(level=4, group_size=n, log_every=0, val_every=0,
                         schedule=ScheduleConfig("constant", 1e-4, 100, 0)))
print("full train step  %.1f ms" % timeit(lambda: tr._steps["rigid"](*batch)))
print("peak GPU MB      %.0f" % (tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2 ** 20))
