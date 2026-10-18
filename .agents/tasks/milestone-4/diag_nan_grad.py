"""Intermittent non-finite conv-kernel gradient in tests.test_group_net.TestGradients (full width,
N = 3, affine). Repeat the test's computation for dense_norm layer / batch and report which
gradients are non-finite. Usage: python diag_nan_grad.py [reps] [device]
"""
import os
import sys

sys.path.insert(0, os.getcwd())
from tests import tf_setup  # noqa: F401,E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from data_loading import params as P  # noqa: E402
from models import group_net as G  # noqa: E402
from models import transforms as tfm  # noqa: E402
from models.config import ModelConfig  # noqa: E402
from tests.test_group_net import images, random_rigid, set_random_heads, free_memory  # noqa: E402

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DEV = sys.argv[2] if len(sys.argv) > 2 else "/GPU:0"
NORMS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["layer", "batch"]

for dn in NORMS:
    for rep in range(REPS):
        with tf.device(DEV):
            tf.random.set_seed(0)
            rng = np.random.default_rng(0)
            n = 3
            x = images(4, n, seed=5)
            m = G.GroupRegLevelModel(4, ModelConfig(group_config="A", dense_norm=dn)).build_model()
            set_random_heads(m, x, rng, training=True)
            t = np.stack([random_rigid(rng) for _ in range(n)])
            target = tfm.barycentric_targets(tf.constant(t, tf.float32))
            s = tfm.brain_second_moment(np.ones(P.LEVEL_SHAPE[4]), P.LEVEL_SPACING[4])
            with tf.GradientTape() as tape:
                out = m(x, task="affine", training=True)
                loss = tf.reduce_mean(tfm.point_loss(out["A"] - target, s))
            grads = tape.gradient(loss, m.trainable_variables)
        bad = []
        for v, g in zip(m.trainable_variables, grads):
            if g is None:
                continue
            g = tf.convert_to_tensor(g).numpy()
            if not np.all(np.isfinite(g)):
                bad.append("%s (nan %d, inf %d of %d)" % (v.name.split("/", 1)[1], np.isnan(g).sum(),
                                                         np.isinf(g).sum(), g.size))
        print("dense_norm=%s rep %d loss %.3f  non-finite: %s" % (dn, rep, float(loss), bad or "none"),
              flush=True)
        del m, grads, tape, out
        free_memory()
