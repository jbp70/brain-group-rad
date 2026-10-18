"""Why does eval mode stay near identity after the overfit? Train the narrow model on one fixed
synthetic group (Adam 3e-4, N=4), then compare train-mode vs eval-mode centered RMS:
(1) right after training, (2) after recalibrating the BN moving statistics with K forward passes
in training mode on the same group (weights fixed), (3) per BN layer: batch var / moving var.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
from tests import tf_setup  # noqa: F401,E402
from tests.runtime_helpers import narrow_config, synthetic_group  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from data_loading.postprocess import LinearPostprocess  # noqa: E402
from models import layers as L  # noqa: E402
from models.group_net import GroupRegLevelModel  # noqa: E402
from runtime import fixed_data as fd, metrics as M, optimizers as O  # noqa: E402
from runtime.run import Trainer, TrainConfig  # noqa: E402
from runtime.schedulers import ScheduleConfig  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
g = synthetic_group(4, 4, "rigid", seed=0, ranges={"log_scale": 0.0, "shear": 0.0}, nuisance=False)
model = GroupRegLevelModel(4, narrow_config(dropout=0.0)).build_model(4)
tr = Trainer(model, O.make_optimizer("adam", lr=3e-4), {"rigid": fd.FixedGroupSource(g["batch"])}, g["S"],
             TrainConfig(level=4, group_size=4, log_every=0, val_every=0, flip_prob=0.0,
                         schedule=ScheduleConfig("constant", 3e-4, STEPS, 0)))
tr.fit(STEPS)
print("train-mode step RMS (last 10): %.3f mm" % np.mean([h["center_rms_mm"] for h in tr.history[-10:]]))
b = LinearPostprocess("rigid", 4, flip_prob=0.0)(*g["batch"])
x = b["images"]
a_star = b["A_star"].numpy().astype(np.float64)


def rms(training):
    a = model(x, task="rigid", training=training)["A"].numpy().astype(np.float64)
    return float(np.mean(M.centered_errors(a, a_star, g["S"])["rms_mm"]))


bns = [m for m in model.submodules if isinstance(m, L.BatchNormWithSeparateAffine)]


def stats_table():
    rows = []
    for m in bns:
        mv = m.bn.moving_variance.numpy()
        rows.append((m.bn.name if hasattr(m.bn, "name") else "?", m.name, mv.mean()))
    return rows


print("eval-mode RMS after training: %.3f mm" % rms(False))
# batch variance of each BN input vs moving variance (hook via a forward in training mode)
captured = {}
orig = {}
for m in bns:
    def call(xx, training=None, m=m):
        xf = tf.cast(xx, tf.float32)
        axes = list(range(xf.shape.rank - 1))
        mean, var = tf.nn.moments(xf, axes=axes)
        captured[m.path if hasattr(m, "path") else id(m)] = (m, var.numpy(), mean.numpy())
        return orig[id(m)](xx, training=training)
    orig[id(m)] = m.call
    m.call = call
model(x, task="rigid", training=False)
for key, (m, var, mean) in captured.items():
    mv = m.bn.moving_variance.numpy()
    mm = m.bn.moving_mean.numpy()
    ratio = np.median(mv / np.maximum(var, 1e-12))
    print("  %-40s n=%5d  median(moving_var / batch_var) %8.3f  median|moving_mean-batch_mean|/std %7.3f" % (
        str(key)[:40], var.size, ratio, np.median(np.abs(mm - mean) / np.sqrt(var + 1e-5))))
for m in bns:
    m.call = orig[id(m)]
for k in (50, 200, 1000):
    for _ in range(k):
        model(x, task="rigid", training=True)
    print("after +%d training-mode forwards (weights fixed): eval RMS %.3f mm, train RMS %.3f mm" % (
        k, rms(False), rms(True)))
