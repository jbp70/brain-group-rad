"""Size of |mu*(G) - mu*(I) G^-1| (check j's barycenter_equivariance_max_abs) in float64, as
computed by tests/gpu/verify_dali_pipeline.py check_j, on synthetic groups with check j's
ranges (G 30 deg / (4, -6, 8) mm; per image 20 deg / 5 mm, affine default scale/shear),
with the matrices rounded to float32 first like the DALI outputs. Used to pick --tol-bary.

Run: bash .agents/tasks/milestone-5/run_py.sh .agents/tasks/milestone-5/measure_bary.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())
from tests import tf_setup  # noqa: F401,E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from data_loading import params as P  # noqa: E402
from models import transforms as tfm  # noqa: E402
from runtime import fixed_data as fd  # noqa: E402
from runtime.curriculum import global_matrix_mm  # noqa: E402

G_AXIS = np.asarray([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
worst = {}
for task in ("rigid", "affine"):
    r = dict(P.DEFAULT_LINEAR_RANGES[task], rotation_deg=20.0, translation_mm=5.0)
    vals = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        T, tgt = fd.random_linear_transforms(rng, 8, task, r)
        g_mm = global_matrix_mm(G_AXIS, 30.0, (4.0, -6.0, 8.0))
        tgt0 = (g_mm[None] @ tgt).astype(np.float32).astype(np.float64)
        g_mm = g_mm.astype(np.float32).astype(np.float64)
        mu_g = tfm.barycenter(tf.constant(np.linalg.inv(tgt0)), iters=10).numpy()
        mu_0 = tfm.barycenter(tf.constant(np.linalg.inv(np.linalg.inv(g_mm)[None] @ tgt0)), iters=10).numpy()
        vals.append(float(np.abs(mu_g - mu_0 @ np.linalg.inv(g_mm)).max()))
    worst[task] = max(vals)
    print("%s: barycenter_equivariance_max_abs over 20 groups: median %.2e, max %.2e"
          % (task, float(np.median(vals)), max(vals)))
