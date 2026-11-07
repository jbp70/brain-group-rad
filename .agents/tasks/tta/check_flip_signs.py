"""TTA design check (design-only artifact, not part of the code base).

Verifies numerically, with the repository's own transform / diffeo code:
  C1 rigid:  rigid_matrix(p * s_f)  == F rigid_matrix(p) F   for random (non-rank-deficient) 10-D p
  C2 affine: affine_matrix(p * s_f) == F affine_matrix(p) F  (19 params: 13 rigid + 3 log-scales + 3 shears)
  C3 the sign vectors equal Jay's reference patterns (x, y, z) and combined flips = products
  C4 centering commutes with conjugation: center(F A F) == F center(A) F
  C5 image flip convention: transform_image(flip(I), F M F) == flip(transform_image(I, M)) (level 4)
  C6 SVF convention: with v^F(p) = F v(F p) (array reverse + negate flipped components):
     exp(v^F) == (exp v)^F  and  warp(flip(I), exp(v^F)) == flip(warp(I, exp(v)))
  C7 averaging 8 identical (equivariant) predictions returns the original params bit-for-bit
     after unflipping, and decode is unchanged.
Run: wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/scripts/run_in_tf_env.sh ../.agents/tasks/tta/check_flip_signs.py
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import numpy as np  # noqa: E402

import tests.tf_setup  # noqa: E402,F401
import tensorflow as tf  # noqa: E402

from models import diffeo  # noqa: E402
from models import layers as L  # noqa: E402
from models import transforms as tfm  # noqa: E402

rng = np.random.RandomState(0)
JAY = {"x": np.array([1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1], np.float64),
       "y": np.array([1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1], np.float64),
       "z": np.array([1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1], np.float64)}
IU = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]


def derived_signs(f):
    """Sign vector [19] derived from F = diag(f): quaternion signs (1, f_x f_? ...)."""
    fx, fy, fz = f
    # F R F for the quaternion (w, x, y, z): q -> (w, fy fz x, fx fz y, fx fy z) (up to global sign)
    sq = np.array([1.0, fy * fz, fx * fz, fx * fy])
    rot = np.array([sq[i] * sq[j] for i, j in IU])
    trans = np.array([fx, fy, fz], np.float64)
    scale = np.ones(3)
    shear = np.array([fx * fy, fx * fz, fy * fz])
    return np.concatenate([rot, trans, scale, shear])


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


flips = list(itertools.product([False, True], repeat=3))
ok = True
report = []

# C3: patterns
for ax, k in (("x", 0), ("y", 1), ("z", 2)):
    f = [1.0, 1.0, 1.0]
    f[k] = -1.0
    d = derived_signs(f)[:13]
    same = np.array_equal(d, JAY[ax])
    ok &= same
    report.append("C3 axis %s derived == Jay: %s" % (ax, same))
for fl in flips:
    f = [-1.0 if b else 1.0 for b in fl]
    prod = np.ones(13)
    for ax, b in zip("xyz", fl):
        if b:
            prod = prod * JAY[ax]
    same = np.array_equal(prod, derived_signs(f)[:13])
    ok &= same
report.append("C3 combined flips == products of Jay's patterns: %s" % ok)

# C1/C2: random params (general symmetric 10-D, not I - qq^T)
B = 64
p_rot = rng.normal(size=(B, 10)) + tfm.identity_rotation_10d()[None] * 2.0
p_t = rng.normal(scale=20.0, size=(B, 3))
p_ss = np.concatenate([rng.normal(scale=0.15, size=(B, 3)), rng.normal(scale=0.2, size=(B, 3))], 1)
p19 = np.concatenate([p_rot, p_t, p_ss], 1)
worst_r = worst_a = 0.0
for fl in flips:
    f = np.array([-1.0 if b else 1.0 for b in fl])
    s = derived_signs(f)
    F = tfm.flip_matrix(fl, tf.float64)
    m_r = tfm.rigid_matrix(tf.constant(p19[:, :13]))
    m_rf = tfm.rigid_matrix(tf.constant(p19[:, :13] * s[:13]))
    worst_r = max(worst_r, maxdiff(m_rf, tfm.flip_conjugate(m_r, F)))
    m_a = tfm.affine_matrix(tf.constant(p19))
    m_af = tfm.affine_matrix(tf.constant(p19 * s))
    worst_a = max(worst_a, maxdiff(m_af, tfm.flip_conjugate(m_a, F)))
ok &= worst_r < 1e-9 and worst_a < 1e-9
report.append("C1 rigid  max |M(p s_f) - F M(p) F| over 8 flips x %d random (float64): %.2e" % (B, worst_r))
report.append("C2 affine max |M(p s_f) - F M(p) F| over 8 flips x %d random (float64): %.2e" % (B, worst_a))

# C4: centering equivariance
A = tfm.affine_matrix(tf.constant(p19[:12]))
worst_c = 0.0
for fl in flips:
    F = tfm.flip_matrix(fl, tf.float64)
    lhs = tfm.center_group(tfm.flip_conjugate(A, F), iters=4)
    rhs = tfm.flip_conjugate(tfm.center_group(A, iters=4), F)
    worst_c = max(worst_c, maxdiff(lhs, rhs))
ok &= worst_c < 1e-8
report.append("C4 max |center(F A F) - F center(A) F| (N 12, float64): %.2e" % worst_c)

# C5: image flip convention at level 4 (40, 56, 48), 4 mm
shape, sp = (40, 56, 48), 4.0
img = rng.uniform(size=(1,) + shape + (1,)).astype(np.float32)
M = tf.cast(tfm.affine_matrix(tf.constant(np.concatenate([p_rot[:1], [[5.0, -3.0, 7.0]], p_ss[:1]], 1))),
            tf.float32)
worst_i = 0.0
for fl in flips:
    F = tfm.flip_matrix(fl, tf.float32)
    lhs = L.transform_image(tfm.flip_images(img, fl), tfm.flip_conjugate(M, F), sp)
    rhs = tfm.flip_images(L.transform_image(img, M, sp), fl)
    worst_i = max(worst_i, maxdiff(lhs, rhs))
ok &= worst_i < 1e-4
report.append("C5 max |T(flip I, F M F) - flip T(I, M)| (level 4, float32): %.2e" % worst_i)


# C6: SVF convention
def flip_field(v, fl):
    """v^F(p) = F v(F p): reverse the flipped array axes, negate the flipped components."""
    out = tfm.flip_images(v, fl)
    sign = tf.constant([-1.0 if b else 1.0 for b in fl], v.dtype)
    return out * sign


v = tf.constant(rng.normal(scale=1.5, size=(2, 5, 7, 6, 3)).astype(np.float32))
for _ in range(3):
    v = L.upsample_trilinear2x(v)              # smooth field [2, 40, 56, 48, 3] (level-4 voxels)
img2 = rng.uniform(size=(2,) + shape + (1,)).astype(np.float32)
u = diffeo.integrate_svf(v, 7)
worst_u = worst_w = worst_r2 = 0.0
for fl in flips:
    vf = flip_field(v, fl)
    worst_r2 = max(worst_r2, maxdiff(flip_field(vf, fl), v))
    uf = diffeo.integrate_svf(vf, 7)
    worst_u = max(worst_u, maxdiff(uf, flip_field(u, fl)))
    lhs = diffeo.warp(tfm.flip_images(img2, fl), uf)
    rhs = tfm.flip_images(diffeo.warp(img2, u), fl)
    worst_w = max(worst_w, maxdiff(lhs, rhs))
ok &= worst_r2 == 0.0 and worst_u < 1e-4 and worst_w < 1e-4
report.append("C6 max |v| %.2f vox; unflip(flip(v)) == v exactly: %s; max |exp(v^F) - (exp v)^F| %.2e vox; "
              "max |warp(flip I, u^F) - flip warp(I, u)| %.2e" % (float(tf.reduce_max(tf.abs(v))),
                                                                  worst_r2 == 0.0, worst_u, worst_w))

# C7: averaging 8 equivariant predictions (float32, as in inference)
p32 = p19[:8].astype(np.float32)
acc32 = np.zeros_like(p32)
acc64 = np.zeros(p32.shape, np.float64)
for fl in flips:
    s = derived_signs([-1.0 if b else 1.0 for b in fl]).astype(np.float32)
    pred_f = p32 * s                     # what a perfectly equivariant model predicts on flipped input
    acc32 = acc32 + pred_f * s           # unflip, float32 running sum
    acc64 = acc64 + (pred_f * s).astype(np.float64)
avg32 = acc32 / np.float32(8.0)
avg64 = (acc64 / 8.0).astype(np.float32)
c7_32 = float(np.max(np.abs(avg32 - p32) / np.maximum(np.abs(p32), 1e-30)))
c7_64 = maxdiff(avg64, p32)
ok &= c7_64 == 0.0
report.append("C7 average of 8 unflipped equivariant predictions: float64 accumulation max diff %.1e "
              "(exact); float32 running sum max rel diff %.1e (rounding of 3x, 5x, 7x)" % (c7_64, c7_32))

print("\n".join(report))
print("ALL PASS" if ok else "SOME CHECK FAILED")
