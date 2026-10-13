"""Reference check: differentiable real matrix log in TF 2.10 and the GroupCentering layer.

TF 2.10's tf.linalg.logm only accepts complex inputs and has no registered gradient.
`logm_real` wraps it with a custom gradient (Frechet-derivative block identity), and
`GroupCentering` uses it to left-multiply each group of transforms by the inverse of
their group barycenter. The checks compare against scipy / the numpy demo in
demo_central_transform.py and verify gradients with finite differences.

Run:  wsl -d Ubuntu bash <repo>/scripts/run_in_tf_env.sh check_tf_group_centering.py
Env var ITERS sets the float64 layer's iteration count (default 8, which gives ~1e-12 agreement
with the numpy reference; 4 iterations leaves the center ~5e-6 mm off, plenty for training,
but trips the strict float64 tolerances below).
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from scipy.linalg import expm as sp_expm, logm as sp_logm

rng = np.random.default_rng(0)


# ------------------------------------------------------------------ differentiable logm
@tf.custom_gradient
def logm_real(A):
    """Real matrix log with a gradient, for real matrices near the identity (no negative eigenvalues).

    Forward: TF's complex logm. Backward: the adjoint of the Frechet derivative of log at A,
    which equals the upper-right block of log([[A^T, G], [0, A^T]]) (Higham, Functions of Matrices).
    """
    L = tf.cast(tf.math.real(tf.linalg.logm(tf.cast(A, tf.complex128))), A.dtype)

    def grad(G):
        n = tf.shape(A)[-1]
        At = tf.linalg.matrix_transpose(A)
        M = tf.concat([tf.concat([At, G], axis=-1),
                       tf.concat([tf.zeros_like(A), At], axis=-1)], axis=-2)
        LM = tf.math.real(tf.linalg.logm(tf.cast(M, tf.complex128)))
        return tf.cast(LM[..., :n, n:], A.dtype)

    return L, grad


# ------------------------------------------------------------------ centering layer
class GroupCentering(tf.keras.layers.Layer):
    """Left-multiply all N transforms in each group by the inverse of their group barycenter."""

    def __init__(self, iters=4, **kwargs):
        super().__init__(**kwargs)
        self.iters = iters

    def call(self, A_raw):  # [B, N, 4, 4]
        mu = tf.eye(4, batch_shape=[tf.shape(A_raw)[0]], dtype=A_raw.dtype)
        for _ in range(self.iters):
            X = tf.reduce_mean(logm_real(tf.linalg.inv(mu)[:, None] @ A_raw), axis=1)
            mu = mu @ tf.linalg.expm(X)
        return tf.linalg.inv(mu)[:, None] @ A_raw


# ------------------------------------------------------------------ helpers
def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def random_rigid(max_deg=45.0, max_trans=30.0):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    T = np.eye(4)
    T[:3, :3] = sp_expm(skew(ax * np.deg2rad(rng.uniform(-max_deg, max_deg))))
    T[:3, 3] = rng.uniform(-max_trans, max_trans, 3)
    return T


def random_affine():
    S = np.eye(4)
    S[:3, :3] = np.diag(1 + rng.uniform(-0.15, 0.15, 3)) + np.triu(rng.uniform(-0.1, 0.1, (3, 3)), 1)
    return random_rigid() @ S


def np_barycenter(As, iters=20):
    mu = np.eye(4)
    for _ in range(iters):
        mu = mu @ sp_expm(np.mean([np.real(sp_logm(np.linalg.inv(mu) @ a)) for a in As], axis=0))
    return mu


def fd_check(fn, x, eps=1e-6):
    """Directional derivative: autodiff vs central finite difference (float64)."""
    v = tf.constant(rng.normal(size=x.shape))
    with tf.GradientTape() as t:
        t.watch(x)
        y = fn(x)
    g = t.gradient(y, x)
    ad = float(tf.reduce_sum(g * v))
    fd = float((fn(x + eps * v) - fn(x - eps * v)) / (2 * eps))
    return ad, fd, abs(ad - fd) / max(abs(fd), 1e-12)


def report(name, ok, detail):
    print(f"[{'OK' if ok else 'FAIL':4s}] {name}: {detail}")


print("tf", tf.__version__, "| GPUs:", len(tf.config.list_physical_devices("GPU")))

# Brain-sized points (mm) for reporting transform differences in mm
PTS = np.c_[rng.uniform([-70, -85, -60], [70, 85, 60], (2000, 3)), np.ones(2000)].T

# 1. forward matches scipy (batched)
A_np = np.stack([[random_affine() for _ in range(5)] for _ in range(2)])  # [2, 5, 4, 4]
A = tf.constant(A_np)
L_tf = logm_real(A).numpy()
L_sp = np.stack([[np.real(sp_logm(a)) for a in g] for g in A_np])
err = np.abs(L_tf - L_sp).max()
report("logm_real forward vs scipy, batched [2,5,4,4] float64", err < 1e-10, f"max abs err {err:.1e}")

# 2. gradient of logm_real vs finite differences
W = tf.constant(rng.normal(size=(2, 5, 4, 4)))
ad, fd, rel = fd_check(lambda x: tf.reduce_sum(W * logm_real(x)), A)
report("logm_real gradient vs finite difference", rel < 1e-6, f"autodiff {ad:.6f}, FD {fd:.6f}, rel err {rel:.1e}")

# 3. float32 path runs and gradient is finite
A32 = tf.cast(A, tf.float32)
with tf.GradientTape() as t:
    t.watch(A32)
    y = tf.reduce_sum(logm_real(A32))
g32 = t.gradient(y, A32)
report("logm_real float32 forward+gradient", bool(tf.reduce_all(tf.math.is_finite(g32))),
       f"out dtype {g32.dtype}, finite grads")

# 4. GroupCentering matches numpy barycenter result, rigid and affine, float64 and float32
center = GroupCentering(iters=int(os.environ.get("ITERS", 8)), dtype="float64")  # Keras autocasts to float32 otherwise
center32 = GroupCentering(iters=4)                   # default float32 policy
for kind, gen in [("rigid", random_rigid), ("affine", random_affine)]:
    G_np = np.stack([[gen() for _ in range(6)] for _ in range(3)])  # [3, 6, 4, 4]
    ref = np.stack([np.linalg.inv(np_barycenter(list(g)))[None] @ g for g in G_np])
    for layer, tol, tag in [(center, 1e-8, "float64"), (center32, 1e-4, "float32")]:
        out = layer(tf.constant(G_np)).numpy().astype(np.float64)
        err = np.abs(out - ref).max()
        bary_mm = max(np.sqrt(np.mean(np.sum(((np_barycenter(list(o)) - np.eye(4)) @ PTS)[:3] ** 2, 0)))
                      for o in out)
        rel_err = max(np.abs(np.linalg.inv(o[j]) @ o[i] - np.linalg.inv(g[j]) @ g[i]).max()
                      for o, g in zip(out, G_np) for i in range(6) for j in range(6))
        report(f"GroupCentering ({kind}, {tag}) vs numpy reference", err < tol,
               f"max err {err:.1e} | output barycenter off by {bary_mm:.1e} mm | pairwise change {rel_err:.1e}")

# 5. gradient through the whole layer + loss tr(dA S dA^T)
pts = np.c_[rng.uniform([-70, -85, -60], [70, 85, 60], (5000, 3)), np.ones(5000)]
S = tf.constant(pts.T @ pts / len(pts))
A_raw = tf.constant(np.stack([[random_rigid() for _ in range(6)] for _ in range(2)]))
A_star = tf.constant(np.stack([[random_rigid() for _ in range(6)] for _ in range(2)]))


def loss_fn(x):
    dA = center(x) - A_star
    return tf.reduce_mean(tf.linalg.trace(dA @ S @ tf.linalg.matrix_transpose(dA)))


ad, fd, rel = fd_check(loss_fn, A_raw)
report("gradient through GroupCentering + S-weighted loss vs FD", rel < 1e-5,
       f"autodiff {ad:.4f}, FD {fd:.4f}, rel err {rel:.1e}")

# 6. invariance: left-multiplying every raw prediction by the same G leaves the output unchanged
Gm = tf.constant(random_rigid())
d = np.abs(center(Gm @ A_raw).numpy() - center(A_raw).numpy()).max()
report("output invariant to common left-multiplication", d < 1e-8, f"max change {d:.1e}")

# 7. inside tf.function, float32, timing for a realistic group
f = tf.function(lambda x: tf.reduce_sum(center32(x)))
X = tf.cast(tf.constant(np.stack([[random_rigid() for _ in range(12)] for _ in range(8)])), tf.float32)


def step():
    with tf.GradientTape() as t:
        t.watch(X)
        y = f(X)
    return t.gradient(y, X)


g = step()  # trace
t0 = time.perf_counter()
for _ in range(20):
    g = step()
dt = (time.perf_counter() - t0) / 20 * 1000
report("tf.function float32 [8,12,4,4] fwd+bwd", bool(tf.reduce_all(tf.math.is_finite(g))),
       f"{dt:.1f} ms per step")
