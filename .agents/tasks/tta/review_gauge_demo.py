"""Reviewer demo (numpy only): raw-parameter averaging under a flip-dependent common gauge.

A model whose CENTERED outputs are exactly flip-equivariant may still emit raw outputs
A_raw_i = C A_i with an arbitrary common C (the losses only see Center(A_raw)). Unflipped,
flip f then gives F C F A_i: a flip-dependent gauge. Averaging raw parameters over the 8 flips
then changes the pairwise maps A_j^-1 A_i, which are gauge-free and should be unchanged.
"""
import itertools
import numpy as np

rng = np.random.default_rng(0)


def rot(axis, deg):
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def quat(r):
    w = np.sqrt(max(0.0, 1 + np.trace(r))) / 2
    x = (r[2, 1] - r[1, 2]) / (4 * w)
    y = (r[0, 2] - r[2, 0]) / (4 * w)
    z = (r[1, 0] - r[0, 1]) / (4 * w)
    return np.array([w, x, y, z])


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def enc(m):          # [R|t] -> (Q 4x4 = I - q q^T, t)
    q = quat(m[:3, :3])
    return np.eye(4) - np.outer(q, q), m[:3, 3].copy()


def dec(Q, t):       # smallest-eigenvalue eigenvector read-out
    w, v = np.linalg.eigh(Q)
    m = np.eye(4)
    m[:3, :3] = qmat(v[:, 0])
    m[:3, 3] = t
    return m


n = 6
A = []
for _ in range(n):
    m = np.eye(4)
    m[:3, :3] = rot(rng.normal(size=3), rng.uniform(-30, 30))
    m[:3, 3] = rng.uniform(-15, 15, 3)
    A.append(m)
C = np.eye(4)
C[:3, :3] = rot([1, 0.3, 0.2], 10.0)          # common raw gauge (e.g. a head bias)
C[:3, 3] = [3.0, -2.0, 4.0]

flips = [np.diag([-1.0 if f else 1.0 for f in fl] + [1.0])
         for fl in itertools.product((False, True), repeat=3)]
avg = []
for i in range(n):
    Qs, ts = [], []
    for F in flips:
        Q, t = enc(F @ C @ F @ A[i])               # unflipped raw output of flip f
        Qs.append(Q)
        ts.append(t)
    avg.append(dec(np.mean(Qs, 0), np.mean(ts, 0)))
err = 0.0
for i in range(n):
    for j in range(n):
        if i != j:
            p_ref = np.linalg.inv(A[j]) @ A[i]
            p_tta = np.linalg.inv(avg[j]) @ avg[i]
            pts = rng.uniform(-60, 60, (500, 3))
            h = np.c_[pts, np.ones(500)]
            err = max(err, np.sqrt(np.mean(np.sum(((p_tta - p_ref) @ h.T)[:3] ** 2, 0))))
print("max pairwise point RMS error of raw-averaged TTA vs exact (mm): %.3f" % err)
