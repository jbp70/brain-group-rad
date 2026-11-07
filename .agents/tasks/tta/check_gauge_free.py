"""Design v2 check (numpy + scipy only, env bgr_preproc38): gauge-free linear TTA averaging.

Simulates one linear cascade level for a group of N images and the 8 group flips:
* a "centered-equivariant" model: unflipped raw output of flip f is G_f A_i (G_f = an arbitrary
  flip-dependent common left factor, e.g. from a head bias), so Center(raw_f) is the same for
  every f, but raw parameters are not;
* v1 method (average raw 10-D / t / log-scale / shear over flips, then center);
* v2 method (per flip: A_f = Center(raw_f A_prev); residual R_f = A_f A_prev^-1; re-encode R_f
  as [10-D of Q, t, log diag K, K01, K02, K12] with L = Q K (positive diagonal); average; decode
  with the exact smallest-eigenvalue eigenvector (Markley mean); compose + center once).
Checks: E1 encode/decode round trip of affine params (our scale_shear_matrix convention);
E2 encode(F R F) == sigma_f * encode(R) (up to the 10-D sign-free form); E3 v1 vs v2 pairwise
error with zero model error; E4 v2 TTA(flip_g x) == F_g TTA(x) F_g with a NON-equivariant model.
"""
import itertools

import numpy as np
from scipy.linalg import expm, logm

rng = np.random.default_rng(0)
FLIPS = list(itertools.product((False, True), repeat=3))


def fmat(fl):
    return np.diag([-1.0 if f else 1.0 for f in fl] + [1.0])


def rot(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def quat(r):                       # any unit quaternion of r (sign irrelevant for I - q q^T)
    m = np.array([[r[0, 0] + r[1, 1] + r[2, 2], r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]],
                  [r[2, 1] - r[1, 2], r[0, 0] - r[1, 1] - r[2, 2], r[0, 1] + r[1, 0], r[0, 2] + r[2, 0]],
                  [r[0, 2] - r[2, 0], r[0, 1] + r[1, 0], r[1, 1] - r[0, 0] - r[2, 2], r[1, 2] + r[2, 1]],
                  [r[1, 0] - r[0, 1], r[0, 2] + r[2, 0], r[1, 2] + r[2, 1], r[2, 2] - r[0, 0] - r[1, 1]]])
    w, v = np.linalg.eigh(m / 3.0)
    return v[:, -1]


def qmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


IU = np.triu_indices(4)            # a00 a01 a02 a03 a11 a12 a13 a22 a23 a33 (row-major upper)


def encode(m):
    """[L | t] -> 19 params: 10-D of Q (I - q q^T), t, log diag K, K01, K02, K12 (L = Q K)."""
    q_, k = np.linalg.qr(m[:3, :3])
    s = np.where(np.diag(k) < 0, -1.0, 1.0)
    q_, k = q_ * s[None, :], s[:, None] * k
    q = quat(q_)
    a = np.eye(4) - np.outer(q, q)
    return np.concatenate([a[IU], m[:3, 3], np.log(np.diag(k)), [k[0, 1], k[0, 2], k[1, 2]]])


def decode(p):
    a = np.zeros((4, 4))
    a[IU] = p[:10]
    a = a + np.triu(a, 1).T
    w, v = np.linalg.eigh(a)
    k = np.array([[np.exp(p[13]), p[16], p[17]], [0, np.exp(p[14]), p[18]], [0, 0, np.exp(p[15])]])
    out = np.eye(4)
    out[:3, :3] = qmat(v[:, 0]) @ k
    out[:3, 3] = p[10:13]
    return out


def sigma(fl):
    f = np.array([-1.0 if x else 1.0 for x in fl])
    sq = np.array([1.0, f[1] * f[2], f[0] * f[2], f[0] * f[1]])
    rot10 = np.outer(sq, sq)[IU]
    return np.concatenate([rot10, f, np.ones(3), [f[0] * f[1], f[0] * f[2], f[1] * f[2]]])


def center(mats, iters=30):
    mu = np.eye(4)
    for _ in range(iters):
        x = np.mean([np.real(logm(np.linalg.solve(mu, a))) for a in mats], 0)
        mu = mu @ expm(x)
    return [np.linalg.solve(mu, a) for a in mats]


def rand_affine(deg=30, mm=15, ls=0.08, sh=0.04):
    m = np.eye(4)
    k = np.eye(3)
    k[np.diag_indices(3)] = np.exp(rng.uniform(-ls, ls, 3))
    k[0, 1], k[0, 2], k[1, 2] = rng.uniform(-sh, sh, 3)
    m[:3, :3] = rot(rng.normal(size=3), rng.uniform(-deg, deg)) @ k
    m[:3, 3] = rng.uniform(-mm, mm, 3)
    return m


def pair_err(a, b):
    pts = np.c_[rng.uniform(-60, 60, (500, 3)), np.ones(500)]
    e = 0.0
    for i in range(len(a)):
        for j in range(len(a)):
            if i != j:
                d = np.linalg.inv(a[j]) @ a[i] - np.linalg.inv(b[j]) @ b[i]
                e = max(e, np.sqrt(np.mean(np.sum((d @ pts.T)[:3] ** 2, 0))))
    return e


def tta_v2(raw_unflipped, a_prev):
    """raw_unflipped[f][i]: unflipped raw residual of flip f. Returns the TTA group."""
    n = len(a_prev)
    params = np.zeros((n, 19))
    for raw in raw_unflipped:
        a_f = center([raw[i] @ a_prev[i] for i in range(n)])
        params += np.array([encode(a_f[i] @ np.linalg.inv(a_prev[i])) for i in range(n)])
    params /= len(raw_unflipped)
    return center([decode(params[i]) @ a_prev[i] for i in range(n)])


def tta_v1(raw_unflipped, a_prev):
    n = len(a_prev)
    params = np.mean([[encode(r[i]) for i in range(n)] for r in raw_unflipped], 0)
    return center([decode(params[i]) @ a_prev[i] for i in range(n)])


n = 6
# E1 round trip
e1 = max(np.abs(decode(encode(m)) - m).max() for m in [rand_affine() for _ in range(50)])
# E2 sign covariance of the encoding
e2 = 0.0
for fl in FLIPS:
    F = fmat(fl)
    for _ in range(20):
        m = rand_affine()
        e2 = max(e2, np.abs(encode(F @ m @ F) - sigma(fl) * encode(m)).max())
# E3 gauge: centered-equivariant model, flip-dependent raw gauge
truth = [rand_affine() for _ in range(n)]
a_prev = center([rand_affine(10, 5) for _ in range(n)])
res_true = [truth[i] @ np.linalg.inv(a_prev[i]) for i in range(n)]     # residual giving `truth`
ref = center([res_true[i] @ a_prev[i] for i in range(n)])
gauges = [rand_affine(15, 6) for _ in FLIPS]
raws = [[g @ res_true[i] for i in range(n)] for g in gauges]
e3_v1, e3_v2 = pair_err(tta_v1(raws, a_prev), ref), pair_err(tta_v2(raws, a_prev), ref)
cen_v2 = max(np.abs(a - b).max() for a, b in zip(tta_v2(raws, a_prev), ref))


# E4 equivariance of the v2 pipeline with a non-equivariant "model"
def model(group_prev_frame_key):   # deterministic, non-equivariant: raw depends on flip state
    return [rand_affine(20, 8) for _ in range(n)]


rng_state = rng.bit_generator.state
outs = {fl: model(fl) for fl in FLIPS}          # flipped-frame raw outputs of the original group
F_all = {fl: fmat(fl) for fl in FLIPS}


def unflip(fl, raw):
    F = F_all[fl]
    return [F @ r @ F for r in raw]


tta_x = tta_v2([unflip(fl, outs[fl]) for fl in FLIPS], a_prev)
e4 = 0.0
for g in [(True, False, False), (False, True, False), (False, False, True), (True, True, True)]:
    G = fmat(g)
    # the group flipped by g: its flip f input is flip_{f g} of x, so its raw outputs are outs[f xor g]
    a_prev_g = [G @ a @ G for a in a_prev]
    raws_g = []
    for fl in FLIPS:
        fg = tuple(a != b for a, b in zip(fl, g))
        raws_g.append(unflip(fl, outs[fg]))
    tta_g = tta_v2(raws_g, a_prev_g)
    e4 = max(e4, max(np.abs(tg - G @ t @ G).max() for tg, t in zip(tta_g, tta_x)))

# E5 the model's centering runs a fixed number of iterations (centering_iters, default 4): the
# gauge is then removed only up to the fixed-point residual
_center_conv = center
center = lambda mats, iters=4: _center_conv(mats, iters)  # noqa: E731
ref4 = center([res_true[i] @ a_prev[i] for i in range(n)])
e5 = pair_err(tta_v2(raws, a_prev), ref4)
cen5 = max(np.abs(a - b)[:3, 3].max() for a, b in zip(tta_v2(raws, a_prev), ref4))
center = _center_conv

print("E1 affine encode/decode round trip max |diff|: %.2e" % e1)
print("E5 4-iteration centering: v2 pair err %.2e mm, centered translation max |diff| %.2e mm"
      % (e5, cen5))
print("E2 max |encode(F M F) - sigma_f * encode(M)| over 8 flips x 20: %.2e" % e2)
print("E3 flip-dependent raw gauge, zero model error: v1 raw averaging pair err %.3f mm; "
      "v2 gauge-free pair err %.2e mm; v2 centered max |diff| %.2e" % (e3_v1, e3_v2, cen_v2))
print("E4 v2 TTA(flip_g x) vs F_g TTA(x) F_g, non-equivariant model, max |diff|: %.2e" % e4)
ok = e1 < 1e-10 and e2 < 1e-10 and e3_v2 < 1e-8 and e4 < 1e-8
print("ALL PASS" if ok else "FAIL")
