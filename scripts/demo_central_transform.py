"""Demo: ground-truth central (barycentric) transforms for groupwise rigid/affine registration.

Conventions
-----------
* All transforms are 4x4 homogeneous matrices acting on world coordinates in mm,
  with the origin at the image center (so rotations are about the image center).
* T_i moves the template brain into image i: a template point p lands at T_i @ p.
  (To *render* image i you sample the template at T_i^{-1}(y) for each output voxel y.)
* T_i^{-1} maps image-i coordinates back to the template frame.
* T_j @ T_i^{-1} maps image-i coordinates to image-j coordinates (true pairwise transform).
* The network predicts A_i: image i -> central space.

Requires: numpy, scipy.
"""
import numpy as np
from scipy.linalg import expm, logm

rng = np.random.default_rng(0)
N = 6  # group size


# ---------------------------------------------------------------- helpers
def skew(w):
    """3-vector -> 3x3 skew-symmetric matrix (so(3) element)."""
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def random_rigid(max_deg=45.0, max_trans=30.0, axis=None):
    """Random rotation (random axis, angle up to max_deg) + random translation (mm)."""
    if axis is None:
        axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(-max_deg, max_deg))
    T = np.eye(4)
    T[:3, :3] = expm(skew(axis * angle))
    T[:3, 3] = rng.uniform(-max_trans, max_trans, 3)
    return T


def random_affine(max_deg=45.0, max_trans=30.0, max_scale=0.15, max_shear=0.1):
    """Rigid composed with a random scale/shear about the image center."""
    T = random_rigid(max_deg, max_trans)
    S = np.eye(4)
    S[:3, :3] = np.diag(1 + rng.uniform(-max_scale, max_scale, 3))
    S[:3, :3] += np.triu(rng.uniform(-max_shear, max_shear, (3, 3)), k=1)
    return T @ S


def log_(A):
    return np.real(logm(A))  # logm may return a ~0 imaginary part


inv = np.linalg.inv

# Points spanning a brain-sized box (mm), used to measure transform differences in mm.
PTS = np.c_[rng.uniform([-70, -85, -60], [70, 85, 60], (2000, 3)), np.ones(2000)].T


def diff_mm(A, B):
    """RMS displacement (mm) between two transforms over brain-sized points."""
    return np.sqrt(np.mean(np.sum(((A - B) @ PTS)[:3] ** 2, axis=0)))


def diff_deg(A, B):
    """Angle (degrees) of the rotation separating A and B (uses the linear 3x3 part)."""
    R = A[:3, :3].T @ B[:3, :3]
    U, _, Vt = np.linalg.svd(R)  # project to nearest rotation (handles affine input)
    R = U @ Vt
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def barycenter(As, iters=10, verbose=False):
    """Group (exponential) barycenter: the mu that solves  sum_i log(mu^{-1} A_i) = 0.

    Fixed-point iteration:  mu <- mu @ expm( mean_i log(mu^{-1} A_i) ).
    Iteration 1 (starting at mu = I) is exactly the one-step log-Euclidean mean
    expm(mean_i log A_i). Later iterations re-linearize around the current guess.
    """
    mu = np.eye(4)
    history = []
    for k in range(iters):
        X = np.mean([log_(inv(mu) @ A) for A in As], axis=0)  # mean in tangent space at mu
        mu = mu @ expm(X)
        history.append((k + 1, np.linalg.norm(X)))
    if verbose:
        for k, r in history:
            print(f"    iter {k:2d}: size of tangent-space update = {r:.2e}")
    return mu


def max_pairwise_error(B, T):
    """Worst-case error of implied pairwise transforms B_j^{-1} B_i vs. truth T_j T_i^{-1}."""
    errs = [(diff_mm(inv(B[j]) @ B[i], T[j] @ inv(T[i])), diff_deg(inv(B[j]) @ B[i], T[j] @ inv(T[i])))
            for i in range(N) for j in range(N) if i != j]
    return max(e[0] for e in errs), max(e[1] for e in errs)


def header(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# ---------------------------------------------------------------- the group
T = [random_rigid() for _ in range(N)]   # random augmentations applied to each image
A = [inv(t) for t in T]                   # each image -> template frame

# ---------------------------------------------------------------- 1. naive average
header("1. Averaging the matrices directly does not give a rotation")
M = np.mean(A, axis=0)
R = M[:3, :3]
print(f"  det(mean R)          = {np.linalg.det(R):.4f}   (a rotation has det = 1)")
print(f"  ||R^T R - I||        = {np.linalg.norm(R.T @ R - np.eye(3)):.4f}   (a rotation has 0)")

# ---------------------------------------------------------------- 2. log-mean subtraction
header("2. Subtracting the mean in log space breaks the pairwise transforms")
P = [log_(a) for a in A]
Pbar = np.mean(P, axis=0)
B_sub = [expm(p - Pbar) for p in P]
mm, deg = max_pairwise_error(B_sub, T)
print(f"  worst pairwise error: {mm:.2f} mm RMS, {deg:.2f} deg")
print("  -> exp(p_i - pbar) != exp(-pbar) exp(p_i) because rotations do not commute.")

# ---------------------------------------------------------------- 3. one step vs iterated
header("3. One-step log-Euclidean mean vs. iterated barycenter")
print("  Iteration history (update size -> 0 means the barycenter condition is satisfied):")
mu = barycenter(A, iters=6, verbose=True)
mu_1step = expm(np.mean([log_(a) for a in A], axis=0))
print(f"\n  one-step mean vs. converged barycenter: {diff_mm(mu_1step, mu):.3f} mm RMS, "
      f"{diff_deg(mu_1step, mu):.3f} deg")

# ---------------------------------------------------------------- 4. invariance
header("4. Invariance: move every input by the same G; does the center move with them?")
G = random_rigid()
A_moved = [a @ inv(G) for a in A]  # all images moved by G  =>  T_i -> G T_i,  A_i -> A_i G^{-1}
for name, fn in [("one-step mean", lambda As: expm(np.mean([log_(a) for a in As], axis=0))),
                 ("iterated barycenter", lambda As: barycenter(As))]:
    lhs, rhs = fn(A_moved), fn(A) @ inv(G)
    print(f"  {name:20s}: mean(A_i G^-1) vs mean(A_i) G^-1 -> {diff_mm(lhs, rhs):.2e} mm, "
          f"{diff_deg(lhs, rhs):.2e} deg")
print("  -> the iterated barycenter depends only on the group, not on the world frame.")

# ---------------------------------------------------------------- 5. ground-truth targets
header("5. Ground-truth central targets  A_i* = mu^{-1} T_i^{-1}")
A_star = [inv(mu) @ a for a in A]
mu_star = barycenter(A_star)
mm, deg = max_pairwise_error(A_star, T)
print(f"  barycenter of targets vs identity: {diff_mm(mu_star, np.eye(4)):.2e} mm "
      f"(targets are central)")
print(f"  worst pairwise error            : {mm:.2e} mm, {deg:.2e} deg "
      f"(pairwise transforms preserved exactly)")

# ---------------------------------------------------------------- 6. commuting case
header("6. Sanity check: if transforms commute, one step is already exact")
z = np.array([0.0, 0.0, 1.0])
Tc = []
for _ in range(N):  # rotations about z plus translation along z commute with each other
    t = random_rigid(axis=z)
    t[:3, 3] = [0, 0, rng.uniform(-30, 30)]
    Tc.append(t)
Ac = [inv(t) for t in Tc]
print("  iteration history:")
barycenter(Ac, iters=3, verbose=True)

# ---------------------------------------------------------------- 7. affine
header("7. The same recipe works for affine transforms")
Ta = [random_affine() for _ in range(N)]
Aa = [inv(t) for t in Ta]
print("  iteration history:")
mu_a = barycenter(Aa, iters=6, verbose=True)
Aa_star = [inv(mu_a) @ a for a in Aa]
mm, _ = max_pairwise_error(Aa_star, Ta)
print(f"\n  worst pairwise error of affine targets: {mm:.2e} mm")
