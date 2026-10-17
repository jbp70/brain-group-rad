"""Pure numpy / scipy references for the unit tests (no TensorFlow import).

Kept TF-free so it can also be checked in a Python 3.8 + numpy 1.21 + scipy 1.4 env.
"""
import contextlib
import importlib.util
import io
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEMO = None


def load_module(name, path):
    """Import a module from a file path; returns None if the file does not exist."""
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_demo():
    """scripts/demo_central_transform.py (runs its printed demo once, output suppressed)."""
    global _DEMO
    if _DEMO is None:
        with contextlib.redirect_stdout(io.StringIO()):
            _DEMO = load_module("demo_central_transform",
                                os.path.join(REPO, "scripts", "demo_central_transform.py"))
    return _DEMO


def index_to_mm(shape, spacing):
    """V (4x4) with x = (p - (n - 1) / 2) * s, built independently of models.transforms."""
    s = np.broadcast_to(np.asarray(spacing, np.float64), (3,))
    c = (np.asarray(shape, np.float64) - 1.0) / 2.0
    v = np.eye(4)
    v[:3, :3] = np.diag(s)
    v[:3, 3] = -c * s
    return v


def voxel_matrix(m_mm, shape, spacing):
    v = index_to_mm(shape, spacing)
    return np.linalg.inv(v) @ m_mm @ v


def block_average(vol, f):
    """Block average the 3 spatial axes of [X, Y, Z, ...] by factor f."""
    x, y, z = vol.shape[:3]
    rest = vol.shape[3:]
    r = vol.reshape((x // f, f, y // f, f, z // f, f) + rest)
    return r.mean(axis=(1, 3, 5))


def smooth_phantom(shape, rng, n_blobs=6, sigma=4.0):
    """Asymmetric sum of Gaussian blobs (voxel units) plus a gentle ramp; values ~[0, 1.5]."""
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    vol = 0.02 * grids[0] / shape[0] + 0.05 * grids[1] / shape[1]
    for _ in range(n_blobs):
        c = [rng.uniform(0.25 * n, 0.75 * n) for n in shape]
        sg = sigma * rng.uniform(0.8, 1.5)
        d2 = sum((g - ci) ** 2 for g, ci in zip(grids, c))
        vol = vol + rng.uniform(0.3, 1.0) * np.exp(-d2 / (2 * sg ** 2))
    return vol


def decode_10d_np(v):
    """Re-derivation of the 10D decode in numpy (adjugate column with the largest diagonal)."""
    v = np.asarray(v, np.float64)
    a00, a01, a02, a03, a11, a12, a13, a22, a23, a33 = v
    a = np.array([[a00, a01, a02, a03], [a01, a11, a12, a13],
                  [a02, a12, a22, a23], [a03, a13, a23, a33]])
    adj = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            minor = np.delete(np.delete(a, i, axis=0), j, axis=1)
            adj[j, i] = (-1) ** (i + j) * np.linalg.det(minor)
    k = int(np.argmax(np.diag(adj)))
    q = adj[:, k]
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def second_moment_bruteforce(mask, spacing):
    """Triple loop over voxels: mean of h h^T with h = [(p - c) * s, 1]."""
    s = np.broadcast_to(np.asarray(spacing, np.float64), (3,))
    c = (np.asarray(mask.shape, np.float64) - 1.0) / 2.0
    acc = np.zeros((4, 4))
    count = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i, j, k]:
                    h = np.array([(i - c[0]) * s[0], (j - c[1]) * s[1], (k - c[2]) * s[2], 1.0])
                    acc += np.outer(h, h)
                    count += 1
    return acc / count


def mean_sq_displacement_bruteforce(d_a, mask, spacing):
    """Mean over mask voxels of |dA h|^2 (mm^2)."""
    s = np.broadcast_to(np.asarray(spacing, np.float64), (3,))
    c = (np.asarray(mask.shape, np.float64) - 1.0) / 2.0
    idx = np.argwhere(mask)
    h = np.c_[(idx - c) * s, np.ones(len(idx))]
    disp = h @ np.asarray(d_a, np.float64).T
    return float(np.mean(np.sum(disp ** 2, axis=1)))


if __name__ == "__main__":
    # Self-check of the TF-free pieces (used by run_tests_wsl.sh in the Python 3.8 env).
    rng = np.random.default_rng(0)
    demo = load_demo()
    mask = rng.uniform(size=(5, 6, 4)) > 0.5
    s_bf = second_moment_bruteforce(mask, 2.0)
    d_a = rng.normal(size=(4, 4))
    d_a[3] = 0
    lhs = np.trace(d_a @ s_bf @ d_a.T)
    rhs = mean_sq_displacement_bruteforce(d_a, mask, 2.0)
    assert abs(lhs - rhs) < 1e-9 * max(1.0, abs(rhs)), (lhs, rhs)
    r = decode_10d_np([0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
    assert np.allclose(r, np.eye(3))
    vol = smooth_phantom((8, 8, 8), rng)
    assert block_average(vol, 2).shape == (4, 4, 4)
    t = [demo.random_rigid() for _ in range(4)]
    mu = demo.barycenter([np.linalg.inv(x) for x in t])
    assert np.allclose(mu[3], [0, 0, 0, 1])
    print("np_reference self-check OK")
