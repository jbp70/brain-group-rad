"""SynthMorph keypoints + BrainMorph-style groupwise mean-keypoint registration (rigid / affine).

SynthMorph's affine and rigid models ("Detector", Hoffmann et al. 2024, Imaging Neuroscience,
https://doi.org/10.1162/imag_a_00197; voxelmorph ``VxmAffineFeatureDetector``) pass ONE image at a
time through a CNN that outputs k = 64 non-negative feature maps (8^3 voxels for extent 256: half-
resolution input, then 4 max-pools; barycenters are continuous). Each map
gives a keypoint (its barycenter) and a channel power (its sum). A pair is registered by a
weighted least-squares (WLS) affine fit between the two keypoint sets, weights = normalised power
of the moving map x normalised power of the fixed map, symmetrised as
0.5 * (fit(1 <- 2) + fit(2 <- 1)^-1); the rigid model keeps only the rotation (QR factor of the
3x3 part) and the translation of that fit.

This wrapper extracts every subject's keypoints and powers with the trained FreeSurfer models
(synthmorph.affine.2.h5 / synthmorph.rigid.1.h5), then runs BrainMorph's groupwise loop
(keymorph ``groupwise_register``, 5 iterations): template = mean keypoints, fit each subject to the
template with SynthMorph's own fit, move its keypoints, repeat; the final transform of subject i
is one fit from its original keypoints to the final template. Weighting (default on, as the
models were trained with WLS; --unweighted for ordinary LS): template keypoint k = power-weighted
mean over subjects, fit weight of subject i = q_ik * mean_j q_jk (q = per-image normalised power),
i.e. the pairwise weight with the group's mean power in place of the fixed image's.

Outputs (``<out>/<align>/``, same layout and convention as baselines/brainmorph/run_brainmorph.py,
``--components xyz --signs +++ --units mm``): ``<subject_dir>_fwd.nii.gz`` (atlas grid,
x + fwd(x) = atlas -> subject), ``_inv.nii.gz`` (exact inverse), ``_moved.nii.gz``,
``_moved_seg.nii.gz``, ``atlas.nii.gz``, ``atlas_seg.nii.gz``, ``matrices.npz`` (4x4 fwd/inv in
voxel and world mm), ``keypoints.npz`` (keypoints in network / processed-voxel / world coords,
powers, fit weights, template), ``pair_check.json``.

Geometry: as mri_synthmorph, each image is resampled to SynthMorph's network space (1 mm, LIA,
extent^3, centred on the image's field of view; synthmorph.registration.network_space) and min-max
normalised. The detector sees that image downsampled by 2 (half-res voxel j = full-res voxel 2j)
and max-pools 4 times, so feature voxel j covers full-res voxels centred at n = f j + f/2 - 1
(f = extent / 8 = 32) and a barycenter J maps to n = f J + f/2 - 1. (SynthMorph's own fit uses
n + 0.5 through its centring constants; pair_check reproduces that exactly, the groupwise fit uses
n.) All subjects must share one processed grid; the template space is the network
space of that grid, mapped back onto the processed grid. Fits run in the centred network frame
(n - (extent - 1) / 2), like SynthMorph's.

Run with FreeSurfer's python (TensorFlow 2.13, voxelmorph, neurite, surfa):
    fspython baselines/synthmorph/run_synthmorph_groupwise.py --self-test
    fspython baselines/synthmorph/run_synthmorph_groupwise.py --processed-dir P --out O --aligns rigid affine
See slurm/synthmorph_groupwise.sh.
"""
import argparse
import json
import os
import sys
import tempfile
import time

import nibabel as nib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from evaluation import fields as F  # noqa: E402
from evaluation.metrics import MajorityVote  # noqa: E402
from baselines.brainmorph.run_brainmorph import find_subjects, load_vol, save_nii  # noqa: E402

MODEL_WEIGHTS = {"affine": "synthmorph.affine.2.h5", "rigid": "synthmorph.rigid.1.h5"}
tf = vxm = sf = smreg = None


def import_freesurfer():
    """TensorFlow / voxelmorph / surfa / FreeSurfer's synthmorph package (as mri_synthmorph)."""
    global tf, vxm, sf, smreg
    os.environ.setdefault("NEURITE_BACKEND", "tensorflow")
    os.environ.setdefault("VXM_BACKEND", "tensorflow")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as _tf
    import voxelmorph as _vxm
    import surfa as _sf
    from synthmorph import registration as _smreg
    tf, vxm, sf, smreg = _tf, _vxm, _sf, _smreg
    # TF32 would degrade the convolutions on Ampere GPUs (our A100 rule); no effect on RTX 8000.
    tf.config.experimental.enable_tensor_float_32_execution(False)


# ------------------------------------------------------------------ SynthMorph fit (numpy, float64)
def sq(m):
    out = np.eye(4)
    out[:3, :] = m[:3, :]
    return out


def fit_affine(x_source, x_target, w=None):
    """voxelmorph utils.fit_affine: mat [3, 4] with x_source ~= mat @ [x_target; 1]."""
    x = np.concatenate([x_target, np.ones((x_target.shape[0], 1))], axis=1)
    xt = x.T if w is None else x.T * w[None, :]
    beta = np.linalg.inv(xt @ x) @ xt @ x_source
    return beta.T


def rigid_part(m):
    """Rotation and translation of an affine as voxelmorph's affine_matrix_to_params (first 6
    params): A = R @ (S @ E) with S @ E = chol(A^T A)^T (first scale negated if det A < 0)."""
    a = m[:3, :3]
    upper = np.linalg.cholesky(a.T @ a).T
    if np.linalg.det(a) < 0:
        upper[0, :] *= -1
    out = np.eye(4)
    out[:3, :3] = a @ np.linalg.inv(upper)
    out[:3, 3] = m[:3, 3]
    return out


def sm_fit(src, tgt, w, rigid):
    """SynthMorph's symmetric WLS fit, centred frame: 4x4 map tgt points -> src points."""
    a1 = fit_affine(src, tgt, w)
    a2 = fit_affine(tgt, src, w)
    m = 0.5 * (sq(a1) + np.linalg.inv(sq(a2)))
    m[3] = (0, 0, 0, 1)
    return rigid_part(m) if rigid else m


def apply(m, x):
    return x @ m[:3, :3].T + m[:3, 3]


def group_weights(power):
    """power [N, K] -> (q per-image normalised power, fit weights eps [N, K])."""
    q = power / np.maximum(power.sum(1, keepdims=True), 1e-30)
    eps = q * q.mean(0, keepdims=True)
    return q, eps / np.maximum(eps.sum(1, keepdims=True), 1e-30)


def template(curr, q=None):
    if q is None:
        return curr.mean(0)
    return (q[..., None] * curr).sum(0) / np.maximum(q.sum(0)[:, None], 1e-30)


def groupwise(points, rigid, num_iters=5, power=None):
    """BrainMorph's groupwise loop with SynthMorph's fit. points [N, K, 3] (centred frame).
    Returns (final template [K, 3], aligned points [N, K, 3], fit weights or None,
    per-subject 4x4 fwd maps template -> subject fitted on the ORIGINAL points)."""
    q, eps = group_weights(power) if power is not None else (None, None)
    curr = points.copy()
    tmpl = None
    for _ in range(num_iters):
        tmpl = template(curr, q)
        new = np.empty_like(curr)
        for i in range(len(curr)):
            m = sm_fit(curr[i], tmpl, None if eps is None else eps[i], rigid)
            new[i] = apply(np.linalg.inv(m), curr[i])
        curr = new
    fwd = [sm_fit(points[i], tmpl, None if eps is None else eps[i], rigid) for i in range(len(points))]
    return tmpl, curr, eps, fwd


# ------------------------------------------------------------------ network
class Detector:
    """Trained SynthMorph Detector; keypoints() returns (n [K, 3] network-space indices, power [K],
    raw feature-grid barycenters J) for one image in network space."""

    def __init__(self, align, extent, weights=None):
        self.align, self.extent = align, int(extent)
        self.shape = (self.extent,) * 3
        if weights is None:
            fs = os.environ.get("FREESURFER_HOME")
            if not fs:
                raise SystemExit("set FREESURFER_HOME (or pass --weights-rigid / --weights-affine)")
            weights = os.path.join(fs, "models", MODEL_WEIGHTS[align])
        self.weights = weights
        self.model = vxm.networks.VxmAffineFeatureDetector(
            in_shape=self.shape, bidir=True, make_dense=False, rigid=align == "rigid", return_feat=True)
        smreg.load_weights(self.model, weights=weights)

    def to_net(self, path):
        """Image file -> (network-space tensor [1, X, Y, Z, 1], net_to_vox 4x4, vox_to_net 4x4)."""
        im = sf.load_volume(path)
        net_to_vox, vox_to_net = smreg.network_space(im, shape=self.shape)
        x = smreg.transform(im, net_to_vox, shape=self.shape, normalize=True, batch=True)
        return x, np.asarray(net_to_vox, np.float64), np.asarray(vox_to_net, np.float64)

    def run(self, x1, x2):
        aff_1, aff_2, feat_1, feat_2 = self.model((x1, x2))
        return np.asarray(aff_1)[0].astype(np.float64), np.asarray(feat_1)[0], np.asarray(feat_2)[0]

    @staticmethod
    def barycenters(feat):
        """feat [h, h, h, K] feature maps -> (J [K, 3] barycenters in feature-voxel indices, power [K])."""
        feat = np.asarray(feat, np.float64)
        power = feat.sum(axis=(0, 1, 2))
        J = np.empty((feat.shape[-1], 3))
        for a in range(3):
            marg = feat.sum(axis=tuple(b for b in range(3) if b != a))  # [h, K]
            J[:, a] = (np.arange(feat.shape[a])[:, None] * marg).sum(0) / np.maximum(power, 1e-30)
        return J, power

    def factor(self, feat):
        """Full-res network voxels per feature voxel (256 / 8 = 32: half-res input, 4 max-pools)."""
        h = feat.shape[0]
        assert feat.shape[:3] == (h, h, h) and self.extent % h == 0, feat.shape
        return self.extent // h

    def keypoints(self, x):
        """Feature voxel j pools half-res voxels [s j, s j + s - 1] (s = f / 2), whose centre is the
        full-res network index f j + f / 2 - 1; barycenter J -> n = f J + f / 2 - 1."""
        _, f1, _ = self.run(x, x)
        J, power = self.barycenters(f1)
        f = self.factor(f1)
        return f * J + f / 2.0 - 1.0, power, J


def pair_check(det, x0, x1, log=print):
    """Reproduce the model's own pairwise matrix (image 0 <- image 1, network index space) from
    our keypoints and weights, with SynthMorph's centring: ne.utils.barycenter(normalize=True,
    shift_center=True) * extent = f (J - (h - 1) / 2) for h^3 feature maps, f = extent / h."""
    aff, f0, f1 = det.run(x0, x1)
    J0, p0 = det.barycenters(f0)
    J1, p1 = det.barycenters(f1)
    e = det.extent
    h, f = f0.shape[0], det.factor(f0)
    c0 = f * (J0 - (h - 1) / 2.0)
    c1 = f * (J1 - (h - 1) / 2.0)
    w = (p0 / p0.sum()) * (p1 / p1.sum())
    m = sm_fit(c0, c1, w, det.align == "rigid")
    cen, un = np.eye(4), np.eye(4)
    cen[:3, 3] = -0.5 * (e - 1)
    un[:3, 3] = 0.5 * (e - 1)
    ours = un @ m @ cen
    err = np.abs(ours[:3] - aff[:3]).max()
    # point error over the network grid corners, in voxels
    corners = np.array([[i, j, k] for i in (0, e - 1) for j in (0, e - 1) for k in (0, e - 1)], float)
    perr = np.abs(apply(ours, corners) - apply(sq(aff), corners)).max()
    ok = perr < 0.05
    log("pair check %s: model vs keypoint refit max matrix diff %.2e, corner diff %.4f vox -> %s"
        % (det.align, err, perr, "PASS" if ok else "FAIL"))
    return {"align": det.align, "max_matrix_diff": float(err), "max_corner_diff_vox": float(perr),
            "model_matrix": aff.tolist(), "refit_matrix": ours[:3].tolist(), "pass": bool(ok)}


# ------------------------------------------------------------------ pipeline
def write_outputs(subs, d, shape, affine, fwd_vox, log=print, tag=""):
    """Dense fields, moved images/labels, atlas, atlas_seg and matrices for 4x4 fwd maps
    (processed voxel coords, atlas -> subject)."""
    os.makedirs(d, exist_ok=True)
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(0))
    g = np.indices(shape, dtype=np.float64).reshape(3, -1).T
    atlas_sum = np.zeros(shape, np.float64)
    labs = []
    t0 = time.time()
    for i, s in enumerate(subs):
        m = fwd_vox[i]
        fwd = (apply(m, g) - g).reshape(shape + (3,)).astype(np.float32)
        inv = (apply(np.linalg.inv(m), g) - g).reshape(shape + (3,)).astype(np.float32)
        img = np.asarray(load_vol(s["image"])[0], np.float32)
        moved = F.warp(img, fwd, order=1)
        atlas_sum += moved
        sp = spacing.astype(np.float32)
        save_nii(fwd * sp, affine, os.path.join(d, s["dir"] + "_fwd.nii.gz"))
        save_nii(inv * sp, affine, os.path.join(d, s["dir"] + "_inv.nii.gz"))
        save_nii(moved, affine, os.path.join(d, s["dir"] + "_moved.nii.gz"))
        if s["labels"]:
            lab = np.rint(np.asarray(load_vol(s["labels"])[0], np.float32)).astype(np.int32)
            ml = F.warp(lab, fwd, order=0)
            labs.append(ml)
            save_nii(ml, affine, os.path.join(d, s["dir"] + "_moved_seg.nii.gz"))
        log("%s %d/%d %s |fwd| mean %.2f mm (%.0f s)" % (tag, i + 1, len(subs), s["dir"],
                                                         float(F.norm_mm(fwd, spacing).mean()), time.time() - t0))
    save_nii((atlas_sum / len(subs)).astype(np.float32), affine, os.path.join(d, "atlas.nii.gz"))
    if len(labs) == len(subs):
        save_nii(MajorityVote(np.stack(labs)).full().astype(np.int32), affine, os.path.join(d, "atlas_seg.nii.gz"))
    fv = np.stack(fwd_vox)
    iv = np.linalg.inv(fv)
    A, Ai = affine, np.linalg.inv(affine)
    np.savez(os.path.join(d, "matrices.npz"), subjects=np.array([s["dir"] for s in subs]), fwd_vox=fv,
             inv_vox=iv, fwd_world=A @ fv @ Ai, inv_world=A @ iv @ Ai,
             note="fwd: atlas -> subject point map (x_subject = M x_atlas), inv = fwd^-1; "
                  "vox = processed-grid voxel coords, world = scanner mm of the grid affine")


def run_align(subs, out, align, extent, num_iters, weighted, weights_file=None, log=print, info=None):
    d = os.path.join(out, align)
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    det = Detector(align, extent, weights_file)
    img0 = nib.load(subs[0]["image"])
    shape, affine = tuple(int(v) for v in img0.shape[:3]), img0.affine
    n_net, powers, nets = [], [], None
    xs01 = []
    for i, s in enumerate(subs):
        im = nib.load(s["image"])
        if tuple(im.shape[:3]) != shape or not np.allclose(im.affine, affine, atol=1e-4):
            raise ValueError("%s: grid differs from the first subject (one processed grid required)" % s["image"])
        x, net_to_vox, vox_to_net = det.to_net(s["image"])
        if nets is None:
            nets = (net_to_vox, vox_to_net)
        elif not np.allclose(nets[0], net_to_vox, atol=1e-6):
            raise ValueError("network space differs between subjects")
        if i < 2:
            xs01.append(x)
        n, p, _ = det.keypoints(x)
        n_net.append(n)
        powers.append(p)
        log("%s keypoints %d/%d %s (%.0f s)" % (align, i + 1, len(subs), s["dir"], time.time() - t0))
    check = pair_check(det, xs01[0], xs01[1], log) if len(xs01) == 2 else None
    with open(os.path.join(d, "pair_check.json"), "w") as f:
        json.dump(check, f, indent=2)
    if check is not None and not check["pass"]:
        raise SystemExit("pair check FAILED: keypoint extraction does not reproduce the model's own fit")
    n_net, powers = np.stack(n_net), np.stack(powers)
    c = 0.5 * (extent - 1)
    tmpl, aligned, eps, fwd_c = groupwise(n_net - c, align == "rigid", num_iters, powers if weighted else None)
    net_to_vox, vox_to_net = nets
    cen, un = np.eye(4), np.eye(4)
    cen[:3, 3], un[:3, 3] = -c, c
    fwd_vox = [net_to_vox @ un @ m @ cen @ vox_to_net for m in fwd_c]  # processed voxel: atlas -> subject
    vox = lambda pts: apply(net_to_vox, pts)  # noqa: E731
    q = group_weights(powers)[0]
    np.savez(os.path.join(d, "keypoints.npz"), subjects=np.array([s["dir"] for s in subs]),
             points_net=n_net.astype(np.float32), points_vox=vox(n_net).astype(np.float32),
             points_world_mm=apply(affine, vox(n_net)).astype(np.float32), power=powers, power_norm=q,
             fit_weights=eps if eps is not None else np.full(powers.shape, 1.0 / powers.shape[1]),
             weighted=weighted, template_vox=vox(tmpl + c).astype(np.float32),
             aligned_vox=vox(aligned + c).astype(np.float32), net_to_vox=net_to_vox, grid_affine=affine,
             extent=extent)
    t_kp = time.time() - t0
    write_outputs(subs, d, shape, affine, fwd_vox, log, align)
    meta = dict(info or {})
    meta.update(align=align, model_weights=det.weights, n_subjects=len(subs), num_iters=num_iters,
                weighted=weighted, extent=extent, num_keypoints=int(n_net.shape[1]), grid=list(shape),
                keypoints_s=t_kp, total_s=time.time() - t0, pair_check=check,
                convention="--components xyz --signs +++ --units mm")
    with open(os.path.join(d, "run_info.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log("%s done (%.1f min)" % (align, (time.time() - t0) / 60))


# ------------------------------------------------------------------ self-test
def self_test():
    """No weights / data: numpy fit vs voxelmorph's TF functions, groupwise recovery of known
    transforms, weighting, and the output convention (run evaluation check-convention after)."""
    rng = np.random.RandomState(1)
    ok = True

    def check(name, cond, val):
        nonlocal ok
        ok &= bool(cond)
        print("%-62s %s  (%s)" % (name, "PASS" if cond else "FAIL", val))

    def rand_affine(scale=0.1, shift=5.0):
        m = np.eye(4)
        m[:3, :3] += rng.uniform(-scale, scale, (3, 3))
        m[:3, 3] = rng.uniform(-shift, shift, 3)
        return m

    # 1. numpy vs voxelmorph (TF) for fit_affine and the rigid projection
    K = 64
    a = rng.uniform(-60, 60, (K, 3))
    b = apply(rand_affine(), a) + rng.normal(0, 1, (K, 3))
    w = rng.uniform(0.1, 1, K)
    w /= w.sum()
    ref = vxm.utils.fit_affine(tf.constant(a[None], tf.float32), tf.constant(b[None], tf.float32),
                               weights=tf.constant(w[None], tf.float32)).numpy()[0]
    check("fit_affine numpy == voxelmorph", np.abs(fit_affine(a, b, w) - ref).max() < 1e-3,
          "%.1e" % np.abs(fit_affine(a, b, w) - ref).max())
    m = rand_affine(0.2)
    par = vxm.utils.affine_matrix_to_params(tf.constant(m[None, :3], tf.float32))[:, :6]
    ref = vxm.utils.params_to_affine_matrix(par, ndims=3).numpy()[0]
    check("rigid projection numpy == voxelmorph params[:6]", np.abs(rigid_part(m)[:3] - ref).max() < 1e-4,
          "%.1e" % np.abs(rigid_part(m)[:3] - ref).max())
    r = rigid_part(m)[:3, :3]
    check("rigid projection is a rotation", abs(np.linalg.det(r) - 1) < 1e-9 and np.abs(r @ r.T - np.eye(3)).max() < 1e-9,
          "%.6f" % np.linalg.det(r))

    # 2. groupwise on keypoints related by known affines: pairwise maps must be recovered exactly
    N = 6
    base = rng.uniform(-50, 50, (K, 3))
    truth = [rand_affine() for _ in range(N)]  # base -> subject
    pts = np.stack([apply(t, base) for t in truth])
    for rigid in (False, True):
        tmpl, _, _, fwd = groupwise(pts, rigid, 5)
        pair = np.abs(apply(fwd[1] @ np.linalg.inv(fwd[0]), pts[0]) - pts[1]).max()
        if not rigid:
            check("affine groupwise: subject 0 -> 1 via template exact [vox]", pair < 1e-6, "%.1e" % pair)
        else:
            check("rigid groupwise: maps are rigid", all(abs(np.linalg.det(f[:3, :3]) - 1) < 1e-9 for f in fwd), "")
    # pure rigid truth: rigid groupwise exact
    truth_r = [rigid_part(rand_affine(0.15)) for _ in range(N)]
    pts_r = np.stack([apply(t, base) for t in truth_r])
    _, _, _, fwd = groupwise(pts_r, True, 5)
    pair = np.abs(apply(fwd[2] @ np.linalg.inv(fwd[0]), pts_r[0]) - pts_r[2]).max()
    check("rigid groupwise on rigid data: subject 0 -> 2 exact [vox]", pair < 1e-6, "%.1e" % pair)

    # 3. weighting: one corrupted low-power keypoint
    bad = pts.copy()
    bad[:, 0] += rng.uniform(-30, 30, (N, 3))
    power = np.ones((N, K))
    power[:, 0] = 1e-6
    errs = {}
    for name, pw in (("unweighted", None), ("weighted", power)):
        _, _, _, fwd = groupwise(bad, False, 5, pw)
        errs[name] = np.abs(apply(fwd[1] @ np.linalg.inv(fwd[0]), pts[0, 1:]) - pts[1, 1:]).mean()
    check("weighted: low-power outlier ignored (%.4f vs unweighted %.4f vox)" % (errs["weighted"], errs["unweighted"]),
          errs["weighted"] < 0.01 < errs["unweighted"], "")

    # 4. write path + convention on a synthetic processed grid (identity affine, 1 mm)
    shape = (40, 48, 36)
    tmp = tempfile.mkdtemp(prefix="sm_selftest_")
    proc, out = os.path.join(tmp, "processed"), os.path.join(tmp, "out")
    xx = np.indices(shape).astype(np.float32)
    blob = np.zeros(shape, np.float32)
    for _ in range(25):
        ctr = rng.uniform(6, np.array(shape) - 6)
        blob += np.exp(-((xx - ctr[:, None, None, None]) ** 2).sum(0) / (2 * rng.uniform(2, 5) ** 2))
    lab0 = np.digitize(blob, np.quantile(blob, [0.5, 0.7, 0.85, 0.95])).astype(np.int32)
    ctr = (np.array(shape) - 1) / 2.0
    subs, maps = [], []
    for i in range(4):
        A = rand_affine(0.12, 5)
        A[:3, 3] += ctr - A[:3, :3] @ ctr  # subject -> base, about the grid centre
        disp = np.moveaxis(np.einsum("ij,j...->i...", A[:3, :3], xx) + A[:3, 3, None, None, None] - xx, 0, -1)
        dd = os.path.join(proc, "S%d_T1w" % i)
        os.makedirs(dd)
        save_nii(F.warp(blob, disp, 1), np.eye(4), os.path.join(dd, "image.nii.gz"))
        save_nii(F.warp(lab0, disp, 0), np.eye(4), os.path.join(dd, "labels_dkt31aseg.nii.gz"))
        subs.append({"dir": "S%d_T1w" % i, "image": os.path.join(dd, "image.nii.gz"),
                     "labels": os.path.join(dd, "labels_dkt31aseg.nii.gz")})
        maps.append(np.linalg.inv(A))  # base -> subject
    kp = rng.uniform(6, np.array(shape) - 6, (40, 3))
    pts = np.stack([apply(m, kp) for m in maps])
    _, _, _, fwd = groupwise(pts, False, 5, np.ones((4, 40)))
    write_outputs(subs, os.path.join(out, "affine"), shape, np.eye(4), fwd, log=lambda *_: None)
    d = os.path.join(out, "affine")
    check("files written (4 x 4 + atlas + atlas_seg + matrices)", len(os.listdir(d)) == 19, len(os.listdir(d)))
    f1 = np.asarray(nib.load(os.path.join(d, "S1_T1w_fwd.nii.gz")).dataobj)
    i1 = np.asarray(nib.load(os.path.join(d, "S1_T1w_inv.nii.gz")).dataobj)
    def inside(fw):  # atlas voxels whose image under fwd lies inside the subject grid (1-voxel margin)
        y = np.moveaxis(xx, 0, -1) + fw
        return np.all((y >= 1) & (y <= np.array(shape) - 2), axis=-1)

    e = F.norm_mm(F.compose(f1, i1), (1, 1, 1))[inside(f1)].max()
    check("|compose(fwd, inv)| max where fwd stays in the grid [mm]", e < 1e-3, "%.1e" % e)
    mv = [np.asarray(nib.load(os.path.join(d, "S%d_T1w_moved.nii.gz" % i)).dataobj) for i in (0, 2)]
    f0 = np.asarray(nib.load(os.path.join(d, "S0_T1w_fwd.nii.gz")).dataobj)
    f2 = np.asarray(nib.load(os.path.join(d, "S2_T1w_fwd.nii.gz")).dataobj)
    msk = inside(f0) & inside(f2) & (mv[0] > 0.2 * mv[0].max())
    ncc = np.corrcoef(mv[0][msk], mv[1][msk])[0, 1]
    check("moved S0 vs moved S2 NCC", ncc > 0.98, round(float(ncc), 4))
    print("self-test files: %s" % out)
    print("SELF-TEST %s" % ("PASSED" if ok else "FAILED"))
    return ok, proc, out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true", help="synthetic checks, no weights / data needed")
    ap.add_argument("--processed-dir", help="<subject>_T1w/image.nii.gz folders on one common grid")
    ap.add_argument("--label-name", default="dkt31aseg", help="labels_<name>.nii.gz (optional per subject)")
    ap.add_argument("--subjects-file", default=None)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--out", help="output folder (one subfolder per alignment)")
    ap.add_argument("--aligns", nargs="+", default=["rigid", "affine"], choices=("rigid", "affine"))
    ap.add_argument("--num-iters", type=int, default=5)
    ap.add_argument("--extent", type=int, default=256, choices=(192, 256), help="mri_synthmorph -e")
    ap.add_argument("--unweighted", action="store_true", help="ordinary LS instead of power-weighted LS")
    ap.add_argument("--weights-rigid", default=None, help="default $FREESURFER_HOME/models/synthmorph.rigid.1.h5")
    ap.add_argument("--weights-affine", default=None, help="default $FREESURFER_HOME/models/synthmorph.affine.2.h5")
    a = ap.parse_args()

    import_freesurfer()
    print("tensorflow %s, GPUs %s" % (tf.__version__, [g.name for g in tf.config.list_physical_devices("GPU")]))
    if a.self_test:
        ok = self_test()[0]
        sys.exit(0 if ok else 1)
    for k in ("processed_dir", "out"):
        if not getattr(a, k):
            ap.error("--%s is required" % k.replace("_", "-"))
    subs = find_subjects(a.processed_dir, a.label_name, a.subjects_file)
    if a.max_subjects:
        subs = subs[:a.max_subjects]
    if len(subs) < 2:
        raise SystemExit("need at least 2 subjects in %s" % a.processed_dir)
    if any(not s["labels"] for s in subs):
        print("WARNING: labels_%s missing for some subjects -> no moved_seg / atlas_seg" % a.label_name)
    info = {"processed_dir": os.path.abspath(a.processed_dir), "tensorflow": tf.__version__,
            "voxelmorph": os.path.dirname(vxm.__file__),
            "weighting": "ordinary LS" if a.unweighted else
            "power-weighted LS: template = power-weighted mean, weight = subject x group-mean normalised power"}
    for align in a.aligns:
        wf = a.weights_rigid if align == "rigid" else a.weights_affine
        run_align(subs, a.out, align, a.extent, a.num_iters, not a.unweighted, wf, info=info)
    print("done: %s" % a.out)


if __name__ == "__main__":
    main()
