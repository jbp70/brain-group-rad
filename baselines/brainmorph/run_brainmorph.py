"""BrainMorph groupwise baseline on the processed grid, in the same file format as MultiMorph.

BrainMorph (Wang et al., MELBA 2025, https://github.com/alanqrwang/brainmorph) extracts K keypoints
per image with a keypoint network, then registers the group by iterating "mean keypoints ->
fit each subject to the mean" (num_iters 5, unweighted, as in keymorph's ``groupwise_register``).
The final transform of subject i is ONE fit from its original keypoints to the final mean.
We reproduce that loop with keymorph's own aligner classes (RigidKeypointAligner,
AffineKeypointAligner, TPS) in BrainMorph's normalised coordinates, then evaluate the
transforms on our processed grid instead of resampling BrainMorph's 256^3 grids.

Outputs per alignment type (``<out>/<align>/``, e.g. rigid, affine, tps_0), in the canonical
convention of ``evaluation/fields.py`` (``--components xyz --signs +++ --units mm``):

    atlas.nii.gz, atlas_seg.nii.gz               mean of moved images / majority vote of moved labels
    <subject_dir>_fwd.nii.gz   [X,Y,Z,3] mm       on the atlas grid: x + fwd(x) = atlas -> subject
    <subject_dir>_inv.nii.gz   [X,Y,Z,3] mm       on the subject grid: y + inv(y) = subject -> atlas
    <subject_dir>_moved.nii.gz / _moved_seg.nii.gz  subject image / labels pulled into atlas space
    matrices.npz (rigid/affine)                   4x4 fwd/inv matrices, voxel and world (mm) coords

``<out>/keypoints.npz``: raw keypoints of every subject (normalised and processed-grid voxel
coordinates, world mm), plus per alignment the final mean keypoints and the aligned keypoints.

Directions: fwd uses BrainMorph's own pulling transform (fit mean keypoints -> subject keypoints,
which is what its grid_sample grid encodes). inv is the exact matrix inverse for rigid/affine.
A TPS has no closed-form inverse, so inv is BrainMorph's own forward fit (subject -> mean,
control points at the subject keypoints): both fits agree at the keypoints (exactly for lambda 0),
not in between; the evaluator's inverse-consistency metric measures the difference.

Geometry: BrainMorph pads every 1 mm RAS image to 256^3 (torchio CropOrPad, centred) and min-max
normalises it. Keypoints come from a centre-of-mass layer in 'ij' (array-axis) order with
p = 2 * idx / (256 - 1) - 1 (linspace(-1, 1) convention). So processed-grid voxel
x = (p + 1) * 255 / 2 - pad_lo exactly. (BrainMorph's own warping calls grid_sample with
align_corners=False on these align_corners=True coordinates, a ~0.4 % zoom; we do not copy that.)

Environment: pt_nv.sif (PyTorch) + a clone of https://github.com/alanqrwang/keymorph passed with
--keymorph-dir; no pip installs (torchio / h5py are only imported by keymorph modules we do not
use and are stubbed when missing). Weights: foundation-numkey512-numlevels6.pth.tar = variant L,
512 keypoints. See slurm/brainmorph_mindboggle.sh for the cluster commands.

    python baselines/brainmorph/run_brainmorph.py --self-test --keymorph-dir <clone>
    python baselines/brainmorph/run_brainmorph.py --processed-dir P --out O \\
        --keymorph-dir K --weights W --aligns rigid affine tps_0
"""
import argparse
import json
import os
import sys
import tempfile
import time
import types

import nibabel as nib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from evaluation import fields as F  # noqa: E402  (numpy/scipy only)
from evaluation.metrics import MajorityVote  # noqa: E402

torch = None  # imported in import_keymorph()
KM = types.SimpleNamespace()


# ------------------------------------------------------------------ keymorph import
def import_keymorph(keymorph_dir):
    """Import torch and the keymorph modules we use; stub optional modules keymorph imports at
    package level but only needs for plotting / h5 checkpoint helpers."""
    global torch
    import torch as _torch
    torch = _torch
    if keymorph_dir:
        keymorph_dir = os.path.abspath(keymorph_dir)
        if not os.path.isdir(os.path.join(keymorph_dir, "keymorph")):
            raise FileNotFoundError("%s has no keymorph/ package (clone of alanqrwang/keymorph)" % keymorph_dir)
        sys.path.insert(0, keymorph_dir)
    stubbed = []
    for name in ("torchio", "h5py"):
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)
            stubbed.append(name)
    from keymorph.keypoint_aligners import RigidKeypointAligner, AffineKeypointAligner, TPS
    from keymorph.layers import CenterOfMass3d
    import keymorph
    KM.Rigid, KM.Affine, KM.TPS, KM.CoM = RigidKeypointAligner, AffineKeypointAligner, TPS, CenterOfMass3d
    KM.path = os.path.dirname(os.path.abspath(keymorph.__file__))
    KM.stubbed = stubbed
    return KM


def build_network(weights, num_keypoints, num_levels, device):
    """BrainMorph keypoint model exactly as scripts/register.py builds it (backbone
    'truncatedunet', 1 truncated layer, f_maps 32, gcr, 8 groups, CoM layer 'ij')."""
    from keymorph.unet3d.model import TruncatedUNet3D
    from keymorph.model import KeyMorph
    net = TruncatedUNet3D(1, num_keypoints, 1, final_sigmoid=False, f_maps=32, layer_order="gcr",
                          num_groups=8, num_levels=num_levels, is_segmentation=False, conv_padding=1)
    net = torch.nn.DataParallel(net)
    model = KeyMorph(net, num_keypoints, 3, keypoint_layer="com", use_amp=False, use_checkpoint=False,
                     weight_keypoints="power")
    try:
        state = torch.load(weights, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13
        state = torch.load(weights, map_location="cpu")
    sd = state["state_dict"] if "state_dict" in state else state
    # brainmorph scripts/script_utils.load_checkpoint: strip ".backbone", load into the backbone
    candidates = [{k.replace(".backbone", ""): v for k, v in sd.items()}]
    candidates.append({("module." + k if not k.startswith("module.") else k): v
                       for k, v in ((k[len("backbone."):] if k.startswith("backbone.") else k, v)
                                    for k, v in sd.items())})
    err = None
    for cand in candidates:
        try:
            model.backbone.load_state_dict(cand, strict=True)
            err = None
            break
        except RuntimeError as e:
            err = e
    if err is not None:
        raise RuntimeError("could not load %s into the BrainMorph backbone: %s" % (weights, err))
    model.backbone = model.backbone.module  # unwrap DataParallel after loading (single device)
    model.to(device).eval()
    return model


# ------------------------------------------------------------------ geometry
class Geometry:
    """Processed grid (shape, affine, spacing) <-> BrainMorph's padded cube in [-1, 1]."""

    def __init__(self, shape, affine, pad_to=256):
        self.shape = tuple(int(s) for s in shape)
        self.affine = np.asarray(affine, np.float64)
        self.spacing = np.sqrt((self.affine[:3, :3] ** 2).sum(0))
        self.pad_to = int(pad_to)
        diff = [self.pad_to - s for s in self.shape]
        if min(diff) < 0:
            raise ValueError("grid %s larger than %d^3 (BrainMorph would crop)" % (self.shape, self.pad_to))
        # torchio CropOrPad pads ceil(diff/2) at the start (all of our diffs are even anyway)
        self.pad_lo = np.array([int(np.ceil(d / 2.0)) for d in diff], np.float64)
        self.pad_hi = np.array(diff, np.float64) - self.pad_lo
        self.s = 2.0 / (self.pad_to - 1)

    def vox2norm(self, x):
        return (x + self.pad_lo) * self.s - 1.0

    def norm2vox(self, p):
        return (p + 1.0) / self.s - self.pad_lo

    def S(self):
        """4x4 matrix vox -> norm."""
        m = np.eye(4)
        m[:3, :3] *= self.s
        m[:3, 3] = self.pad_lo * self.s - 1.0
        return m

    def vox2world(self, x):
        return x @ self.affine[:3, :3].T + self.affine[:3, 3]

    def pad_image(self, img):
        """Zero-pad to the cube and min-max normalise to [0, 1] (BrainMorph preprocessing)."""
        lo = self.pad_lo.astype(int)
        hi = self.pad_hi.astype(int)
        out = np.pad(img.astype(np.float32), [(lo[a], hi[a]) for a in range(3)])
        mn, mx = float(out.min()), float(out.max())
        return (out - mn) / max(mx - mn, 1e-12)

    def grid_points(self):
        return np.indices(self.shape, dtype=np.float32).reshape(3, -1).T  # [V, 3] voxel coords


# ------------------------------------------------------------------ groupwise registration
def parse_align(s):
    if s in ("rigid", "affine"):
        return s, None
    if s.startswith("tps_"):
        return "tps", float(s[4:])
    raise ValueError("alignment must be rigid, affine or tps_<lambda>, got %r" % s)


def make_aligner(kind, lmbda, points_m, points_f, w=None):
    """keymorph aligner; ``w`` [1, K] keypoint weights summing to 1 (None = unweighted)."""
    if kind == "rigid":
        return KM.Rigid(points_m=points_m, points_f=points_f, w=w, dim=3)
    if kind == "affine":
        return KM.Affine(points_m=points_m, points_f=points_f, w=w, dim=3)
    lm = torch.tensor([float(lmbda)], dtype=torch.float32, device=points_m.device)
    return KM.TPS(points_m=points_m, points_f=points_f, lmbda=lm, w=w, dim=3, use_checkpoint=False)


def group_weights(power):
    """Power weighting (optional, not in the BrainMorph code). power [N, K] channel powers
    (sum of the ReLU'd heatmap). Returns (q [N, K]: per-subject normalised power,
    eps [N, K]: fit weights q_ik * qbar_k normalised per subject, as BrainMorph's / SynthMorph's
    pairwise weights p_moving * p_fixed with the group-mean power as the 'fixed' image)."""
    q = power / power.sum(dim=1, keepdim=True).clamp_min(1e-12)
    qbar = q.mean(dim=0, keepdim=True)
    eps = q * qbar
    return q, eps / eps.sum(dim=1, keepdim=True).clamp_min(1e-12)


def template(curr, q=None):
    """Template keypoints [1, K, 3]: plain mean (BrainMorph) or per-keypoint power-weighted mean."""
    if q is None:
        return curr.mean(dim=0, keepdim=True)
    w = q[..., None]
    return (w * curr).sum(dim=0, keepdim=True) / w.sum(dim=0, keepdim=True).clamp_min(1e-12)


def groupwise(points_norm, align, num_iters, power=None):
    """keymorph KeyMorph.groupwise_register loop. points_norm: tensor [N, K, 3]; power [N, K]
    enables the optional power weighting (None = BrainMorph's unweighted groupwise code).
    Returns (final mean points [1, K, 3], aligned points [N, K, 3])."""
    kind, lmbda = parse_align(align)
    q, eps = group_weights(power) if power is not None else (None, None)
    curr = points_norm.clone()
    mean = None
    for _ in range(num_iters):
        mean = template(curr, q)
        new = torch.zeros_like(curr)
        for i in range(curr.shape[0]):
            w = None if eps is None else eps[i:i + 1]
            al = make_aligner(kind, lmbda, curr[i:i + 1], mean, w)
            new[i:i + 1] = al.get_forward_transformed_points(curr[i:i + 1])
        curr = new
    return mean, curr


class SubjectTransform:
    """Final transform of one subject: fitted from its ORIGINAL keypoints to the final mean.
    fwd_fn: atlas voxel -> subject voxel; inv_fn: subject voxel -> atlas voxel."""

    def __init__(self, align, points_i, mean, geom, device, chunk, w=None):
        self.kind, self.lmbda = parse_align(align)
        self.geom, self.device, self.chunk = geom, device, int(chunk)
        self.w = w
        al = make_aligner(self.kind, self.lmbda, points_i, mean, w)
        if self.kind in ("rigid", "affine"):
            m_norm = al.inverse_transform_matrix[0].detach().double().cpu().numpy()  # mean -> subject
            S = geom.S()
            self.fwd_vox = np.linalg.inv(S) @ m_norm @ S
            self.inv_vox = np.linalg.inv(self.fwd_vox)
        else:
            self.al = al
            self.ctrl_f, self.ctrl_m = mean, points_i
            self.theta_fwd = al.inverse_theta  # fit(points_f -> points_m), control points at mean
            self.theta_inv = al.fit(points_i, mean, al.lmbda, weights=w)  # subject -> mean

    def _apply_tps(self, theta, ctrl, x_vox):
        out = np.empty_like(x_vox, dtype=np.float32)
        with torch.no_grad():
            for a in range(0, x_vox.shape[0], self.chunk):
                q = torch.from_numpy(self.geom.vox2norm(x_vox[a:a + self.chunk].astype(np.float64))
                                     .astype(np.float32))[None].to(self.device)
                r = self.al.transform_points(theta, ctrl, q)[0].double().cpu().numpy()
                out[a:a + self.chunk] = self.geom.norm2vox(r)
        return out

    def fwd_fn(self, x_vox):
        if self.kind == "tps":
            return self._apply_tps(self.theta_fwd, self.ctrl_f, x_vox)
        return (x_vox @ self.fwd_vox[:3, :3].T + self.fwd_vox[:3, 3]).astype(np.float32)

    def inv_fn(self, y_vox):
        if self.kind == "tps":
            return self._apply_tps(self.theta_inv, self.ctrl_m, y_vox)
        return (y_vox @ self.inv_vox[:3, :3].T + self.inv_vox[:3, 3]).astype(np.float32)

    def fields(self):
        """Canonical voxel displacement fields [X, Y, Z, 3] (fwd on atlas grid, inv on subject grid)."""
        g = self.geom.grid_points()
        fwd = (self.fwd_fn(g) - g).reshape(self.geom.shape + (3,))
        inv = (self.inv_fn(g) - g).reshape(self.geom.shape + (3,))
        return fwd.astype(np.float32), inv.astype(np.float32)


# ------------------------------------------------------------------ data
def find_subjects(processed_dir, label_name, subjects_file=None):
    dirs = sorted(d for d in os.listdir(processed_dir)
                  if os.path.isfile(os.path.join(processed_dir, d, "image.nii.gz")))
    if subjects_file:
        with open(subjects_file) as f:
            want = [ln.strip() for ln in f if ln.strip()]
        keep = []
        for w in want:
            m = [d for d in dirs if d == w or d == w + "_T1w"]
            if not m:
                raise FileNotFoundError("subject %s not in %s" % (w, processed_dir))
            keep.append(m[0])
        dirs = keep
    subs = []
    for d in dirs:
        lab = os.path.join(processed_dir, d, "labels_%s.nii.gz" % label_name)
        subs.append({"dir": d, "image": os.path.join(processed_dir, d, "image.nii.gz"),
                     "labels": lab if os.path.isfile(lab) else None})
    return subs


def load_vol(path):
    im = nib.load(path)
    return np.asanyarray(im.dataobj), im.affine


def save_nii(arr, affine, path):
    nib.save(nib.Nifti1Image(arr, affine), path)


# ------------------------------------------------------------------ pipeline
def run(subs, out, aligns, geom, keypoint_fn, device, num_iters=5, chunk=131072, log=print,
        info=None, weighted=False):
    """subs: list of {'dir','image','labels'}; keypoint_fn(i, padded [D,H,W] float32) ->
    (points [1, K, 3] normalised 'ij' keypoints, power [1, K] or None), torch on device.
    weighted: optional power weighting of the groupwise fits (see group_weights)."""
    os.makedirs(out, exist_ok=True)
    n = len(subs)
    t0 = time.time()
    pts, pows = [], []
    for i, s in enumerate(subs):
        img, aff = load_vol(s["image"])
        if img.shape != geom.shape or not np.allclose(aff, geom.affine, atol=1e-4):
            raise ValueError("%s: grid %s / affine differ from the first subject" % (s["image"], img.shape))
        with torch.no_grad():
            p, pw = keypoint_fn(i, geom.pad_image(np.asarray(img, np.float32)))
        pts.append(p.detach().float())
        pows.append(None if pw is None else pw.detach().float())
        log("keypoints %d/%d %s  (%.1f s)" % (i + 1, n, s["dir"], time.time() - t0))
    points = torch.cat(pts, 0)  # [N, K, 3] normalised
    power = None if any(pw is None for pw in pows) else torch.cat(pows, 0)  # [N, K]
    if weighted and power is None:
        raise ValueError("weighted groupwise fit needs keypoint powers")
    p_np = points.double().cpu().numpy()
    kp = {"subjects": np.array([s["dir"] for s in subs]), "points_norm": p_np.astype(np.float32),
          "points_vox": geom.norm2vox(p_np).astype(np.float32),
          "points_world_mm": geom.vox2world(geom.norm2vox(p_np)).astype(np.float32),
          "grid_affine": geom.affine, "pad_lo": geom.pad_lo, "pad_to": geom.pad_to,
          "weighted": weighted}
    eps = None
    if power is not None:
        q, eps_all = group_weights(power)
        kp["power"] = power.cpu().numpy()
        kp["power_norm"] = q.cpu().numpy()
        if weighted:
            eps = eps_all
            kp["fit_weights"] = eps.cpu().numpy()
    t_kp = time.time() - t0

    timings = {"keypoints_s": t_kp}
    for align in aligns:
        ta = time.time()
        d = os.path.join(out, align)
        os.makedirs(d, exist_ok=True)
        mean, aligned = groupwise(points, align, num_iters, power if weighted else None)
        mean_np = mean.double().cpu().numpy()
        kp["%s_mean_vox" % align] = geom.norm2vox(mean_np[0]).astype(np.float32)
        kp["%s_aligned_vox" % align] = geom.norm2vox(aligned.double().cpu().numpy()).astype(np.float32)
        mats = {"fwd_vox": [], "inv_vox": []}
        atlas_sum = np.zeros(geom.shape, np.float64)
        moved_labs = []
        for i, s in enumerate(subs):
            tr = SubjectTransform(align, points[i:i + 1], mean, geom, device, chunk,
                                  None if eps is None else eps[i:i + 1])
            fwd, inv = tr.fields()
            if tr.kind != "tps":
                mats["fwd_vox"].append(tr.fwd_vox)
                mats["inv_vox"].append(tr.inv_vox)
            img = np.asarray(load_vol(s["image"])[0], np.float32)
            moved = F.warp(img, fwd, order=1)
            atlas_sum += moved
            sp = geom.spacing.astype(np.float32)
            save_nii((fwd * sp).astype(np.float32), geom.affine, os.path.join(d, s["dir"] + "_fwd.nii.gz"))
            save_nii((inv * sp).astype(np.float32), geom.affine, os.path.join(d, s["dir"] + "_inv.nii.gz"))
            save_nii(moved, geom.affine, os.path.join(d, s["dir"] + "_moved.nii.gz"))
            if s["labels"]:
                lab = np.rint(np.asarray(load_vol(s["labels"])[0], np.float32)).astype(np.int32)
                ml = F.warp(lab, fwd, order=0)
                moved_labs.append(ml)
                save_nii(ml, geom.affine, os.path.join(d, s["dir"] + "_moved_seg.nii.gz"))
            log("%s %d/%d %s  |fwd| mean %.2f mm  (%.1f s)" % (
                align, i + 1, n, s["dir"], float(F.norm_mm(fwd, geom.spacing).mean()), time.time() - ta))
        save_nii((atlas_sum / n).astype(np.float32), geom.affine, os.path.join(d, "atlas.nii.gz"))
        if len(moved_labs) == n:
            seg = MajorityVote(np.stack(moved_labs)).full()
            save_nii(seg.astype(np.int32), geom.affine, os.path.join(d, "atlas_seg.nii.gz"))
        del moved_labs
        if mats["fwd_vox"]:
            fv, iv = np.stack(mats["fwd_vox"]), np.stack(mats["inv_vox"])
            A, Ai = geom.affine, np.linalg.inv(geom.affine)
            np.savez(os.path.join(d, "matrices.npz"), subjects=kp["subjects"], fwd_vox=fv, inv_vox=iv,
                     fwd_world=A @ fv @ Ai, inv_world=A @ iv @ Ai,
                     note="fwd: atlas -> subject point map (x_subject = M x_atlas), inv = fwd^-1; "
                          "vox = processed-grid voxel coords, world = scanner mm of the grid affine")
        timings[align + "_s"] = time.time() - ta
        log("%s done (%.1f min)" % (align, (time.time() - ta) / 60))
    np.savez(os.path.join(out, "keypoints.npz"), **kp)
    meta = dict(info or {})
    meta.update(n_subjects=n, aligns=list(aligns), num_iters=num_iters, grid=list(geom.shape),
                weighted=weighted,
                pad_lo=geom.pad_lo.tolist(), pad_to=geom.pad_to, timings=timings,
                convention="--components xyz --signs +++ --units mm")
    with open(os.path.join(out, "run_info.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return kp


# ------------------------------------------------------------------ self-test
def self_test(device="cpu"):
    """Synthetic checks without weights or data: CoM axis order, groupwise math, field
    conventions, and the full write path (then run `python -m evaluation check-convention`)."""
    rng = np.random.RandomState(0)
    ok = True

    def check(name, cond, val):
        nonlocal ok
        ok &= bool(cond)
        print("%-58s %s  (%s)" % (name, "PASS" if cond else "FAIL", val))

    shape, P = (40, 48, 36), 64
    geom = Geometry(shape, np.diag([1.0, 1.0, 1.0, 1.0]), pad_to=P)

    # 1. centre-of-mass layer: 'ij' order and linspace(-1, 1) convention
    vol = torch.zeros(1, 1, P, P, P)
    vox = (30, 20, 45)
    vol[0, 0][vox] = 1.0
    p = KM.CoM(indexing="ij")(vol)[0, 0].double().numpy()
    back = (p + 1.0) * (P - 1) / 2.0
    check("CoM keypoint -> padded voxel index (ij order)", np.abs(back - vox).max() < 1e-3, back.round(3))

    # 2. groupwise math on keypoints related by known affines (+ jitter for TPS)
    K, N = 60, 5
    base = rng.uniform([8, 8, 8], np.array(shape) - 8, size=(K, 3))
    c = np.array(shape) / 2.0
    subj = []
    for _ in range(N):
        ang = rng.uniform(-0.15, 0.15, 3)
        cx, cy, cz = np.cos(ang)
        sx, sy, sz = np.sin(ang)
        R = (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
             @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))
        A = R @ np.diag(rng.uniform(0.95, 1.05, 3))
        subj.append((base - c) @ A.T + c + rng.uniform(-2, 2, 3))
    subj = np.stack(subj)
    jit = subj + rng.normal(0, 0.7, subj.shape)
    for align, pts_vox in (("affine", subj), ("tps_0", subj), ("tps_0", jit), ("rigid", subj)):
        pts = torch.from_numpy(geom.vox2norm(pts_vox).astype(np.float32)).to(device)
        mean, _ = groupwise(pts, align, 5)
        mean_vox = geom.norm2vox(mean[0].double().cpu().numpy()).astype(np.float32)
        tag = align + (" jitter" if pts_vox is jit else "")
        err_f, err_i, err_c = [], [], []
        for i in range(N):
            tr = SubjectTransform(align, pts[i:i + 1], mean, geom, device, 4096)
            err_f.append(np.abs(tr.fwd_fn(mean_vox) - pts_vox[i]).max())
            err_i.append(np.abs(tr.inv_fn(pts_vox[i].astype(np.float32)) - mean_vox).max())
            q = rng.uniform(0, np.array(shape) - 1, size=(200, 3)).astype(np.float32)
            err_c.append(np.abs(tr.inv_fn(tr.fwd_fn(q)) - q).max())
            if align == "rigid":
                Rm = tr.fwd_vox[:3, :3]
                ok_r = abs(np.linalg.det(Rm) - 1) < 1e-4 and np.abs(Rm @ Rm.T - np.eye(3)).max() < 1e-4
                if i == 0:
                    check("rigid: fwd matrix is a rotation (vox coords)", ok_r, round(np.linalg.det(Rm), 6))
        if align != "rigid":
            check("%s: fwd(mean kp) == subject kp [vox]" % tag, max(err_f) < 0.02, "%.2e" % max(err_f))
            check("%s: inv(subject kp) == mean kp [vox]" % tag, max(err_i) < 0.02, "%.2e" % max(err_i))
        if pts_vox is jit:
            # not an assertion: the swapped TPS fit is only an approximate inverse between keypoints
            print("%-58s INFO  (max %.2f vox; swapped fits agree only at the keypoints)"
                  % ("%s: inv(fwd(x)) vs x on random points" % tag, max(err_c)))
        else:
            check("%s: inv(fwd(x)) == x on random points [vox]" % tag, max(err_c) < 0.02, "%.2e" % max(err_c))

    # 3. full write path on synthetic subjects (subject = base image pulled through a known affine)
    tmp = tempfile.mkdtemp(prefix="bm_selftest_")
    proc, out = os.path.join(tmp, "processed"), os.path.join(tmp, "out")
    xx = np.indices(shape).astype(np.float32)
    blob = np.zeros(shape, np.float32)
    for _ in range(25):
        ctr = rng.uniform(6, np.array(shape) - 6)
        blob += np.exp(-((xx - ctr[:, None, None, None]) ** 2).sum(0) / (2 * rng.uniform(2, 5) ** 2))
    lab0 = np.digitize(blob, np.quantile(blob, [0.5, 0.7, 0.85, 0.95])).astype(np.int32)
    subs, true_maps = [], []
    for i in range(4):
        A = np.eye(4)
        A[:3, :3] += rng.uniform(-0.12, 0.12, (3, 3))
        A[:3, 3] = rng.uniform(-5, 5, 3) + c - A[:3, :3] @ c  # subject -> base point map
        disp = (np.einsum("ij,j...->i...", A[:3, :3], xx) + A[:3, 3, None, None, None] - xx)
        disp = np.moveaxis(disp, 0, -1)
        img, lab = F.warp(blob, disp, 1), F.warp(lab0, disp, 0)
        dd = os.path.join(proc, "S%d_T1w" % i)
        os.makedirs(dd)
        save_nii(img, np.eye(4), os.path.join(dd, "image.nii.gz"))
        save_nii(lab, np.eye(4), os.path.join(dd, "labels_dkt31aseg.nii.gz"))
        subs.append({"dir": "S%d_T1w" % i, "image": os.path.join(dd, "image.nii.gz"),
                     "labels": os.path.join(dd, "labels_dkt31aseg.nii.gz")})
        true_maps.append(np.linalg.inv(A))  # base -> subject
    kp_base = rng.uniform([6, 6, 6], np.array(shape) - 6, size=(40, 3))

    def kp_fn(i, padded):
        assert padded.shape == (P, P, P) and 0 <= padded.min() and padded.max() <= 1
        M = true_maps[i]
        v = kp_base @ M[:3, :3].T + M[:3, 3]
        return torch.from_numpy(geom.vox2norm(v).astype(np.float32))[None].to(device), None

    # 2b. optional power weighting: one corrupted low-power keypoint must barely matter
    pts_bad = subj.copy()
    pts_bad[:, 0] += rng.uniform(-15, 15, (N, 3))
    power = np.ones((N, K))
    power[:, 0] = 1e-4
    pts_t = torch.from_numpy(geom.vox2norm(pts_bad).astype(np.float32)).to(device)
    pw_t = torch.from_numpy(power.astype(np.float32)).to(device)
    for align in ("rigid", "affine"):
        res = {}
        for name, pw in (("unweighted", None), ("weighted", pw_t)):
            mean, _ = groupwise(pts_t, align, 5, pw)
            eps = None if pw is None else group_weights(pw)[1]
            # transfer error on the clean keypoints 1..K-1 between subjects 0 and 1 via the template
            t0_ = SubjectTransform(align, pts_t[0:1], mean, geom, device, 4096, None if eps is None else eps[0:1])
            t1_ = SubjectTransform(align, pts_t[1:2], mean, geom, device, 4096, None if eps is None else eps[1:2])
            x0 = subj[0, 1:].astype(np.float32)
            res[name] = float(np.abs(t1_.fwd_fn(t0_.inv_fn(x0)) - subj[1, 1:]).mean())
        tol = 0.05 if align == "affine" else 0.5  # rigid: data are affine, so a residual remains
        check("%s weighted: outlier ignored (pair err %.3f vs unweighted %.3f vox)"
              % (align, res["weighted"], res["unweighted"]),
              res["weighted"] < tol and res["weighted"] < res["unweighted"], "")

    run(subs, out, ["affine", "tps_0"], geom, kp_fn, device, chunk=8192, log=lambda *_: None)
    files = sorted(os.listdir(os.path.join(out, "tps_0")))
    check("files written (4 subjects x 4 + atlas + atlas_seg)", len(files) == 18, len(files))
    for align in ("affine", "tps_0"):
        d = os.path.join(out, align)
        fwd = np.asarray(nib.load(os.path.join(d, "S1_T1w_fwd.nii.gz")).dataobj)
        inv = np.asarray(nib.load(os.path.join(d, "S1_T1w_inv.nii.gz")).dataobj)
        e = F.norm_mm(F.compose(fwd, inv), (1, 1, 1))[5:-5, 5:-5, 5:-5]
        check("%s: |compose(fwd, inv)| interior max [mm]" % align, e.max() < 0.05, "%.2e" % e.max())
        # moved images of all subjects should agree (they are the same base image)
        mv = [np.asarray(nib.load(os.path.join(d, "S%d_T1w_moved.nii.gz" % i)).dataobj) for i in range(4)]
        m = blob > 0.2 * blob.max()
        ncc = np.corrcoef(mv[0][m], mv[2][m])[0, 1]
        check("%s: moved S0 vs moved S2 NCC" % align, ncc > 0.98, round(float(ncc), 4))
    print("self-test files: %s" % out)
    print("convention check (run with the repo's python):\n  python -m evaluation check-convention "
          "--processed-dir %s --n-subjects 4 --fwd '%s/tps_0/{subject_dir}_fwd.nii.gz' "
          "--inv '%s/tps_0/{subject_dir}_inv.nii.gz' --moved '%s/tps_0/{subject_dir}_moved.nii.gz'"
          % (proc, out, out, out))
    print("SELF-TEST %s" % ("PASSED" if ok else "FAILED"))
    return ok


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keymorph-dir", default=None, help="clone of github.com/alanqrwang/keymorph")
    ap.add_argument("--self-test", action="store_true", help="synthetic checks, no weights / data needed")
    ap.add_argument("--processed-dir", help="e.g. .../data/processed/mindboggle (<subject>_T1w/image.nii.gz)")
    ap.add_argument("--label-name", default="dkt31aseg")
    ap.add_argument("--subjects-file", default=None, help="optional list of subject folder names")
    ap.add_argument("--max-subjects", type=int, default=0, help="first N subjects only (smoke test)")
    ap.add_argument("--out", help="output folder (one subfolder per alignment)")
    ap.add_argument("--weights", help="foundation-numkey512-numlevels6.pth.tar")
    ap.add_argument("--num-keypoints", type=int, default=512)
    ap.add_argument("--variant", choices=("S", "M", "L"), default="L", help="S/M/L = 4/5/6 U-Net levels")
    ap.add_argument("--aligns", nargs="+", default=["rigid", "affine", "tps_0"])
    ap.add_argument("--num-iters", type=int, default=5, help="groupwise iterations (BrainMorph eval: 5)")
    ap.add_argument("--pad-to", type=int, default=256)
    ap.add_argument("--tps-chunk", type=int, default=131072, help="grid points per TPS evaluation chunk")
    ap.add_argument("--weighted", action="store_true",
                    help="power-weighted groupwise fits (NOT in BrainMorph's groupwise code; its pairwise "
                         "code uses power weights). Template = power-weighted mean keypoints; fit weights "
                         "= subject power x group-mean power. No effect on tps_0 (lambda 0 interpolates).")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import_keymorph(a.keymorph_dir)
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        a.device = "cpu"
    if KM.stubbed:
        print("stubbed missing optional modules: %s" % ", ".join(KM.stubbed))
    if a.self_test:
        sys.exit(0 if self_test(a.device) else 1)

    for k in ("processed_dir", "out", "weights"):
        if not getattr(a, k):
            ap.error("--%s is required" % k.replace("_", "-"))
    for s in a.aligns:
        parse_align(s)
    subs = find_subjects(a.processed_dir, a.label_name, a.subjects_file)
    if a.max_subjects:
        subs = subs[:a.max_subjects]
    if not subs:
        raise SystemExit("no subjects in %s" % a.processed_dir)
    missing = [s["dir"] for s in subs if not s["labels"]]
    if missing:
        print("WARNING: no labels_%s for %d subjects -> no moved_seg / atlas_seg" % (a.label_name, len(missing)))
    img0 = nib.load(subs[0]["image"])
    geom = Geometry(img0.shape[:3], img0.affine, a.pad_to)
    if np.abs(geom.spacing - 1).max() > 1e-3:
        raise SystemExit("BrainMorph expects 1 mm voxels; grid spacing %s" % geom.spacing)
    levels = {"S": 4, "M": 5, "L": 6}[a.variant]
    print("subjects %d  grid %s  pad_lo %s  keymorph %s  device %s" % (
        len(subs), geom.shape, geom.pad_lo.tolist(), KM.path, a.device))
    model = build_network(a.weights, a.num_keypoints, levels, a.device)

    def kp_fn(i, padded):
        x = torch.from_numpy(padded)[None, None].to(a.device)
        points, feat = model.get_keypoints(x, return_feat=True)
        # channel power as in KeyMorph.weight_by_power (sum of the ReLU'd heatmap per keypoint)
        power = torch.relu(feat).flatten(2).sum(-1)
        return points, power

    info = {"weights": os.path.abspath(a.weights), "num_keypoints": a.num_keypoints, "variant": a.variant,
            "keymorph": KM.path, "processed_dir": os.path.abspath(a.processed_dir),
            "torch": torch.__version__,
            "keypoint_weighting": ("power: template = power-weighted mean, fit weight = subject x group-mean "
                                   "normalised power (wrapper option, not BrainMorph code)") if a.weighted
            else "none (as keymorph groupwise_register)"}
    run(subs, a.out, a.aligns, geom, kp_fn, a.device, a.num_iters, a.tps_chunk, info=info, weighted=a.weighted)
    print("done: %s" % a.out)


if __name__ == "__main__":
    main()
