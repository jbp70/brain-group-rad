#!/usr/bin/env python
"""GPU verification of the milestone-2 DALI pipeline (DALI 1.18, TF 2.10, tf10_nv.sif).

Not part of unittest discovery (tests/gpu has no __init__.py and the file name does not match
test*.py). Run on ONE GPU with CUDA_VISIBLE_DEVICES already set, normally through
tests/gpu/run_verify_glacier.sh:

    python tests/gpu/verify_dali_pipeline.py --data <npy_v2 export root> \
        --scratch <writable folder> --out <summary.json>

Checks (each prints PASS / FAIL / INFO; the exit code is 1 if any check FAILs):
  (a) pixel-center / direction / axis order: DALI warp of a synthetic phantom by known
      transforms vs TF transform_image, over candidate conventions; the empirically best one
      must be the one the code uses (params.DALI_PIXEL_CENTER, pull with T^-1, reversed axes).
  (b) re-warp on real level-4 data (rigid and affine, intensity aug OFF): the DALI render equals
      the TF render of the same T_i (single interpolation, max |diff| < tol-direct); undoing the
      DALI warp with transform_image(., T_mm) reproduces the source as well as a TF-only render
      + undo does (NCC in the template brain mask over voxels that stayed in the field of view,
      gap < tol-ncc-gap; two linear interpolations at 4 mm give NCC ~0.95, so there is no
      absolute threshold); sampled rotations and translations stay within the configured ranges
      (45 deg, 15 mm per axis).
  (c) flip consistency: flip(batch) with F T F, then undo, gives the same NCC.
  (d) target sanity: barycentric targets central and pair-consistent; rigid targets exclude the
      nuisance S_i (S_i = Rt_i^-1 T_i is a small upper-triangular scale/shear).
  (e) throughput and GPU memory at level 4, N = 48 (rigid, affine, both concurrently).
  (f) determinism: same seed -> identical first batches (both reader shuffle modes).
  (g) INFO: float16 input to warp_affine.
Control file (curriculum / global transform G, design 3.2 v3.5; real level-4 data, aug OFF):
  (h) re-read: version v1 with rot 5 deg, 3 batches, rewrite v2 with rot 40 deg; within <= 3
      batches the emitted version changes and the sampled max rotation follows it (lag reported).
  (i) G shared: identical G_dali over the batch, equal in TF mm to R(axis, +angle) with the
      written SIGNED axis and angle and the written translation (global_matrix_mm), and to the
      numpy mirror fed with the angle DALI receives (params.G_DALI_ANGLE_SIGN); T_i = G Rt_i S_i
      with Rt_i rigid within the per-image ranges and S_i the nuisance (rigid), or Rt_i S_i
      within the ranges (affine).
  (j) targets with G: barycentric targets from G Rt_i (rigid) / T_i (affine) central and
      pair-consistent; the barycenter moves with G (mu*(G) = mu*(I) G^-1 within --tol-bary);
      DALI render == TF render of the emitted T_i (single interpolation) and the round trip
      matches the TF-only one (criteria of (b)).
  (k) curriculum start values written: sampled rotation / translation / log-scale / shear
      magnitudes stay within them (and are not collapsed to 0).
Select checks with --checks (e.g. h,i,j,k). All temporary files go to a fresh folder under
--scratch (never /tmp), removed at the end.
"""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from data_loading import data_module as dm  # noqa: E402

dm.enable_memory_growth()  # before any TF GPU allocation

from data_loading import dali_loader as dl  # noqa: E402
from data_loading import params as P  # noqa: E402
from data_loading import postprocess as pp  # noqa: E402
from models import layers  # noqa: E402
from models import transforms as tfm  # noqa: E402
from runtime import args as A  # noqa: E402
from runtime.curriculum import ControlWriter, global_matrix_mm  # noqa: E402

LEVEL = 4
SHAPE = P.LEVEL_SHAPE[LEVEL]
SPACING = P.LEVEL_SPACING[LEVEL]
RESULTS = {}
FLIPS = [(True, False, False), (False, True, True), (True, True, True)]


# ---------------------------------------------------------------------------- helpers
def record(name, passed, summary, **info):
    tag = "INFO" if passed is None else ("PASS" if passed else "FAIL")
    print("[%s] %s: %s" % (tag, name, summary), flush=True)
    RESULTS[name] = dict(status=tag, summary=summary, **info)


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


class FnPipeline(dl.Pipeline):
    """Small pipeline whose graph is a closure (built from the dali_loader functions)."""

    def __init__(self, graph_fn, batch_size, seed=0, num_threads=2, device_id=0):
        dl.require_dali()
        super().__init__(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                         seed=seed, prefetch_queue_depth=1)
        self._graph_fn = graph_fn

    def define_graph(self):
        return self._graph_fn()


def run_once(pipe):
    """Build, run one batch, return numpy arrays (GPU outputs copied to host)."""
    pipe.build()
    outs = pipe.run()
    arrays = []
    for o in outs:
        o = o.as_cpu() if hasattr(o, "as_cpu") else o
        arrays.append(np.array(o.as_array()))
    return arrays


def ncc(a, b, mask):
    a = a[mask].astype(np.float64)
    b = b[mask].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b) + 1e-300))


def index_grid_np(shape):
    return np.stack(np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij"), -1)


def valid_mask(m_mm, shape, spacing, margin=1.0):
    """Voxels y whose sample position M y lies inside the volume (margin in voxels)."""
    mv = tfm.mm_to_voxel_matrix(np.asarray(m_mm, np.float64), shape, spacing).numpy()
    q = index_grid_np(shape) @ mv[:3, :3].T + mv[:3, 3]
    ok = np.ones(shape, bool)
    for a in range(3):
        ok &= (q[..., a] >= margin) & (q[..., a] <= shape[a] - 1 - margin)
    return ok


def rotation_angle_deg(m):
    lin = np.asarray(m, np.float64)[:3, :3]
    u, _, vt = np.linalg.svd(lin)
    r = u @ vt
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def phantom(shape, seed=0):
    """Asymmetric smooth phantom inside a smooth off-center window, plus one bright voxel."""
    rng = np.random.RandomState(seed)
    g = index_grid_np(shape)
    n = np.asarray(shape, np.float64)
    c = (n - 1.0) / 2.0
    vol = 0.2 + 0.4 * (g[..., 0] / n[0]) + 0.2 * (g[..., 1] / n[1]) ** 2
    for _ in range(6):
        mu = c + rng.uniform(-0.2, 0.2, 3) * n
        sig = rng.uniform(2.0, 4.0)
        vol += rng.uniform(0.3, 0.8) * np.exp(-np.sum((g - mu) ** 2, -1) / (2 * sig ** 2))
    r2 = np.sum(((g - c - np.array([1.5, -1.0, 0.5])) / (0.28 * n)) ** 2, -1)
    vol *= np.where(r2 < 1.0, (1.0 - r2) ** 2, 0.0)
    bright = tuple(int(v) for v in np.round(c + np.array([5, -7, 4])))
    vol[bright] += 2.0
    return vol.astype(np.float32), bright


def gpu_memory():
    info = {}
    try:
        m = tf.config.experimental.get_memory_info("GPU:0")
        info["tf_current_mb"] = m["current"] / 2 ** 20
        info["tf_peak_mb"] = m["peak"] / 2 ** 20
    except Exception as e:  # noqa: BLE001
        info["tf_error"] = repr(e)
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, universal_newlines=True, timeout=30,
                             stdin=subprocess.DEVNULL).stdout
        mine = [ln for ln in out.splitlines() if ln.split(",")[0].strip() == str(os.getpid())]
        info["smi_process_mb"] = float(mine[0].split(",")[1]) if mine else None
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if vis and "," not in vis:
            out = subprocess.run(["nvidia-smi", "-i", vis, "--query-gpu=index,memory.used,memory.total",
                                  "--format=csv,noheader,nounits"], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, universal_newlines=True, timeout=30,
                                 stdin=subprocess.DEVNULL).stdout.strip()
            info["smi_gpu_used_total_mb"] = out
    except Exception as e:  # noqa: BLE001
        info["smi_error"] = repr(e)
    return info


# ---------------------------------------------------------------------------- (a)
CASES_A = {
    "rotation": dict(angle_deg=30.0, axis=(0.2, -0.5, 1.0), translation_vox=(0.0, 0.0, 0.0)),
    "translation": dict(angle_deg=0.0, axis=(0.0, 0.0, 1.0), translation_vox=(2.3, -1.7, 0.6)),
    "combined": dict(angle_deg=-40.0, axis=(1.0, 0.4, -0.3), translation_vox=(1.5, -2.2, 3.1),
                     log_scale=(0.08, -0.06, 0.04), shear=(0.04, -0.03, 0.05)),
}
EXPECTED_A = (P.DALI_PIXEL_CENTER, "pull_T_inv", "reversed")


def check_a(tmp, args):
    vol, bright = phantom(SHAPE)
    np.save(os.path.join(tmp, "phantom.npy"), vol)
    inner = np.zeros(SHAPE, bool)
    inner[2:-2, 2:-2, 2:-2] = True
    p = tfm.axis_reversal_matrix(tf.float64).numpy()
    cases, ok = {}, True
    for name, prm in CASES_A.items():
        def graph(prm=prm):
            imgs = dl.read_images(["phantom.npy"], SHAPE, file_root=tmp, shuffle=False)
            t_fwd, t_pull, _ = dl.build_linear_matrices(dl.constant_linear_params(**prm), SHAPE)
            return dl.warp_linear(imgs, t_pull), t_fwd.gpu(), t_pull.gpu()

        warped, t_d, t_pull_d = run_once(FnPipeline(graph, 1, num_threads=args.num_threads))
        warped = warped[0, ..., 0]
        t_d = np.vstack([t_d[0].astype(np.float64), [0, 0, 0, 1]])
        t_pull_d = np.vstack([t_pull_d[0].astype(np.float64), [0, 0, 0, 1]])
        inv_err = float(np.abs(t_d @ t_pull_d - np.eye(4)).max())
        cand = {}
        for pc in (0.5, 0.0):
            for direction in ("pull_T_inv", "pull_T"):
                for axes in ("reversed", "same"):
                    m_in = t_d if axes == "reversed" else p @ t_d @ p
                    t_mm = tfm.dali_voxel_to_tf_mm(m_in, SHAPE, SPACING, pixel_center=pc).numpy()
                    m = np.linalg.inv(t_mm) if direction == "pull_T_inv" else t_mm
                    ref = layers.transform_image(vol[None, ..., None], m[None].astype(np.float32),
                                                 SPACING).numpy()[0, ..., 0]
                    d = np.abs(ref - warped)[inner]
                    cand["pc=%g|%s|%s" % (pc, direction, axes)] = {
                        "pixel_center": pc, "direction": direction, "axes": axes,
                        "max_abs": float(d.max()), "mean_abs": float(d.mean())}
        best_key = min(cand, key=lambda k: cand[k]["mean_abs"])
        best = cand[best_key]
        expected_key = "pc=%g|%s|%s" % EXPECTED_A
        # Pass on the MEAN error: the max is dominated by the last voxel at the field-of-view
        # border, where DALI hard-fills while transform_image fades out over one voxel
        # (documented in models/layers.py). The max is still reported.
        case_ok = (best_key == expected_key and cand[expected_key]["mean_abs"] < args.tol_a
                   and inv_err < 1e-4)
        # The translation case cannot separate pixel centers (both are exact there).
        if name == "translation":
            same = [k for k in cand if cand[k]["mean_abs"] <= best["mean_abs"] + 1e-7]
            case_ok = (expected_key in same and cand[expected_key]["mean_abs"] < args.tol_a
                       and inv_err < 1e-4)
        ok &= case_ok
        cases[name] = {"params": prm, "best": best_key, "expected": expected_key,
                       "expected_max_abs": cand[expected_key]["max_abs"],
                       "expected_mean_abs": cand[expected_key]["mean_abs"],
                       "bright_voxel": bright, "T_times_Tpull_minus_I": inv_err,
                       "T_dali": t_d[:3], "candidates": cand, "passed": bool(case_ok)}
        print("    (a) %-11s best=%s expected max/mean=%.2e/%.2e  |T Tpull - I|=%.1e"
              % (name, best_key, cand[expected_key]["max_abs"], cand[expected_key]["mean_abs"],
                 inv_err), flush=True)
    empirical = cases["rotation"]["best"]
    summary = ("expected convention %s confirmed" % (EXPECTED_A,) if ok else
               "EMPIRICAL convention (rotation case) = %s; code uses %s -> update "
               "data_loading/params.DALI_PIXEL_CENTER / conversion" % (empirical, EXPECTED_A))
    record("a_pixel_center_direction", ok, summary, empirical=empirical,
           expected=list(EXPECTED_A), cases=cases)


# ---------------------------------------------------------------------------- (b)-(d)
def real_batch(task, files, args):
    pipe = dl.LinearPipeline(task, LEVEL, files, len(files), seed=args.seed + 11,
                             file_root=args.data, shuffle=False,
                             intensity_aug=P.IntensityAugConfig.off(), num_threads=args.num_threads)
    outs = run_once(pipe)
    del pipe
    gc.collect()
    return outs


def check_b_c(task, outs, src, brain, args):
    imgs, t_d = outs[0], outs[1]
    t_mm = pp.dali_to_mm(t_d, LEVEL).numpy()
    t_inv = np.linalg.inv(t_mm.astype(np.float64)).astype(np.float32)
    undo = layers.transform_image(imgs, t_mm, SPACING).numpy()[..., 0]
    render_tf = layers.transform_image(src[..., None], t_inv, SPACING).numpy()
    undo_tf = layers.transform_image(render_tf, t_mm, SPACING).numpy()[..., 0]
    direct = np.abs(render_tf - imgs)
    per = []
    for i in range(len(src)):
        valid = valid_mask(t_mm[i], SHAPE, SPACING) & brain
        n_dali = ncc(undo[i], src[i], valid)
        n_tf = ncc(undo_tf[i], src[i], valid)
        per.append({"rotation_deg": rotation_angle_deg(t_mm[i]),
                    "translation_mm": np.abs(t_mm[i][:3, 3]).max(),
                    "ncc": n_dali, "ncc_full_brain": ncc(undo[i], src[i], brain),
                    "ncc_tf_double_interp": n_tf,
                    "error_ratio_vs_tf": (1 - n_dali) / max(1 - n_tf, 1e-12),
                    "brain_lost_fraction": 1.0 - valid.sum() / float(brain.sum()),
                    "direct_max_abs": float(direct[i].max()),
                    "direct_mean_abs": float(direct[i].mean())})
    min_ncc = min(p["ncc"] for p in per)
    max_direct = max(p["direct_max_abs"] for p in per)
    # Sampled magnitudes must respect the ranges: catches degree/radian, centering and combine
    # order errors directly. The rotation is that of Rt (rigid) or the polar factor of R K
    # (affine), whose shear adds at most ~|skew(U)| <= sqrt(3) * shear rad on top of the angle.
    rng = P.DEFAULT_LINEAR_RANGES[task]
    rot_m = pp.dali_to_mm(outs[2], LEVEL).numpy() if task == "rigid" else t_mm
    rot = [rotation_angle_deg(m) for m in rot_m]
    rot_bound = rng["rotation_deg"] + (0.01 if task == "rigid" else np.degrees(np.sqrt(3.0) * rng["shear"]))
    t_max = float(np.abs(t_mm[:, :3, 3]).max())
    t_bound = rng["translation_mm"] + 1e-3
    mag_ok = max(rot) <= rot_bound and t_max <= t_bound
    for p, r in zip(per, rot):
        p["rotation_deg"] = r
    # The decisive test is the single-interpolation agreement (DALI render == TF render of the
    # same T). The round trip goes through two linear interpolations, which at 4 mm blurs real
    # images to NCC ~0.95 whichever warp is used, so it must match the TF-only round trip
    # rather than an absolute threshold.
    worst_gap = max(p["ncc_tf_double_interp"] - p["ncc"] for p in per)
    ok = max_direct < args.tol_direct and mag_ok and worst_gap < args.tol_ncc_gap
    record("b_rewarp_%s" % task, ok,
           "min NCC %.5f (TF-only double interp min %.5f), max rot %.1f deg (<= %.2f), max |t| %.2f mm "
           "(<= %.3f), single-interp DALI vs TF max |diff| %.2e"
           % (min_ncc, min(p["ncc_tf_double_interp"] for p in per), max(rot), rot_bound, t_max,
              t_bound, max_direct),
           magnitudes_ok=mag_ok, rotation_bound_deg=rot_bound, translation_bound_mm=t_bound,
           per_image=per)
    # (c) flips
    flips = {}
    max_dev = 0.0
    for flags in FLIPS:
        img_f, (t_f,) = pp.apply_flip(imgs, [t_mm], flags)
        undo_f = layers.transform_image(img_f, t_f, SPACING).numpy()[..., 0]
        src_f = tfm.flip_images(src[..., None], flags).numpy()[..., 0]
        devs = []
        for i in range(len(src)):
            valid = valid_mask(t_mm[i], SHAPE, SPACING) & brain
            valid_f = tfm.flip_images(valid[None, ..., None].astype(np.float32), flags).numpy()[0, ..., 0] > 0.5
            devs.append(abs(ncc(undo_f[i], src_f[i], valid_f) - per[i]["ncc"]))
        flips[str(flags)] = max(devs)
        max_dev = max(max_dev, max(devs))
    record("c_flip_%s" % task, max_dev < 1e-4, "max |NCC_flipped - NCC| = %.2e" % max_dev,
           per_flip=flips)
    return t_mm


def check_d(outs_rigid, outs_affine, s_mom):
    eye = np.eye(4)
    info, ok = {}, True
    for task, outs in (("rigid", outs_rigid), ("affine", outs_affine)):
        post = pp.LinearPostprocess(task, LEVEL, flip_prob=0.5, seed=5)
        out = post(*outs)
        a = tf.cast(out["A_star"], tf.float64)
        tgt = tf.cast(out["T_target"], tf.float64)
        n = int(a.shape[0])
        mu = tfm.barycenter(a, iters=10)
        rms_center = float(np.sqrt(tfm.point_loss(mu - eye, s_mom).numpy()))
        ii, jj = np.where(~np.eye(n, dtype=bool))
        pred = tfm.pair_predictions(a, ii, jj)
        true = tfm.pair_targets(tgt, ii, jj)
        rms_pair = float(np.sqrt(np.mean(tfm.point_loss(pred - true, s_mom).numpy())))
        sp = tfm.pair_predictions(a, out["pair_i"], out["pair_j"])
        rms_sampled = float(np.sqrt(np.mean(tfm.point_loss(sp - tf.cast(out["P_star"], tf.float64),
                                                           s_mom).numpy())))
        d = {"rms_center_mm": rms_center, "rms_pair_mm": rms_pair, "rms_sampled_pairs_mm": rms_sampled,
             "n_sampled_pairs": int(out["pair_i"].shape[0]), "flip_axes": out["flip_axes"].numpy()}
        task_ok = rms_center < 1e-3 and rms_pair < 1e-3 and rms_sampled < 1e-3
        if task == "rigid":
            lin = out["A_star"].numpy()[:, :3, :3].astype(np.float64)
            d["rigidity_err"] = float(np.abs(lin @ np.transpose(lin, (0, 2, 1)) - np.eye(3)).max())
            t_full = out["T_render"].numpy().astype(np.float64)
            s_i = np.linalg.inv(tgt.numpy()) @ t_full                    # nuisance S_i
            l, h = P.DEFAULT_LINEAR_RANGES["rigid"]["log_scale"], P.DEFAULT_LINEAR_RANGES["rigid"]["shear"]
            diag = np.diagonal(s_i[:, :3, :3], axis1=1, axis2=2)
            lower = np.array([s_i[:, 1, 0], s_i[:, 2, 0], s_i[:, 2, 1]])
            upper = np.array([s_i[:, 0, 1], s_i[:, 0, 2], s_i[:, 1, 2]])
            nuis_rms = np.sqrt(tfm.point_loss(s_i - eye, s_mom).numpy())
            a_full = tfm.barycentric_targets(tf.constant(t_full), 10)
            diff_rms = float(np.sqrt(np.mean(tfm.point_loss(a_full - a, s_mom).numpy())))
            d.update({"nuisance_translation_max_mm": float(np.abs(s_i[:, :3, 3]).max()),
                      "nuisance_lower_max": float(np.abs(lower).max()),
                      "nuisance_upper_max": float(np.abs(upper).max()),
                      "nuisance_diag_range": [float(diag.min()), float(diag.max())],
                      "nuisance_rms_mm_max": float(nuis_rms.max()),
                      "full_vs_rigid_target_rms_mm": diff_rms})
            task_ok &= (d["rigidity_err"] < 1e-4 and d["nuisance_translation_max_mm"] < 1e-3
                        and d["nuisance_lower_max"] < 1e-5
                        and np.exp(-l) - 1e-5 <= diag.min() and diag.max() <= np.exp(l) + 1e-5
                        and d["nuisance_upper_max"] <= h * np.exp(l) + 1e-5
                        and 0.0 < diff_rms <= 3.0 * d["nuisance_rms_mm_max"] + 1e-3)
        else:
            d["target_equals_render"] = bool(np.allclose(out["T_target"].numpy(), out["T_render"].numpy()))
            task_ok &= d["target_equals_render"]
        d["passed"] = bool(task_ok)
        info[task] = d
        ok &= task_ok
    r = info["rigid"]
    record("d_targets", ok,
           "rigid center %.1e mm, pairs %.1e mm, nuisance rms <= %.3f mm, |A_full - A_rigid| rms %.3f mm; "
           "affine center %.1e mm, pairs %.1e mm"
           % (r["rms_center_mm"], r["rms_pair_mm"], r["nuisance_rms_mm_max"],
              r["full_vs_rigid_target_rms_mm"], info["affine"]["rms_center_mm"],
              info["affine"]["rms_pair_mm"]), **info)


# ---------------------------------------------------------------------------- (e)
def time_iterator(it, n_warmup, n_batches, shapes):
    """Returns (batches/s, outputs ok, last batch, first-batch seconds)."""
    t0 = time.time()
    float(tf.reduce_sum(next(it)[0]).numpy())  # includes pipeline build + shuffle-buffer fill
    first_s = time.time() - t0
    for _ in range(n_warmup):
        b = next(it)
        float(tf.reduce_sum(b[0]).numpy())
    t0 = time.time()
    last = None
    for _ in range(n_batches):
        last = next(it)
        float(tf.reduce_sum(last[0]).numpy())
    dt = time.time() - t0
    shapes_ok = all(tuple(x.shape) == tuple(s) for x, s in zip(last, shapes))
    imgs = last[0].numpy()
    range_ok = bool(np.all(np.isfinite(imgs)) and imgs.min() >= 0.0 and imgs.max() <= 1.0)
    return n_batches / dt, shapes_ok and range_ok, last, first_s


def check_e(args):
    info, ok = {"memory_start": gpu_memory()}, True
    kw = dict(level=LEVEL, batch_size=args.n_throughput, seed=args.seed, num_threads=args.num_threads)
    for task in P.TASKS:
        data = dm.LinearDataModule(args.data, **kw)
        t0 = time.time()
        it = data.iterator(task)
        shapes, _ = dl.linear_output_signature(task, LEVEL, args.n_throughput)
        bps, good, last, first_s = time_iterator(it, args.warmup, args.batches, shapes)
        info[task] = {"first_batch_s": first_s, "batches_per_s": bps,
                      "images_per_s": bps * args.n_throughput, "outputs_ok": good,
                      "memory": gpu_memory(), "wall_s": time.time() - t0}
        if task == "rigid":
            post = pp.LinearPostprocess("rigid", LEVEL, seed=0)
            fn_post = tf.function(lambda *x: post(*x))
            fn_post(*last)
            t1 = time.time()
            for _ in range(20):
                o = fn_post(*last)
                float(o["A_star"][0, 0, 0].numpy())
            info["postprocess_ms"] = (time.time() - t1) / 20 * 1e3
        ok &= good
        print("    (e) %s alone: %.2f batches/s, memory %s" % (task, bps, info[task]["memory"]), flush=True)
        del it, data
        gc.collect()
    data = dm.LinearDataModule(args.data, **kw)
    its = {task: data.iterator(task) for task in P.TASKS}
    for task in P.TASKS:
        next(its[task])
    t0 = time.time()
    for _ in range(args.batches):
        for task in P.TASKS:
            float(tf.reduce_sum(next(its[task])[0]).numpy())
    info["concurrent"] = {"pairs_per_s": args.batches / (time.time() - t0), "memory": gpu_memory(),
                          "prefetch_queue_depth": data.prefetch_queue_depth}
    del its, data
    gc.collect()
    record("e_throughput_memory", ok,
           "rigid %.2f, affine %.2f batches/s (N=%d); concurrent %.2f rigid+affine pairs/s; memory "
           "concurrent %s" % (info["rigid"]["batches_per_s"], info["affine"]["batches_per_s"],
                              args.n_throughput, info["concurrent"]["pairs_per_s"],
                              info["concurrent"]["memory"]),
           note="DALI memory is not in TF's get_memory_info; smi_process_mb is the whole process "
                "(TF + DALI pools, which DALI keeps after a pipeline is deleted).", **info)


# ---------------------------------------------------------------------------- (f), (g)
def check_f(files, args):
    info, ok = {}, True
    for task in P.TASKS:
        for mode in dl.SHUFFLE_MODES:
            runs = []
            for seed in (args.seed + 100, args.seed + 100, args.seed + 101):
                pipe = dl.LinearPipeline(task, LEVEL, files, args.n_real, seed=seed, file_root=args.data,
                                         shuffle=True, shuffle_mode=mode, num_threads=args.num_threads)
                runs.append(run_once(pipe))
                del pipe
                gc.collect()
            same = all(np.array_equal(a, b) for a, b in zip(runs[0], runs[1]))
            differ = not all(np.array_equal(a, b) for a, b in zip(runs[0], runs[2]))
            info["%s_%s" % (task, mode)] = {
                "same_seed_identical": same, "other_seed_differs": differ,
                "max_abs_diff_same_seed": [float(np.abs(a - b).max()) for a, b in zip(runs[0], runs[1])]}
            ok &= same and differ
    record("f_determinism", ok, "same seed identical / other seed differs: %s"
           % ", ".join("%s %s/%s" % (k, v["same_seed_identical"], v["other_seed_differs"])
                       for k, v in info.items()), **info)


def check_g(tmp, args):
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    prm = CASES_A["combined"]

    def graph(half):
        def g():
            imgs = dl.read_images(["phantom.npy"], SHAPE, file_root=tmp, shuffle=False)
            if half:
                imgs = fn.cast(imgs, dtype=types.FLOAT16)
            _, t_pull, _ = dl.build_linear_matrices(dl.constant_linear_params(**prm), SHAPE)
            out = dl.warp_linear(imgs, t_pull)
            return fn.cast(out, dtype=types.FLOAT) if half else out
        return g

    try:
        ref = run_once(FnPipeline(graph(False), 1, num_threads=args.num_threads))[0]
        half = run_once(FnPipeline(graph(True), 1, num_threads=args.num_threads))[0]
        d = float(np.abs(ref - half).max())
        record("g_float16_warp_affine", None, "float16 input supported, max |fp16 - fp32| = %.2e" % d,
               supported=True, max_abs_diff=d)
    except Exception as e:  # noqa: BLE001
        record("g_float16_warp_affine", None, "float16 input NOT supported: %r" % (e,), supported=False)


# ---------------------------------------------------------------------------- (h)-(k) control file
G_AXIS = tuple(np.asarray([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8]))
G_ANGLE = 30.0
G_T_MM = (4.0, -6.0, 8.0)


def ctl_values(rot, trans, log_scale=0.0, shear=0.0, axis=P.DEFAULT_G_AXIS, angle=0.0,
               t_mm=(0.0, 0.0, 0.0)):
    return {"rotation_deg": rot, "translation_mm": trans, "log_scale": log_scale, "shear": shear,
            "g_axis": axis, "g_angle_deg": angle, "g_translation_mm": t_mm}


def control_pipeline(task, files, args, control_file, seed):
    return dl.LinearPipeline(task, LEVEL, files, len(files), seed=seed, file_root=args.data,
                             shuffle=False, intensity_aug=P.IntensityAugConfig.off(),
                             num_threads=args.num_threads, control_file=control_file)


def run_next(pipe):
    """One more batch of a built pipeline as numpy arrays."""
    arrays = []
    for o in pipe.run():
        o = o.as_cpu() if hasattr(o, "as_cpu") else o
        arrays.append(np.array(o.as_array()))
    return arrays


def rotation_axis(m):
    r = np.asarray(m, np.float64)[:3, :3]
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return v / max(np.linalg.norm(v), 1e-12)


def qr_scale_shear(lin):
    """L = Q K with K upper triangular, positive diagonal -> (log diag (3,), shears (a, b, c))."""
    q, k = np.linalg.qr(np.asarray(lin, np.float64))
    sgn = np.sign(np.diag(k))
    k = sgn[:, None] * k
    d = np.diag(k)
    return np.log(d), np.array([k[0, 1] / d[1], k[0, 2] / d[2], k[1, 2] / d[2]])


def check_h(files, tmp, args):
    """Control file re-read: version and range change within <= 3 batches of a rewrite."""
    path = os.path.join(tmp, "control_h", "rigid.npy")
    w = ControlWriter(path)
    w.write(ctl_values(5.0, 2.0))
    v1 = w.version
    pipe = control_pipeline("rigid", files, args, path, args.seed + 21)
    pipe.build()
    rec = []

    def one(phase):
        o = run_next(pipe)
        ver = o[-1].reshape(-1)
        rot = [rotation_angle_deg(m) for m in pp.dali_to_mm(o[2], LEVEL).numpy()]
        rec.append({"phase": phase, "versions": sorted(set(float(v) for v in ver)),
                    "max_rot_deg": max(rot)})

    for _ in range(3):
        one("before")
    w.write(ctl_values(40.0, 2.0))
    v2 = w.version
    for _ in range(6):
        one("after")
    del pipe
    gc.collect()
    single = all(len(r["versions"]) == 1 for r in rec)
    after = [r for r in rec if r["phase"] == "after"]
    lag = next((k + 1 for k, r in enumerate(after) if r["versions"] == [float(v2)]), None)
    v1_ok = all(r["max_rot_deg"] <= 5.0 + 0.01 for r in rec if r["versions"] == [float(v1)])
    v2_rot = [r["max_rot_deg"] for r in rec if r["versions"] == [float(v2)]]
    v2_ok = bool(v2_rot) and max(v2_rot) > 10.0 and max(v2_rot) <= 40.0 + 0.01
    before_ok = all(r["versions"] == [float(v1)] for r in rec if r["phase"] == "before")
    ok = single and before_ok and lag is not None and lag <= 3 and v1_ok and v2_ok
    record("h_control_reread", ok,
           "rewrite v%d -> v%d seen after %s batch(es) (<= 3); max rot v1 batches <= 5.01: %s, v2 "
           "batches max %.1f deg (> 10, <= 40.01); one version per batch: %s"
           % (v1, v2, lag, v1_ok, max(v2_rot) if v2_rot else float("nan"), single),
           lag_batches=lag, batches=rec)


def check_i(files, tmp, args):
    """G shared within a batch and T_i = G Rt_i S_i with G as written."""
    info, ok = {}, True
    rng = P.DEFAULT_LINEAR_RANGES
    axis_d, t_d = P.control_global_dali(G_AXIS, G_T_MM, LEVEL)
    for task in P.TASKS:
        path = os.path.join(tmp, "control_i", "%s.npy" % task)
        w = ControlWriter(path)
        w.write(ctl_values(20.0, 5.0, rng[task]["log_scale"], rng[task]["shear"], G_AXIS, G_ANGLE, G_T_MM))
        pipe = control_pipeline(task, files, args, path, args.seed + 31)
        pipe.build()
        o = run_next(pipe)
        del pipe
        gc.collect()
        g_d = o[-2].astype(np.float64)
        shared = float(np.abs(g_d - g_d[:1]).max())
        g_mm = pp.dali_to_mm(g_d[:1], LEVEL).numpy()[0].astype(np.float64)
        # Strict on the SIGNED axis and angle (milestone 5): rotation_axis gives the axis whose
        # rotation angle is in [0, 180], so for G = R(G_AXIS, +G_ANGLE) (0 < G_ANGLE < 180) it
        # must be +G_AXIS, and the matrix must equal global_matrix_mm of the written values.
        ang = rotation_angle_deg(g_mm)
        axis_dot = float(np.dot(rotation_axis(g_mm), G_AXIS))
        t_err = float(np.abs(g_mm[:3, 3] - np.asarray(G_T_MM)).max())
        g_ref = global_matrix_mm(G_AXIS, G_ANGLE, G_T_MM)
        g_ref_err = float(np.abs(g_mm - g_ref).max())
        # numpy mirror of the DALI graph with the angle DALI receives (params.G_DALI_ANGLE_SIGN)
        m = P.linear_matrices_np(0.0, (0.0, 0.0, 1.0), (0, 0, 0), (0, 0, 0), (0, 0, 0), SHAPE,
                                 g_angle_deg=P.control_global_dali_angle(G_ANGLE), g_axis_dali=axis_d,
                                 g_t_vox_dali=t_d)
        mirror = float(np.abs(m["G"][:3] - g_d[0]).max())
        t_mm = pp.dali_to_mm(o[1], LEVEL).numpy().astype(np.float64)
        g_inv = np.linalg.inv(g_mm)
        d = {"G_shared_max_abs": shared, "G_angle_deg": ang, "G_axis_signed_dot": axis_dot,
             "G_translation_err_mm": t_err, "G_vs_global_matrix_mm_max_abs": g_ref_err,
             "G_vs_numpy_mirror_max_abs": mirror,
             "version": sorted(set(o[-1].reshape(-1).tolist()))}
        task_ok = (shared < 1e-6 and abs(ang - G_ANGLE) < 0.01 and axis_dot > 1 - 1e-5
                   and t_err < 1e-3 and g_ref_err < 1e-3 and mirror < 1e-3)
        ls, sh = rng[task]["log_scale"], rng[task]["shear"]
        if task == "rigid":
            tgt = pp.dali_to_mm(o[2], LEVEL).numpy().astype(np.float64)
            rt = g_inv @ tgt                                 # Rt_i
            s_i = np.linalg.inv(tgt) @ t_mm                  # S_i
            lin = rt[:, :3, :3]
            rigid_err = float(np.abs(lin @ np.transpose(lin, (0, 2, 1)) - np.eye(3)).max())
            rot = [rotation_angle_deg(m) for m in rt]
            diag = np.diagonal(s_i[:, :3, :3], axis1=1, axis2=2)
            lower = np.array([s_i[:, 1, 0], s_i[:, 2, 0], s_i[:, 2, 1]])
            # (The former recompose sub-check G (G^-1 tgt)(tgt^-1 T) == T was an identity for any
            # output and is dropped; the render of T itself is checked independently in (j).)
            d.update({"Rt_rigidity_err": rigid_err, "Rt_max_rot_deg": max(rot),
                      "Rt_max_translation_mm": float(np.abs(rt[:, :3, 3]).max()),
                      "S_translation_max_mm": float(np.abs(s_i[:, :3, 3]).max()),
                      "S_lower_max": float(np.abs(lower).max()),
                      "S_diag_range": [float(diag.min()), float(diag.max())]})
            task_ok &= (rigid_err < 1e-4 and max(rot) <= 20.0 + 0.01
                        and d["Rt_max_translation_mm"] <= 5.0 + 1e-3
                        and d["S_translation_max_mm"] < 1e-3 and d["S_lower_max"] < 1e-5
                        and np.exp(-ls) - 1e-5 <= diag.min() and diag.max() <= np.exp(ls) + 1e-5)
        else:
            m = g_inv @ t_mm                                 # Rt_i S_i
            rot = [rotation_angle_deg(x) for x in m]
            q = [qr_scale_shear(x[:3, :3]) for x in m]
            bound = 20.0 + np.degrees(np.sqrt(3.0) * sh) + 0.01
            d.update({"max_rot_deg": max(rot), "rot_bound_deg": bound,
                      "max_translation_mm": float(np.abs(m[:, :3, 3]).max()),
                      "max_abs_log_scale": float(max(np.abs(a).max() for a, _ in q)),
                      "max_abs_shear": float(max(np.abs(b).max() for _, b in q))})
            task_ok &= (max(rot) <= bound and d["max_translation_mm"] <= 5.0 + 1e-3
                        and d["max_abs_log_scale"] <= ls + 1e-5 and d["max_abs_shear"] <= sh + 1e-5)
        d["passed"] = bool(task_ok)
        info[task] = d
        ok &= task_ok
    record("i_global_shared", ok,
           "G identical over the batch (max diff rigid %.1e / affine %.1e), angle %.3f deg (written %g), "
           "signed axis dot %.6f, |G - global_matrix_mm(written)| %.1e, |t - written| %.1e mm, "
           "T = G Rt S with Rt within 20 deg / 5 mm"
           % (info["rigid"]["G_shared_max_abs"], info["affine"]["G_shared_max_abs"],
              info["rigid"]["G_angle_deg"], G_ANGLE, info["rigid"]["G_axis_signed_dot"],
              info["rigid"]["G_vs_global_matrix_mm_max_abs"], info["rigid"]["G_translation_err_mm"]),
           **info)


def check_j(files, src, brain, s_mom, tmp, args):
    """Targets with G (central, pair-consistent) and the single-interpolation / round-trip criteria
    of check (b) for the G-composed T_i."""
    info, ok = {}, True
    eye = np.eye(4)
    rng = P.DEFAULT_LINEAR_RANGES
    for task in P.TASKS:
        path = os.path.join(tmp, "control_j", "%s.npy" % task)
        w = ControlWriter(path)
        w.write(ctl_values(20.0, 5.0, rng[task]["log_scale"], rng[task]["shear"], G_AXIS, G_ANGLE, G_T_MM))
        pipe = control_pipeline(task, files, args, path, args.seed + 41)
        pipe.build()
        o = run_next(pipe)
        del pipe
        gc.collect()
        k = dl.base_output_count(task)
        base = o[:k]
        imgs = base[0]
        g_mm = pp.dali_to_mm(o[k][:1], LEVEL).numpy()[0].astype(np.float64)
        post = pp.LinearPostprocess(task, LEVEL, flip_prob=0.5, seed=5)
        out = post(*base)
        a = tf.cast(out["A_star"], tf.float64)
        tgt = tf.cast(out["T_target"], tf.float64)
        n = int(a.shape[0])
        mu = tfm.barycenter(a, iters=10)
        rms_center = float(np.sqrt(tfm.point_loss(mu - eye, s_mom).numpy()))
        ii, jj = np.where(~np.eye(n, dtype=bool))
        rms_pair = float(np.sqrt(np.mean(tfm.point_loss(
            tfm.pair_predictions(a, ii, jj) - tfm.pair_targets(tgt, ii, jj), s_mom).numpy())))
        # Barycenter of the unflipped targets moves with G: mu*(G) ~ mu*(G = I) G^-1.
        tgt0 = pp.dali_to_mm(base[2] if task == "rigid" else base[1], LEVEL).numpy().astype(np.float64)
        mu_g = tfm.barycenter(tf.constant(np.linalg.inv(tgt0)), iters=10).numpy()
        mu_0 = tfm.barycenter(tf.constant(np.linalg.inv(np.linalg.inv(g_mm)[None] @ tgt0)), iters=10).numpy()
        bary_equiv = float(np.abs(mu_g - mu_0 @ np.linalg.inv(g_mm)).max())
        # Re-warp (as check b): DALI render == TF render of the same T; round trip vs TF-only.
        t_mm = pp.dali_to_mm(base[1], LEVEL).numpy()
        t_inv = np.linalg.inv(t_mm.astype(np.float64)).astype(np.float32)
        undo = layers.transform_image(imgs, t_mm, SPACING).numpy()[..., 0]
        render_tf = layers.transform_image(src[..., None], t_inv, SPACING).numpy()
        undo_tf = layers.transform_image(render_tf, t_mm, SPACING).numpy()[..., 0]
        max_direct = float(np.abs(render_tf - imgs).max())
        gaps, nccs = [], []
        for i in range(len(src)):
            valid = valid_mask(t_mm[i], SHAPE, SPACING) & brain
            n_dali, n_tf = ncc(undo[i], src[i], valid), ncc(undo_tf[i], src[i], valid)
            gaps.append(n_tf - n_dali)
            nccs.append(n_dali)
        d = {"rms_center_mm": rms_center, "rms_pair_mm": rms_pair,
             "barycenter_rot_deg": rotation_angle_deg(mu_g), "barycenter_equivariance_max_abs": bary_equiv,
             "direct_max_abs": max_direct, "worst_ncc_gap": max(gaps), "min_ncc": min(nccs),
             "flip_axes": out["flip_axes"].numpy()}
        task_ok = (rms_center < 1e-3 and rms_pair < 1e-3 and max_direct < args.tol_direct
                   and max(gaps) < args.tol_ncc_gap and bary_equiv < args.tol_bary)
        d["passed"] = bool(task_ok)
        info[task] = d
        ok &= task_ok
    r = info["rigid"]
    record("j_targets_with_global", ok,
           "rigid center %.1e / pairs %.1e mm, barycenter rot %.1f deg, |mu*(G) - mu*(I) G^-1| rigid "
           "%.1e / affine %.1e (< %g); single-interp max |diff| rigid %.1e / affine %.1e; worst NCC "
           "gap rigid %.1e / affine %.1e"
           % (r["rms_center_mm"], r["rms_pair_mm"], r["barycenter_rot_deg"],
              r["barycenter_equivariance_max_abs"], info["affine"]["barycenter_equivariance_max_abs"],
              args.tol_bary, r["direct_max_abs"], info["affine"]["direct_max_abs"], r["worst_ncc_gap"],
              info["affine"]["worst_ncc_gap"]),
           **info)


def check_k(files, tmp, args):
    """Curriculum start values: sampled per-image magnitudes respect the written ranges."""
    start = {t: {k: (v if v is not None else P.DEFAULT_LINEAR_RANGES[t][k]) for k, v in d.items()}
             for t, d in A.CURRICULUM_START_DEFAULTS.items()}
    info, ok = {}, True
    for task in P.TASKS:
        s = start[task]
        path = os.path.join(tmp, "control_k", "%s.npy" % task)
        w = ControlWriter(path)
        w.write(ctl_values(s["rotation_deg"], s["translation_mm"], s["log_scale"], s["shear"]))
        pipe = control_pipeline(task, files, args, path, args.seed + 51)
        pipe.build()
        outs = [run_next(pipe) for _ in range(4)]
        del pipe
        gc.collect()
        t_mm = np.concatenate([pp.dali_to_mm(o[1], LEVEL).numpy() for o in outs]).astype(np.float64)
        if task == "rigid":
            rt = np.concatenate([pp.dali_to_mm(o[2], LEVEL).numpy() for o in outs]).astype(np.float64)
            lin = (np.linalg.inv(rt) @ t_mm)[:, :3, :3]
            rot = [rotation_angle_deg(m) for m in rt]
            rot_bound = s["rotation_deg"] + 0.01
        else:
            lin = t_mm[:, :3, :3]
            rot = [rotation_angle_deg(m) for m in t_mm]
            rot_bound = s["rotation_deg"] + np.degrees(np.sqrt(3.0) * s["shear"]) + 0.01
        q = [qr_scale_shear(x) for x in lin]
        max_t = float(np.abs(t_mm[:, :3, 3]).max())
        d = {"n_samples": len(rot), "start": s, "max_rot_deg": max(rot), "rot_bound_deg": rot_bound,
             "max_translation_mm": max_t,
             "max_abs_log_scale": float(max(np.abs(a).max() for a, _ in q)),
             "max_abs_shear": float(max(np.abs(b).max() for _, b in q))}
        task_ok = (max(rot) <= rot_bound and max_t <= s["translation_mm"] + 1e-3
                   and d["max_abs_log_scale"] <= s["log_scale"] + 1e-5
                   and d["max_abs_shear"] <= s["shear"] + 1e-5
                   # the ranges are active, not collapsed to 0 (32 samples)
                   and max(rot) >= 0.5 * s["rotation_deg"] and max_t >= 0.5 * s["translation_mm"])
        d["passed"] = bool(task_ok)
        info[task] = d
        ok &= task_ok
    record("k_curriculum_magnitudes", ok,
           "rigid max rot %.2f (<= %.2f) deg, max |t| %.2f mm; affine max rot %.2f (<= %.2f) deg, max "
           "|t| %.2f mm, |log s| %.4f, |shear| %.4f"
           % (info["rigid"]["max_rot_deg"], info["rigid"]["rot_bound_deg"],
              info["rigid"]["max_translation_mm"], info["affine"]["max_rot_deg"],
              info["affine"]["rot_bound_deg"], info["affine"]["max_translation_mm"],
              info["affine"]["max_abs_log_scale"], info["affine"]["max_abs_shear"]), **info)


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="npy_v2 export root (export_manifest.csv, template/)")
    ap.add_argument("--scratch", required=True, help="writable folder for temporary files")
    ap.add_argument("--out", required=True, help="JSON summary path")
    ap.add_argument("--checks", default="a,b,c,d,e,f,g,h,i,j,k")
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--n-real", type=int, default=8)
    ap.add_argument("--n-throughput", type=int, default=48)
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol-a", type=float, default=1e-4, help="mean |DALI - TF| for check (a)")
    ap.add_argument("--tol-ncc-gap", type=float, default=1e-3,
                    help="max NCC shortfall of the DALI round trip vs the TF-only round trip (b)")
    ap.add_argument("--tol-direct", type=float, default=2e-3, help="max |DALI - TF| single interp (b)")
    ap.add_argument("--tol-bary", type=float, default=1e-6,
                    help="max |mu*(G) - mu*(I) G^-1| entry (matrix entries, translations in mm) in (j); "
                    "the fixed point starts at I, so only its convergence error remains (max 1.7e-14 "
                    "over 20 synthetic groups per task at 10 iterations, G 30 deg / Rt 20 deg, float64; "
                    ".agents/tasks/milestone-5/measure_bary.py)")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()
    checks = set(c.strip() for c in args.checks.split(",") if c.strip())

    os.makedirs(args.scratch, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="verify_m2_", dir=args.scratch)
    env = {"python": sys.version.split()[0], "tensorflow": tf.__version__, "numpy": np.__version__,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
           "pixel_center": P.DALI_PIXEL_CENTER, "level": LEVEL, "data": args.data}
    try:
        import nvidia.dali as dali
        env["dali"] = dali.__version__
    except ImportError as e:
        env["dali"] = None
        print("nvidia.dali not importable: %r" % (e,))
    print("environment: %s" % env, flush=True)

    def guarded(name, fn_, *a):
        try:
            return fn_(*a)
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            print(tb, flush=True)
            record(name, False, "exception: %s" % tb.strip().splitlines()[-1], traceback=tb)
            return None

    try:
        rows = dm.read_manifest(os.path.join(args.data, dm.MANIFEST_FILE))
        train_files = dm.manifest_files(rows, LEVEL, "train")
        env["n_train"] = len(train_files)
        env["n_val"] = len(dm.manifest_files(rows, LEVEL, "val"))
        files = train_files[:args.n_real]
        if "a" in checks:
            guarded("a_pixel_center_direction", check_a, tmp, args)
        if checks & {"b", "c", "d"}:
            src = np.stack([np.load(os.path.join(args.data, f)) for f in files]).astype(np.float32)
            brain = np.load(os.path.join(args.data, dm.TEMPLATE_DIR, "template_mask_4.npy")) > 0
            outs = {}
            for task in P.TASKS:
                outs[task] = guarded("b_rewarp_%s" % task, real_batch, task, files, args)
                if outs[task] is not None and checks & {"b", "c"}:
                    guarded("b_rewarp_%s" % task, check_b_c, task, outs[task], src, brain, args)
            if "d" in checks and all(outs[t] is not None for t in P.TASKS):
                s_mom = pp.brain_S_matrix(LEVEL, os.path.join(args.data, dm.TEMPLATE_DIR))
                guarded("d_targets", check_d, outs["rigid"], outs["affine"], s_mom)
        if "f" in checks:
            # A 200-file subset keeps the 'buffer' mode shuffle buffer (initial_fill) small and
            # the check fast. Both shuffle modes are checked.
            guarded("f_determinism", check_f, train_files[:200], args)
        if "g" in checks:
            if not os.path.isfile(os.path.join(tmp, "phantom.npy")):
                np.save(os.path.join(tmp, "phantom.npy"), phantom(SHAPE)[0])
            guarded("g_float16_warp_affine", check_g, tmp, args)
        if "e" in checks:
            guarded("e_throughput_memory", check_e, args)
        # (h)-(k): control file (curriculum / global transform). Control files go to tmp.
        if "h" in checks:
            guarded("h_control_reread", check_h, files, tmp, args)
        if "i" in checks:
            guarded("i_global_shared", check_i, files, tmp, args)
        if "j" in checks:
            src = np.stack([np.load(os.path.join(args.data, f)) for f in files]).astype(np.float32)
            brain = np.load(os.path.join(args.data, dm.TEMPLATE_DIR, "template_mask_4.npy")) > 0
            s_mom = pp.brain_S_matrix(LEVEL, os.path.join(args.data, dm.TEMPLATE_DIR))
            guarded("j_targets_with_global", check_j, files, src, brain, s_mom, tmp, args)
        if "k" in checks:
            guarded("k_curriculum_magnitudes", check_k, files, tmp, args)
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        failed = [k for k, v in RESULTS.items() if v["status"] == "FAIL"]
        summary = {"environment": env, "passed": not failed, "failed": failed, "checks": RESULTS}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(to_jsonable(summary), f, indent=2)
        print("summary written to %s" % args.out)
        print("OVERALL: %s%s" % ("PASS" if not failed else "FAIL", "" if not failed else " %s" % failed),
              flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
