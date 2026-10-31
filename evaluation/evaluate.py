"""Groupwise registration evaluation on the processed Mindboggle grid (any method, same code).

Inputs: the processed subjects (``<processed-dir>/<subject>_T1w/image.nii.gz`` +
``labels_<label-name>.nii.gz``, all on one grid) and, per method, a forward and an inverse
displacement field per subject (path templates with ``{subject}`` / ``{subject_dir}``), plus
optionally the method's atlas image and atlas segmentation. Field convention: see
evaluation/fields.py; find it with ``check-convention`` first.

Metrics (masks: subject brain = image > 0; atlas brain = voxels where >= 50 % of the warped
subject images are > 0; folding / Jacobian statistics inside these masks):

per subject (template space)
  template_dice[_cortical|_subcortical]   warped labels_i vs the atlas segmentation
  template_dice_loo[...]                  warped labels_i vs majority vote of the OTHER warped labels
                                          (method-independent atlas segmentation)
  transfer_dice[...]                      atlas segmentation warped to subject i (inv_i) vs labels_i
  template_ncc / template_lncc            warped image_i vs atlas image, atlas mask
  fold_fwd_pct / fold_inv_pct             % voxels det J <= 0 (fwd: atlas mask; inv: subject mask)
  logjac_std_fwd / logjac_std_inv         std of log det J (det J > 0 voxels)
  inv_err_fwd_mm / _max                   |fwd_i then inv_i| = |compose(fwd_i, inv_i)|, atlas mask
  inv_err_inv_mm / _max                   |compose(inv_i, fwd_i)|, subject mask
  template_hd95 (--hd95)                  mean HD95 over labels, warped labels_i vs atlas segmentation
per pair i -> j (labels/image of i moved to j with u_ij = compose(inv_j, fwd_i), j's mask)
  pair_dice[...], pair_ncc, pair_lncc, pair_fold_pct, pair_logjac_std, pair_hd95 (--hd95)
group
  centrality_mean_mm / centrality_max_mm  |mean_i fwd_i| over the atlas mask (mm)
  mean_logjac_abs                         mean over the atlas mask of |mean_i log det J_i|
  atlas_intensity_std                     mean voxelwise std of the warped images (atlas mask)
  label list: labels present in every subject (background excluded) unless --labels is given.

``--identity``: zero fields (the no-registration reference); the atlas defaults to the mean image
and the atlas segmentation to the majority vote, as for any method without them.

Outputs in --out: per_subject.csv, per_pair.csv, per_label.csv, summary.json (mean/std/median/
min/max per metric + group metrics + inputs and convention), labels.txt.

    python -m evaluation check-convention --processed-dir P --fwd 'F/{subject}_fwd.nii.gz' \\
        --inv 'F/{subject}_inv.nii.gz' --moved 'F/{subject}_moved.nii.gz'
    python -m evaluation run --processed-dir P --method baseline --fwd ... --inv ... \\
        --atlas F/atlas.nii.gz --atlas-seg F/atlas_seg.nii.gz <convention flags> --out R/baseline --workers 8
    python -m evaluation run --processed-dir P --method identity --identity --out R/identity --workers 8
"""
import argparse
import csv
import glob
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

from . import fields as F
from . import metrics as M

log = logging.getLogger("evaluation")

# Shared read-only state for forked workers (set before each Pool is created).
S: Dict[str, object] = {}


# ------------------------------------------------------------------ subjects
def find_subjects(processed_dir: str, label_name: str, subjects_file: Optional[str] = None) -> List[dict]:
    dirs = sorted(d for d in glob.glob(os.path.join(processed_dir, "*")) if os.path.isdir(d)
                  and os.path.isfile(os.path.join(d, "image.nii.gz")))
    subs = []
    for d in dirs:
        name = os.path.basename(d)
        subj = name
        meta = os.path.join(d, "meta.json")
        if os.path.isfile(meta):
            try:
                with open(meta) as f:
                    subj = json.load(f).get("row", {}).get("subject") or name
            except ValueError:
                pass
        elif name.endswith("_T1w"):
            subj = name[:-4]
        lab = os.path.join(d, "labels_%s.nii.gz" % label_name)
        if not os.path.isfile(lab):
            raise FileNotFoundError("missing %s" % lab)
        subs.append({"subject": subj, "dir": name, "image": os.path.join(d, "image.nii.gz"), "labels": lab})
    if subjects_file:
        with open(subjects_file) as f:
            want = [x.split("#", 1)[0].strip() for x in f]
        want = [w for w in want if w]
        by = {s["subject"]: s for s in subs}
        by.update({s["dir"]: s for s in subs})
        missing = [w for w in want if w not in by]
        if missing:
            raise ValueError("subjects not found in %s: %s" % (processed_dir, missing[:5]))
        subs = [by[w] for w in want]
    if len(subs) < 2:
        raise ValueError("need at least 2 subjects, found %d in %s" % (len(subs), processed_dir))
    return subs


def load_grid_data(subs: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, ...], np.ndarray]:
    """images [N,X,Y,Z] float32, labels [N,X,Y,Z] (uint16/int32), masks [N,X,Y,Z] bool, shape, affine."""
    ref = nib.load(subs[0]["image"])
    shape = tuple(int(s) for s in ref.shape[:3])
    aff = ref.affine
    n = len(subs)
    images = np.empty((n,) + shape, np.float32)
    labels = None
    for k, s in enumerate(subs):
        im = nib.load(s["image"])
        lb = nib.load(s["labels"])
        for what, x in (("image", im), ("labels", lb)):
            if tuple(x.shape[:3]) != shape or not np.allclose(x.affine, aff, atol=1e-3):
                raise ValueError("%s of %s is not on the common grid %s" % (what, s["subject"], shape))
        images[k] = np.asanyarray(im.dataobj, dtype=np.float32).reshape(shape)
        lab = np.asanyarray(lb.dataobj).reshape(shape)
        if labels is None:
            dt = np.uint16 if lab.max() < 65536 and lab.min() >= 0 else np.int32
            labels = np.empty((n,) + shape, dt)
        labels[k] = lab
    return images, labels, images > 0, shape, aff


def spacing_of(aff: np.ndarray) -> Tuple[float, float, float]:
    sp = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))
    off = aff[:3, :3] / sp
    if not np.allclose(off, np.diag(np.diag(off)), atol=1e-4):
        log.warning("grid axes are not aligned with world axes; mm-unit fields are interpreted per grid axis")
    return tuple(float(x) for x in sp)


def load_volume(path: str, shape: Sequence[int], aff: np.ndarray, what: str, dtype=None) -> np.ndarray:
    if path.endswith(".npy"):
        a = np.load(path)
    else:
        img = nib.load(path)
        if tuple(img.shape[:3]) == tuple(shape) and not np.allclose(img.affine, aff, atol=1e-3):
            log.warning("%s: affine differs from the processed grid (shape matches; using voxels as is)", path)
        a = np.asanyarray(img.dataobj)
    a = np.asarray(a).reshape([d for d in np.shape(a) if d != 1] if np.ndim(a) > 3 else np.shape(a))
    if tuple(a.shape) != tuple(shape):
        raise ValueError("%s %s has shape %s, expected %s" % (what, path, a.shape, tuple(shape)))
    return a.astype(dtype) if dtype is not None else a


# ------------------------------------------------------------------ convention check
def cmd_check_convention(a) -> int:
    subs = find_subjects(a.processed_dir, a.label_name, a.subjects_file)[: a.n_subjects]
    ref = nib.load(subs[0]["image"])
    shape, aff = tuple(int(s) for s in ref.shape[:3]), ref.affine
    sp = spacing_of(aff)
    results = {}
    ident = []
    for s in subs:
        img = np.asanyarray(nib.load(s["image"]).dataobj, dtype=np.float32).reshape(shape)
        moved = load_volume(a.moved.format(subject=s["subject"], subject_dir=s["dir"]), shape, aff, "moved image",
                            np.float32)
        mask = (moved != 0) | (img != 0)
        ident.append(M.ncc(img, moved, mask))
        raw = {r: F.load_raw_field(t.format(subject=s["subject"], subject_dir=s["dir"]), shape)
               for r, t in (("fwd", a.fwd), ("inv", a.inv))}
        for conv in F.all_conventions(a.units):
            fwd = conv.to_canonical(raw[conv.file_role("fwd")], sp)
            results.setdefault(conv.describe(), [conv, []])[1].append(M.ncc(F.warp(img, fwd), moved, mask))
    ranked = sorted(results.values(), key=lambda cv: -float(np.mean(cv[1])))
    print("subjects: %s" % ", ".join(s["subject"] for s in subs))
    print("NCC(moved image, image without warping) = %.4f" % float(np.mean(ident)))
    print("top conventions by NCC(image warped with fwd, method's moved image):")
    for conv, v in ranked[:8]:
        print("  %.4f  %s" % (float(np.mean(v)), conv.describe()))
    best, second = ranked[0], ranked[1]
    b, c = float(np.mean(best[1])), float(np.mean(second[1]))
    conv = best[0]
    # inverse consistency of the winner (both files with the same component/sign convention)
    s = subs[0]
    src = F.FieldSource(a.fwd, a.inv, conv, shape, sp)
    fwd, inv = src.get("fwd", s), src.get("inv", s)
    img = np.asanyarray(nib.load(s["image"]).dataobj, dtype=np.float32).reshape(shape)
    m = img > 0
    e1 = F.norm_mm(F.compose(fwd, inv), sp)
    e2 = F.norm_mm(F.compose(inv, fwd), sp)
    mag = F.norm_mm(fwd, sp)
    print("winner on %s: |fwd| mean %.3f mm; inverse consistency |fwd then inv| %.3f mm, |inv then fwd| %.3f mm "
          "(brain mask of the subject; small vs |fwd| = the inv file is the inverse in this convention)"
          % (s["subject"], float(mag[m].mean()), float(e1[m].mean()), float(e2[m].mean())))
    ok = b > 0.9 and b - c > 0.002 and b > float(np.mean(ident)) + 0.002
    if ok:
        print("RESULT: use  %s" % conv.cli_flags())
    else:
        print("RESULT: AMBIGUOUS (best %.4f, second %.4f, identity %.4f): check the fields/moved images by hand"
              % (b, c, float(np.mean(ident))))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"subjects": [s["subject"] for s in subs], "identity_ncc": float(np.mean(ident)),
                       "ranking": [dict(conv.as_dict(), ncc=float(np.mean(v))) for conv, v in ranked],
                       "best": conv.as_dict(), "unambiguous": ok}, f, indent=1)
    return 0 if ok else 2


# ------------------------------------------------------------------ workers
def _lab_metrics(pred: np.ndarray, true: np.ndarray, prefix: str, row: dict, per_label: dict, key) -> None:
    labels = S["labels_list"]
    d = M.dice_per_label(pred, true, labels)
    row.update(M.dice_summary(d, labels, prefix))
    per_label[key] = d.astype(np.float32)
    if S["hd95"]:
        h = M.hd95_per_label(pred, true, labels, S["spacing"])
        row[prefix.replace("dice", "hd95")] = float(np.nanmean(h)) if np.isfinite(h).any() else float("nan")


def _pass1(i: int):
    """Warp subject i into atlas space."""
    fwd = S["fields"].get("fwd", S["subs"][i])
    return i, F.warp(S["images"][i], fwd, 1), F.warp(S["labels"][i], fwd, 0)


def _pass2(i: int):
    """Template-space metrics of subject i (atlas, atlas mask/seg known)."""
    sub = S["subs"][i]
    src = S["fields"]
    sp = S["spacing"]
    fwd = src.get("fwd", sub)
    inv = src.get("inv", sub)
    am, sm = S["atlas_mask"], S["masks"][i]
    row = {"subject": sub["subject"]}
    per = {}
    wimg = S["warped_images"][i]
    row["template_ncc"] = M.ncc(wimg, S["atlas"], am)
    row["template_lncc"] = M.lncc(wimg, S["atlas"], am, S["lncc_window"])
    _lab_metrics(S["warped_labels"][i], S["atlas_seg"], "template_dice", row, per, ("template", i))
    trans = F.warp(S["atlas_seg"], inv, 0)
    _lab_metrics(trans, S["labels"][i], "transfer_dice", row, per, ("transfer", i))
    jf = F.jacobian_det(fwd)
    ji = F.jacobian_det(inv)
    row["fold_fwd_pct"] = M.folding_pct(jf, am)
    row["fold_inv_pct"] = M.folding_pct(ji, sm)
    row["logjac_std_fwd"] = M.logjac_std(jf, am)
    row["logjac_std_inv"] = M.logjac_std(ji, sm)
    e1 = F.norm_mm(F.compose(fwd, inv), sp)
    e2 = F.norm_mm(F.compose(inv, fwd), sp)
    row["inv_err_fwd_mm"], row["inv_err_fwd_max_mm"] = float(e1[am].mean()), float(e1[am].max())
    row["inv_err_inv_mm"], row["inv_err_inv_max_mm"] = float(e2[sm].mean()), float(e2[sm].max())
    row["disp_fwd_mean_mm"] = float(F.norm_mm(fwd, sp)[am].mean())
    logj = np.log(np.maximum(jf, 1e-6)).astype(np.float32)
    return i, row, per, logj


def _pairs_for_j(task):
    """Pair metrics for every i in ``ii`` moved into subject j's space."""
    j, ii = task
    src = S["fields"]
    subs = S["subs"]
    inv_j = src.get("inv", subs[j])
    mj = S["masks"][j]
    rows, per = [], {}
    for i in ii:
        u = F.compose(inv_j, src.get("fwd", subs[i]))
        row = {"moving": subs[i]["subject"], "fixed": subs[j]["subject"]}
        _lab_metrics(F.warp(S["labels"][i], u, 0), S["labels"][j], "pair_dice", row, per, ("pair", i, j))
        wimg = F.warp(S["images"][i], u, 1)
        row["pair_ncc"] = M.ncc(wimg, S["images"][j], mj)
        row["pair_lncc"] = M.lncc(wimg, S["images"][j], mj, S["lncc_window"])
        jac = F.jacobian_det(u)
        row["pair_fold_pct"] = M.folding_pct(jac, mj)
        row["pair_logjac_std"] = M.logjac_std(jac, mj)
        rows.append(row)
    return rows, per


def _map(func, items, workers: int):
    """Ordered results of func over items (forked pool when workers > 1)."""
    if workers <= 1 or len(items) <= 1:
        for x in items:
            yield func(x)
        return
    ctx = mp.get_context("fork")
    with ctx.Pool(min(workers, len(items))) as pool:
        for r in pool.imap(func, items, chunksize=1):
            yield r


# ------------------------------------------------------------------ run
def select_pairs(n: int, spec: str, seed: int) -> List[Tuple[int, int]]:
    allp = [(i, j) for j in range(n) for i in range(n) if i != j]
    if spec == "all":
        return allp
    if spec in ("none", "0"):
        return []
    k = int(spec)
    if k >= len(allp):
        return allp
    rng = np.random.RandomState(seed)
    sel = sorted(rng.choice(len(allp), size=k, replace=False))
    return [allp[s] for s in sel]


def _summ(vals: List[float]) -> dict:
    v = np.asarray([x for x in vals if x is not None], np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return {"n": 0}
    return {"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)), "min": float(v.min()),
            "max": float(v.max()), "n": int(v.size)}


def _write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def cmd_run(a) -> int:
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    subs = find_subjects(a.processed_dir, a.label_name, a.subjects_file)
    n = len(subs)
    images, labels, masks, shape, aff = load_grid_data(subs)
    sp = spacing_of(aff)
    conv = F.Convention(a.components, a.signs, a.units, a.swap_roles)
    src = F.FieldSource(a.fwd, a.inv, conv, shape, sp, identity=a.identity, cache=a.field_cache)
    if a.labels:
        with open(a.labels) as f:
            lab_list = sorted({int(x.split("#", 1)[0]) for x in f if x.split("#", 1)[0].strip()})
    else:
        common = None
        for k in range(n):
            u = set(np.unique(labels[k]).tolist()) - {0}
            common = u if common is None else common & u
        lab_list = sorted(common)
    with open(os.path.join(a.out, "labels.txt"), "w") as f:
        f.write("\n".join(map(str, lab_list)) + "\n")
    log.info("%d subjects, grid %s, spacing %s, %d labels (%d cortical), method %s, convention %s",
             n, shape, sp, len(lab_list), int(M.is_cortical(lab_list).sum()), a.method,
             "identity" if a.identity else conv.describe())
    if a.preload_fwd:
        src.preload("fwd", subs)
        log.info("preloaded %d forward fields, %.0f s", n, time.time() - t0)
    S.clear()
    S.update(subs=subs, images=images, labels=labels, masks=masks, fields=src, spacing=sp,
             labels_list=np.asarray(lab_list), hd95=a.hd95, lncc_window=a.lncc_window)

    # pass 1: warp everything into atlas space
    wimgs = np.empty_like(images)
    wlabs = np.empty_like(labels)
    for i, wi, wl in _map(_pass1, list(range(n)), a.workers):
        wimgs[i], wlabs[i] = wi, wl
    log.info("pass 1 (warp to atlas space) done, %.0f s", time.time() - t0)
    atlas_mask = np.mean(wimgs > 0, axis=0) >= 0.5
    if a.atlas:
        atlas = load_volume(a.atlas, shape, aff, "atlas", np.float32)
        atlas_src = a.atlas
    else:
        atlas = wimgs.mean(axis=0)
        atlas_src = "mean of the warped images"
    mv = M.MajorityVote(wlabs)
    if a.atlas_seg:
        atlas_seg = load_volume(a.atlas_seg, shape, aff, "atlas segmentation")
        atlas_seg = np.rint(atlas_seg).astype(labels.dtype)
        seg_src = a.atlas_seg
    else:
        atlas_seg = mv.full()
        seg_src = "majority vote of the warped labels"
    S.update(warped_images=wimgs, warped_labels=wlabs, atlas=atlas, atlas_mask=atlas_mask, atlas_seg=atlas_seg)

    # pass 2: template-space metrics
    rows: List[Optional[dict]] = [None] * n
    per_label: Dict = {}
    sum_logj = np.zeros(shape, np.float64)
    for i, row, per, logj in _map(_pass2, list(range(n)), a.workers):
        rows[i] = row
        per_label.update(per)
        sum_logj += logj
    for i in range(n):  # leave-one-out majority vote (parent: counts are modified in place)
        d = M.dice_per_label(wlabs[i], mv.loo(i), lab_list)
        rows[i].update(M.dice_summary(d, lab_list, "template_dice_loo"))
        per_label[("template_loo", i)] = d.astype(np.float32)
    del mv
    log.info("pass 2 (template metrics) done, %.0f s", time.time() - t0)

    # group metrics
    mean_fwd = np.zeros(shape + (3,), np.float64)
    for i in range(n):
        mean_fwd += src.get("fwd", subs[i])
    cen = F.norm_mm((mean_fwd / n).astype(np.float32), sp)
    group = {"centrality_mean_mm": float(cen[atlas_mask].mean()), "centrality_max_mm": float(cen[atlas_mask].max()),
             "mean_logjac_abs": float(np.abs(sum_logj / n)[atlas_mask].mean()),
             "atlas_intensity_std": float(wimgs.std(axis=0)[atlas_mask].mean()),
             "atlas_mask_voxels": int(atlas_mask.sum())}

    # pass 3: pairs, grouped by the fixed subject j
    pairs = select_pairs(n, a.pairs, a.seed)
    by_j: Dict[int, List[int]] = {}
    for i, j in pairs:
        by_j.setdefault(j, []).append(i)
    prow: List[dict] = []
    for k, (rr, per) in enumerate(_map(_pairs_for_j, sorted(by_j.items()), a.workers)):
        prow += rr
        per_label.update(per)
        if (k + 1) % 10 == 0:
            log.info("pairs: %d / %d fixed subjects, %.0f s", k + 1, len(by_j), time.time() - t0)
    log.info("pass 3 (%d pairs) done, %.0f s", len(prow), time.time() - t0)

    # outputs
    _write_csv(os.path.join(a.out, "per_subject.csv"), rows)
    _write_csv(os.path.join(a.out, "per_pair.csv"), prow)
    plabel = []
    for li, lab in enumerate(lab_list):
        r = {"label": lab, "cortical": bool(M.is_cortical([lab])[0])}
        for kind in ("template", "template_loo", "transfer", "pair"):
            v = [d[li] for key, d in per_label.items() if key[0] == kind]
            vv = np.asarray(v, np.float64)
            vv = vv[np.isfinite(vv)]
            r[kind + "_dice"] = float(vv.mean()) if vv.size else float("nan")
        plabel.append(r)
    _write_csv(os.path.join(a.out, "per_label.csv"), plabel)
    summary = {"method": a.method, "n_subjects": n, "n_pairs": len(prow), "n_labels": len(lab_list),
               "identity": a.identity, "convention": None if a.identity else conv.as_dict(),
               "fwd": a.fwd, "inv": a.inv, "atlas": atlas_src, "atlas_seg": seg_src,
               "processed_dir": os.path.abspath(a.processed_dir), "grid": list(shape), "spacing": list(sp),
               "lncc_window": a.lncc_window, "pairs": a.pairs, "seed": a.seed, "group": group,
               "subject_metrics": {k: _summ([r.get(k) for r in rows]) for k in rows[0] if k != "subject"},
               "pair_metrics": ({k: _summ([r.get(k) for r in prow]) for k in prow[0] if k not in ("moving", "fixed")}
                                if prow else {}),
               "seconds": time.time() - t0}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    _print_summary(summary)
    return 0


def _print_summary(s: dict) -> None:
    print("method %s: %d subjects, %d pairs, %d labels" % (s["method"], s["n_subjects"], s["n_pairs"], s["n_labels"]))
    for sec in ("subject_metrics", "pair_metrics"):
        for k, v in s[sec].items():
            if v.get("n"):
                print("  %-28s %.4f +- %.4f  (median %.4f)" % (k, v["mean"], v["std"], v["median"]))
    for k, v in s["group"].items():
        print("  %-28s %s" % (k, v))


# ------------------------------------------------------------------ CLI
def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--processed-dir", required=True, help="e.g. .../data/processed/mindboggle")
    p.add_argument("--label-name", default="dkt31aseg", help="labels_<name>.nii.gz (default %(default)s)")
    p.add_argument("--subjects-file", default=None, help="optional subject list (names or <subject>_T1w dirs)")
    p.add_argument("--fwd", default=None, help="forward-field path template ({subject}, {subject_dir})")
    p.add_argument("--inv", default=None, help="inverse-field path template")
    p.add_argument("--units", choices=("mm", "vox"), default="mm")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m evaluation", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check-convention", help="find component order / signs / roles from saved moved images")
    _common(c)
    c.add_argument("--moved", required=True, help="method's moved (subject -> atlas) image path template")
    c.add_argument("--n-subjects", type=int, default=2)
    c.add_argument("--out", default=None, help="optional JSON with the full ranking")
    c.set_defaults(func=cmd_check_convention)
    r = sub.add_parser("run", help="compute all metrics for one method")
    _common(r)
    r.add_argument("--method", required=True, help="name stored in summary.json")
    r.add_argument("--out", required=True)
    r.add_argument("--identity", action="store_true", help="zero fields (no-registration reference)")
    r.add_argument("--components", default="xyz", help="stored component order, e.g. zyx (default %(default)s)")
    r.add_argument("--signs", default="+++", help="sign per x/y/z axis, e.g. --+ for LPS fields")
    r.add_argument("--swap-roles", action="store_true", help="the method's 'forward' file is our inverse")
    r.add_argument("--atlas", default=None, help="method's atlas image (default: mean of the warped images)")
    r.add_argument("--atlas-seg", default=None, help="method's atlas segmentation (default: majority vote)")
    r.add_argument("--labels", default=None, help="label list file (default: labels present in every subject)")
    r.add_argument("--pairs", default="all", help="'all' (default), 'none' or a number of seeded random pairs")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--lncc-window", type=int, default=9)
    r.add_argument("--hd95", action="store_true", help="also HD95 per label (slow)")
    r.add_argument("--workers", type=int, default=1, help="forked worker processes (Linux)")
    r.add_argument("--field-cache", type=int, default=16, help="fields kept in memory per process")
    r.add_argument("--no-preload-fwd", dest="preload_fwd", action="store_false",
                   help="do not keep all forward fields in memory (default: keep, ~83 MB each at "
                        "160x224x192; every pair needs fwd_i, so this avoids re-reading them per pair)")
    r.set_defaults(func=cmd_run)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    return int(a.func(a))
