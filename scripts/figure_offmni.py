#!/usr/bin/env python
"""Paper figure 2: groupwise registration of images that are NOT pre-aligned to MNI.

Two steps, so the large volumes stay on the cluster and the plot can be iterated locally:

  extract  (cluster, CPU, tf10_nv.sif; last step of slurm/fig2_offmni_predict.sh)
           Reads the perturbed processed group, the predictions of every row and the evaluation
           summaries; writes one small .npz with axial slabs:
             rows        identity (unregistered inputs), rigid, affine, affine_deform
             candidates  the --n-candidates subjects that moved furthest from their MNI-aligned
                         position (perturbation_summary.csv disp_mean_mm = mean |P x - x| over the
                         brain), after dropping subjects with brain FOV loss > --max-lost-frac
             per row     candidate images (input or moved), group mean image (atlas.nii.gz; for the
                         identity row the voxelwise mean of the inputs), label atlas (atlas_seg.nii.gz
                         / majority vote of the inputs) and the per-voxel label agreement = fraction
                         of the N subjects whose warped label equals the majority label
             metrics     evaluation summary.json means (template LNCC / Dice, pair metrics if run,
                         centrality) of every row
           Slab = axial slices between --z-min-mm and --z-max-mm (world z, step --z-step-mm), so the
           plotted slice can be chosen locally.

  plot     (anywhere with matplotlib; seaborn optional, only for the style)
           4 rows x (cases + mean image + label map), one axial slice, high-DPI PNG.

    python scripts/figure_offmni.py extract --processed-dir <perturbed> --root <fig2 out> --out fig2_data.npz
    python scripts/figure_offmni.py plot --data fig2_data.npz --out fig2.png --z-mm 10 --dpi 600

Display: radiological (subject's right on the left of the panel), anterior up, unless
--neurological. Intensity window [0, --vmax] for every image panel. Label colours: FreeSurfer
LUT (read at extract time from $FREESURFER_HOME/FreeSurferColorLUT.txt, else a fixed tab20 cycle).
Python 3.8 compatible (extract runs in tf10_nv.sif: numpy 1.21, scipy 1.4, nibabel 3.1, mpl 3.5).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

ROWS = ("identity", "rigid", "affine", "affine_deform")
ROW_TITLES = {"identity": "Unregistered", "rigid": "Rigid", "affine": "Affine", "affine_deform": "Deformable"}
METRIC_KEYS = ("template_lncc", "template_dice", "template_dice_cortical", "template_dice_subcortical",
               "pair_lncc", "pair_dice")
FS_LUT_CANDIDATES = ("FreeSurferColorLUT.txt",)


# ============================================================================ extract
def _fs_lut_path(explicit=None):
    if explicit:
        return explicit
    roots = [os.environ.get("FREESURFER_HOME", ""), "/usr/local/freesurfer/dev", "/usr/local/freesurfer"]
    for r in roots:
        for name in FS_LUT_CANDIDATES:
            p = os.path.join(r, name)
            if r and os.path.isfile(p):
                return p
    return None


def label_colors(ids, lut_path=None):
    """[len(ids), 3] float RGB in [0, 1]; FreeSurfer LUT when readable, else a tab20 cycle."""
    ids = [int(i) for i in ids]
    table = {}
    path = _fs_lut_path(lut_path)
    if path:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5 and parts[0].isdigit():
                    table[int(parts[0])] = [int(parts[2]) / 255.0, int(parts[3]) / 255.0, int(parts[4]) / 255.0]
    tab20 = [[0.12, 0.47, 0.71], [0.68, 0.78, 0.91], [1.0, 0.5, 0.05], [1.0, 0.73, 0.47], [0.17, 0.63, 0.17],
             [0.6, 0.87, 0.54], [0.84, 0.15, 0.16], [1.0, 0.6, 0.59], [0.58, 0.4, 0.74], [0.77, 0.69, 0.84],
             [0.55, 0.34, 0.29], [0.77, 0.61, 0.58], [0.89, 0.47, 0.76], [0.97, 0.71, 0.82], [0.5, 0.5, 0.5],
             [0.78, 0.78, 0.78], [0.74, 0.74, 0.13], [0.86, 0.86, 0.55], [0.09, 0.75, 0.81], [0.62, 0.85, 0.9]]
    out = []
    for k, i in enumerate(ids):
        if i == 0:
            out.append([0.0, 0.0, 0.0])
        elif i in table:
            out.append(table[i])
        else:
            out.append(tab20[k % len(tab20)])
    return np.asarray(out, np.float32), (path or "tab20 fallback")


def _read_summary_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in list(r):
            if k not in ("subject_dir", "subject"):
                r[k] = float(r[k])
    return {r["subject_dir"]: r for r in rows}


def _metrics(summary_path):
    if not os.path.isfile(summary_path):
        return {}
    with open(summary_path) as f:
        s = json.load(f)
    out = {}
    for sec in ("subject_metrics", "pair_metrics"):
        for k, v in (s.get(sec) or {}).items():
            if k in METRIC_KEYS and v.get("n"):
                out[k] = float(v["mean"])
    for k, v in (s.get("group") or {}).items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    out["n_subjects"] = int(s.get("n_subjects", 0))
    out["n_pairs"] = int(s.get("n_pairs", 0))
    return out


def _slab(vol, ks):
    return np.asarray(vol)[:, :, ks]


def _vote(stack):
    """Majority label (ties -> smaller label, = evaluation.metrics.MajorityVote) and agreement."""
    from evaluation.metrics import MajorityVote
    mv = MajorityVote(stack)
    maj = mv.full()
    agree = (mv.counts.max(axis=0).reshape(mv.shape) / float(stack.shape[0])).astype(np.float32)
    return maj, agree


def cmd_extract(a):
    import nibabel as nib
    from scipy import ndimage
    from evaluation import fields as F
    from evaluation.evaluate import find_subjects

    subs = find_subjects(a.processed_dir, a.label_name)
    n = len(subs)
    ref = nib.load(subs[0]["image"])
    aff = np.asarray(ref.affine, np.float64)
    shape = tuple(int(s) for s in ref.shape[:3])
    spacing = tuple(float(np.linalg.norm(aff[:3, c])) for c in range(3))
    if not np.allclose(aff[:3, :3], np.diag(np.diag(aff[:3, :3])), atol=1e-6) or np.any(np.diag(aff)[:3] <= 0):
        raise SystemExit("expected a positive diagonal grid affine (RAS), got\n%s" % aff)
    zs = np.arange(a.z_min_mm, a.z_max_mm + 1e-6, a.z_step_mm)
    ks = np.unique(np.clip(np.round((zs - aff[2, 3]) / aff[2, 2]).astype(int), 0, shape[2] - 1))
    z_mm = aff[2, 3] + aff[2, 2] * ks
    print("%d subjects, grid %s, slab %d slices z %.0f..%.0f mm" % (n, shape, len(ks), z_mm[0], z_mm[-1]), flush=True)

    # candidates: furthest displaced, FOV loss filtered
    summ_path = os.path.join(a.processed_dir, "perturbation_summary.csv")
    pert = _read_summary_csv(summ_path) if os.path.isfile(summ_path) else {}
    dirs = [s["dir"] for s in subs]
    if a.subjects:
        cand = [d for d in a.subjects]
        bad = [d for d in cand if d not in dirs]
        if bad:
            raise SystemExit("--subjects not in the group: %s" % bad)
    else:
        if not pert:
            raise SystemExit("%s missing: pass --subjects" % summ_path)
        ok = [d for d in dirs if pert[d]["lost_frac"] <= a.max_lost_frac]
        print("%d of %d subjects with FOV loss <= %.2f %%" % (len(ok), n, 100 * a.max_lost_frac))
        cand = sorted(ok, key=lambda d: -pert[d]["disp_mean_mm"])[:a.n_candidates]
    ci = [dirs.index(d) for d in cand]
    print("candidates:", ", ".join("%s (%.1f mm)" % (d, pert[d]["disp_mean_mm"]) if d in pert else d for d in cand))

    out = {"z_mm": z_mm.astype(np.float32), "z_index": ks.astype(np.int32), "spacing": np.asarray(spacing),
           "affine": aff, "candidates": np.asarray(cand), "candidate_subjects": np.asarray([subs[i]["subject"] for i in ci]),
           "n_subjects": np.int32(n)}
    for key in ("disp_mean_mm", "rotation_deg", "total_rotation_deg", "translation_norm_mm", "scale_max_pct",
                "shear_max", "lost_frac"):
        out["cand_" + key] = np.asarray([pert[d].get(key, np.nan) if d in pert else np.nan for d in cand], np.float32)
    shared = [pert[d].get("shared_rotation_deg", 0.0) for d in dirs if d in pert]
    out["shared_rotation_deg"] = np.float32(shared[0] if shared else 0.0)
    meta = {"processed_dir": os.path.abspath(a.processed_dir), "root": os.path.abspath(a.root),
            "label_name": a.label_name, "rows": [], "metrics": {}, "checks": {}}

    # identity row: inputs and their labels
    lab_slabs = np.zeros((n,) + shape[:2] + (len(ks),), np.uint16)
    img_sum = np.zeros(shape[:2] + (len(ks),), np.float64)
    cand_img = {}
    for i, s in enumerate(subs):
        im = _slab(nib.load(s["image"]).dataobj, ks).astype(np.float32)
        img_sum += im
        lab_slabs[i] = _slab(nib.load(s["labels"]).dataobj, ks)
        if i in ci:
            cand_img[i] = im
    maj, agree = _vote(lab_slabs)
    rows_present = ["identity"]
    out["identity_images"] = np.stack([cand_img[i] for i in ci]).astype(np.float16)
    out["identity_mean"] = (img_sum / n).astype(np.float32)
    out["identity_labels"] = maj.astype(np.uint16)
    out["identity_agree"] = agree.astype(np.float16)
    meta["metrics"]["identity"] = _metrics(os.path.join(a.root, "eval", "identity", "summary.json"))
    all_ids = set(np.unique(maj).tolist())

    # registered rows
    conv = F.Convention("xyz", "+++", "mm")
    grid = np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), ks, indexing="ij")).astype(np.float32)
    for row in ROWS[1:]:
        fdir = os.path.join(a.root, row, "fields")
        if not os.path.isfile(os.path.join(fdir, "atlas.nii.gz")):
            print("row %s: no %s/atlas.nii.gz, skipped" % (row, fdir))
            continue
        print("row %s ..." % row, flush=True)
        warped = np.zeros_like(lab_slabs)
        cand_moved = []
        for i, s in enumerate(subs):
            fwd = F.load_field(os.path.join(fdir, "%s_fwd.nii.gz" % s["dir"]), shape, spacing, conv)
            coords = grid + np.moveaxis(fwd[:, :, ks], -1, 0)
            del fwd
            labels = np.asarray(nib.load(s["labels"]).dataobj)
            warped[i] = ndimage.map_coordinates(labels, coords, order=0, mode="constant", cval=0, output=labels.dtype)
            if i in ci:
                cand_moved.append((ci.index(i), _slab(nib.load(os.path.join(fdir, "%s_moved.nii.gz" % s["dir"])).dataobj, ks)))
        cand_moved.sort(key=lambda t: t[0])
        maj, agree = _vote(warped)
        atlas_seg = _slab(nib.load(os.path.join(fdir, "atlas_seg.nii.gz")).dataobj, ks)
        mismatch = float(np.mean(maj != atlas_seg))
        meta["checks"][row] = {"majority_vs_atlas_seg_mismatch_frac": mismatch}
        if mismatch > 1e-3:
            print("WARNING row %s: slab majority vote differs from atlas_seg in %.3f %% of voxels" % (row, 100 * mismatch))
        out[row + "_images"] = np.stack([m for _, m in cand_moved]).astype(np.float16)
        out[row + "_mean"] = _slab(nib.load(os.path.join(fdir, "atlas.nii.gz")).dataobj, ks).astype(np.float32)
        out[row + "_labels"] = atlas_seg.astype(np.uint16)
        out[row + "_agree"] = agree.astype(np.float16)
        meta["metrics"][row] = _metrics(os.path.join(a.root, "eval", row, "summary.json"))
        pj = os.path.join(fdir, "predict.json")
        if os.path.isfile(pj):
            with open(pj) as f:
                p = json.load(f)
            meta["checks"][row]["predict"] = {k: p.get(k) for k in ("status", "branches", "tta", "recenter", "checkpoint")
                                              if k in p}
        all_ids |= set(np.unique(atlas_seg).tolist())
        rows_present.append(row)
    ids = sorted(int(i) for i in all_ids)
    rgb, lut_src = label_colors(ids, a.lut)
    out["lut_ids"] = np.asarray(ids, np.int32)
    out["lut_rgb"] = rgb
    meta["rows"] = rows_present
    meta["lut"] = lut_src
    out["meta_json"] = np.asarray(json.dumps(meta))
    np.savez_compressed(a.out, **out)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote %s (%.1f MB); rows %s" % (a.out, os.path.getsize(a.out) / 1e6, rows_present))
    for r in rows_present:
        m = meta["metrics"].get(r, {})
        print("  %-14s template LNCC %s  Dice %s" % (r, _fmt(m.get("template_lncc")), _fmt(m.get("template_dice"))))
    return 0


def _fmt(v, d=3):
    return "n/a" if v is None else ("%.*f" % (d, v))


# ============================================================================ plot
def _display(sl, neurological):
    """[X, Y] slice -> [Y, X] display: anterior up; radiological (R on the left) by default."""
    d = sl.T[::-1]
    return d if neurological else d[:, ::-1]


def _bbox(panels, thr, margin):
    m = np.zeros(panels[0].shape[:2], bool)
    for p in panels:
        m |= (p if p.ndim == 2 else p.max(-1)) > thr
    r, c = np.nonzero(m)
    if not r.size:
        return slice(None), slice(None)
    return (slice(max(r.min() - margin, 0), r.max() + margin + 1),
            slice(max(c.min() - margin, 0), c.max() + margin + 1))


def label_rgb(labels, agree, lut_ids, lut_rgb, style):
    idx = np.searchsorted(lut_ids, labels)
    idx = np.clip(idx, 0, len(lut_ids) - 1)
    rgb = lut_rgb[idx].astype(np.float32)
    rgb[lut_ids[idx] != labels] = 0.0
    if style == "agreement":
        rgb = rgb * np.asarray(agree, np.float32)[..., None]
    return np.clip(rgb, 0, 1)


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    try:
        import seaborn as sns
        sns.set_theme(context="paper", style="white", font=a.font)
    except ImportError:
        plt.rcParams["font.family"] = a.font
    d = np.load(a.data, allow_pickle=False)
    meta = json.loads(str(d["meta_json"]))
    rows = [r for r in (a.rows.split(",") if a.rows else meta["rows"]) if r in meta["rows"]]
    z_mm = d["z_mm"]
    zi = int(np.argmin(np.abs(z_mm - a.z_mm)))
    n_cand = len(d["candidates"])
    cases = [int(c) for c in a.cases.split(",")] if a.cases else list(range(min(6, n_cand)))
    if any(c < 0 or c >= n_cand for c in cases):
        raise SystemExit("--cases must index the %d candidates" % n_cand)
    lut_ids, lut_rgb = d["lut_ids"], d["lut_rgb"]

    grid = []
    for r in rows:
        imgs = [_display(d[r + "_images"][c][:, :, zi].astype(np.float32), a.neurological) for c in cases]
        mean = _display(d[r + "_mean"][:, :, zi], a.neurological)
        lab = label_rgb(d[r + "_labels"][:, :, zi], d[r + "_agree"][:, :, zi].astype(np.float32),
                        lut_ids, lut_rgb, a.label_style)
        lab = _display_rgb(lab, a.neurological)
        grid.append((imgs, mean, lab))
    if a.crop:
        rs, cs = _bbox([p for row in grid for p in row[0] + [row[1], row[2]]], 0.02, a.margin_px)
        grid = [([im[rs, cs] for im in imgs], mean[rs, cs], lab[rs, cs]) for imgs, mean, lab in grid]
    h, w = grid[0][1].shape[:2]

    n_img = len(cases)
    gap = a.gap
    ratios = [1.0] * n_img + [gap, 1.0, 1.0]
    label_w = 0.22                                    # inches for the row titles
    panel_w = (a.width_in - label_w) / sum(ratios)
    panel_h = panel_w * h / float(w)
    head_h = 0.18 if a.column_titles else 0.02
    fig_h = head_h + panel_h * len(rows) * (1 + a.hspace) + 0.02
    fig = plt.figure(figsize=(a.width_in, fig_h))
    gs = gridspec.GridSpec(len(rows), len(ratios), figure=fig, width_ratios=ratios,
                           left=label_w / a.width_in, right=1.0, bottom=0.01 / fig_h,
                           top=1.0 - head_h / fig_h, wspace=0.03, hspace=a.hspace)
    fs = a.fontsize
    for ri, (r, (imgs, mean, lab)) in enumerate(zip(rows, grid)):
        m = meta["metrics"].get(r, {})
        cols = list(range(n_img)) + [n_img + 1, n_img + 2]
        panels = imgs + [mean, lab]
        for k, (col, p) in enumerate(zip(cols, panels)):
            ax = fig.add_subplot(gs[ri, col])
            if p.ndim == 2:
                ax.imshow(p, cmap="gray", vmin=0.0, vmax=a.vmax, interpolation="nearest")
            else:
                ax.imshow(p, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if col == 0:
                ax.set_ylabel(ROW_TITLES.get(r, r), fontsize=fs + 1, labelpad=3)
            if ri == 0 and a.column_titles:
                if k < n_img:
                    title = a.case_title % (k + 1) if a.case_title else ""
                else:
                    title = "Mean image" if col == n_img + 1 else "Label map"
                ax.set_title(title, fontsize=fs + 1, pad=2)
            txt = None
            if k == n_img and m.get(a.image_metric) is not None:
                txt = "LNCC %.3f" % m[a.image_metric]
            elif k == n_img + 1 and m.get(a.label_metric) is not None:
                txt = "Dice %.3f" % m[a.label_metric]
            elif k < n_img and r == "identity" and a.annotate_inputs:
                dm = float(d["cand_disp_mean_mm"][cases[k]])
                rot = float(d["cand_total_rotation_deg"][cases[k]]) if "cand_total_rotation_deg" in d else np.nan
                if not np.isfinite(rot):
                    rot = float(d["cand_rotation_deg"][cases[k]])
                if np.isfinite(dm):
                    txt = "%.0f\u00b0, %.0f mm" % (rot, dm)
            if txt:
                ax.text(0.03, 0.03, txt, transform=ax.transAxes, color="white", fontsize=fs,
                        ha="left", va="bottom")
    fig.savefig(a.out, dpi=a.dpi, facecolor="white")
    print("wrote %s (%.2f x %.2f in, %d dpi); z = %.0f mm, cases %s, rows %s"
          % (a.out, a.width_in, fig_h, a.dpi, z_mm[zi], [str(d["candidates"][c]) for c in cases], rows))
    for r in rows:
        m = meta["metrics"].get(r, {})
        print("  %-14s template LNCC %s  Dice %s  pair Dice %s" % (r, _fmt(m.get("template_lncc")),
              _fmt(m.get("template_dice")), _fmt(m.get("pair_dice"))))
    return 0


def _display_rgb(rgb, neurological):
    d = np.transpose(rgb, (1, 0, 2))[::-1]
    return d if neurological else d[:, ::-1]


# ============================================================================ CLI
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command")
    sub.required = True
    e = sub.add_parser("extract", help="slabs + metrics -> npz (cluster)")
    e.add_argument("--processed-dir", required=True)
    e.add_argument("--label-name", default="synthseg")
    e.add_argument("--root", required=True, help="slurm/fig2_offmni_predict.sh OUT (<row>/fields, eval/<row>)")
    e.add_argument("--out", required=True, help=".npz path (a .json with the metadata is written next to it)")
    e.add_argument("--n-candidates", type=int, default=10)
    e.add_argument("--max-lost-frac", type=float, default=0.002, help="max brain FOV loss of a candidate (fraction)")
    e.add_argument("--subjects", nargs="+", default=None, help="explicit candidate subject_dirs (overrides the ranking)")
    e.add_argument("--z-min-mm", type=float, default=-20.0)
    e.add_argument("--z-max-mm", type=float, default=40.0)
    e.add_argument("--z-step-mm", type=float, default=2.0)
    e.add_argument("--lut", default=None, help="FreeSurferColorLUT.txt (default: $FREESURFER_HOME)")
    e.set_defaults(func=cmd_extract)
    q = sub.add_parser("plot", help="npz -> PNG")
    q.add_argument("--data", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--z-mm", type=float, default=10.0, help="world z of the axial slice (nearest slab slice)")
    q.add_argument("--cases", default=None, help="comma-separated candidate indices (default 0..5 = 6 furthest)")
    q.add_argument("--rows", default=None, help="comma-separated rows (default all extracted)")
    q.add_argument("--dpi", type=int, default=600)
    q.add_argument("--width-in", type=float, default=6.5, help="figure width (MIDL text width ~6.5 in)")
    q.add_argument("--vmax", type=float, default=0.9)
    q.add_argument("--label-style", choices=("agreement", "plain"), default="agreement",
                   help="agreement: label colour scaled by the fraction of subjects agreeing with the majority")
    q.add_argument("--image-metric", default="template_lncc")
    q.add_argument("--label-metric", default="template_dice")
    q.add_argument("--neurological", action="store_true", help="subject's right on the right")
    q.add_argument("--no-crop", dest="crop", action="store_false")
    q.add_argument("--margin-px", type=int, default=3)
    q.add_argument("--gap", type=float, default=0.12, help="spacer between the cases and the summary columns (panel widths)")
    q.add_argument("--hspace", type=float, default=0.03)
    q.add_argument("--no-column-titles", dest="column_titles", action="store_false")
    q.add_argument("--case-title", default="", help="e.g. 'Subject %%d' (default: no case titles)")
    q.add_argument("--no-annotate-inputs", dest="annotate_inputs", action="store_false",
                   help="drop the rotation / mean brain displacement under the unregistered inputs")
    q.add_argument("--fontsize", type=float, default=6.0)
    q.add_argument("--font", default="DejaVu Sans")
    q.set_defaults(func=cmd_plot)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())
