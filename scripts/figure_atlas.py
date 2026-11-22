"""Figure 4: Mindboggle atlases of every method, full slices plus zoom boxes.

Inputs (all on the same 1 mm grid, 160 x 224 x 192, world = voxel + origin):
  <src>/<method>/atlas.nii.gz       method atlas (mean of the warped images)
  <src>/affine_mean/atlas.nii.gz    affine-only reference = mean of the processed inputs
                                    (MNI-affine preprocessing); built with --inputs if missing
  <summaries>/<eval dir>/summary.json   evaluation output (group.atlas_intensity_std, ...)

    python scripts/figure_atlas.py --src .agents/tasks/paper-outline/fig4/src \
        --summaries .agents/tasks/paper-outline/src/eval_mindboggle \
        --out .agents/tasks/paper-outline/fig4/fig4.png

Display: radiological (subject's right on the left), anterior / superior up, window [0, --vmax]
for every panel. Zoom boxes are given in world mm: plane:a0,a1,b0,b1 with plane axial (a = x,
b = y at --z-mm) or coronal (a = x, b = z at --y-mm).
Sharpness annotation (--metric):
  std    evaluation's atlas_intensity_std: mean voxelwise SD of the warped images in the atlas
         mask (lower = images agree better); from summary.json
  grad   mean gradient magnitude of the atlas image in a common brain mask (higher = sharper)
  lncc   template LNCC (mean over subjects); from summary.json
  none
Python 3.8 compatible (numpy, scipy, nibabel, matplotlib).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

METHODS = [  # key, panel title, evaluation dir
    ("affine_mean", "Affine (mean)", "identity"),
    ("brainmorph", "BrainMorph", "brainmorph_tps0"),
    ("ants", "ANTs SyN", "ants_syn"),
    ("multimorph", "MultiMorph", "multimorph_baseline"),
    ("ours", "Ours", "r1_v3_gB_tta_flip8_rc2"),
]
DEFAULT_ZOOMS = ["axial:28,73,-12,24", "axial:-35,15,-30,10", "coronal:-62,-12,-32,8"]
ZOOM_COLORS = ["#00c8ff", "#ff9a00", "#ff3cc8", "#7CFC00"]


def _load(p):
    import nibabel as nib
    im = nib.load(p)
    return np.asarray(im.dataobj, np.float32), im.affine


def affine_mean(src, inputs):
    p = os.path.join(src, "affine_mean", "atlas.nii.gz")
    if os.path.exists(p):
        return p
    import nibabel as nib
    files = sorted(glob.glob(os.path.join(inputs, "*.nii.gz")))
    if not files:
        raise SystemExit("no affine_mean atlas and no inputs in %s" % inputs)
    acc, aff = None, None
    for f in files:
        a, aff = _load(f)
        acc = a.astype(np.float64) if acc is None else acc + a
    os.makedirs(os.path.dirname(p), exist_ok=True)
    nib.save(nib.Nifti1Image((acc / len(files)).astype(np.float32), aff), p)
    print("wrote %s (mean of %d inputs)" % (p, len(files)))
    return p


def _load_int(p):
    import nibabel as nib
    return np.asarray(nib.load(p).dataobj).astype(np.int32)


def affine_majority(src, input_labels):
    """<src>/affine_mean/atlas_seg.nii.gz = majority vote of the processed input label maps (ties to
    the smaller label, as evaluation.metrics.MajorityVote); built once."""
    p = os.path.join(src, "affine_mean", "atlas_seg.nii.gz")
    if os.path.exists(p):
        return p
    import nibabel as nib
    files = sorted(glob.glob(os.path.join(input_labels, "*.nii.gz")))
    if not files:
        raise SystemExit("no affine_mean atlas_seg and no label maps in %s" % input_labels)
    labs = np.stack([np.asarray(nib.load(f).dataobj).astype(np.int32) for f in files])
    ids = np.unique(labs)
    best = np.full(labs.shape[1:], -1, np.int32)
    out = np.zeros(labs.shape[1:], np.int32)
    for i in ids:                   # ascending: a later label must beat the count strictly
        c = (labs == i).sum(0, dtype=np.int32)
        win = c > best
        out[win], best[win] = i, c[win]
    os.makedirs(os.path.dirname(p), exist_ok=True)
    nib.save(nib.Nifti1Image(out, nib.load(files[0]).affine), p)
    print("wrote %s (majority of %d label maps, %d ids)" % (p, len(files), len(ids)))
    return p


def grad_sharpness(vol, mask):
    g = np.gradient(vol)
    return float(np.sqrt(sum(x * x for x in g))[mask].mean())


def parse_zoom(s):
    plane, rest = s.split(":")
    v = [float(x) for x in rest.split(",")]
    if plane not in ("axial", "coronal", "sagittal") or len(v) != 4:
        raise SystemExit("bad zoom %r (plane:a0,a1,b0,b1)" % s)
    return plane, min(v[0], v[1]), max(v[0], v[1]), min(v[2], v[3]), max(v[2], v[3])


def get_slice(vol, origin, plane, pos):
    """-> 2D [a, b] slice and the world coordinates of both axes.

    axial: a = x, b = y at pos['axial'] (z); coronal: a = x, b = z at pos['coronal'] (y);
    sagittal: a = y, b = z at pos['sagittal'] (x).
    """
    c = [origin[d] + np.arange(vol.shape[d]) for d in range(3)]
    if plane == "axial":
        return vol[:, :, int(round(pos[plane] - origin[2]))], c[0], c[1]
    if plane == "coronal":
        return vol[:, int(round(pos[plane] - origin[1])), :], c[0], c[2]
    return vol[int(round(pos[plane] - origin[0])), :, :], c[1], c[2]


def to_display(sl):
    """[a, b] -> [b, a] image: b up, a decreasing to the right (axial/coronal: radiological, right on
    the left; sagittal: anterior on the left)."""
    return sl.T[::-1, ::-1]


def crop_up(sl, i0, i1, j0, j1, factor, order, vmax):
    """Crop sl[i0:i1, j0:j1] and upsample it by ``factor`` with a spline of ``order``.

    The spline sees a few voxels beyond the crop (no edge artefacts); sample k of the output sits
    at voxel i0 - 0.5 + (k + 0.5) / factor, so the upsampled panel covers the same extent.
    """
    if factor <= 1:
        return to_display(sl[i0:i1, j0:j1])
    from scipy import ndimage
    p = 4
    pi0, pj0 = max(i0 - p, 0), max(j0 - p, 0)
    sub = sl[pi0:min(i1 + p, sl.shape[0]), pj0:min(j1 + p, sl.shape[1])].astype(np.float64)
    ka = (np.arange((i1 - i0) * factor) + 0.5) / factor - 0.5 + (i0 - pi0)
    kb = (np.arange((j1 - j0) * factor) + 0.5) / factor - 0.5 + (j0 - pj0)
    ga, gb = np.meshgrid(ka, kb, indexing="ij")
    if order == 0:                  # labels: nearest voxel, keep the integer ids
        ia = np.clip(np.floor(ga + 0.5).astype(int), 0, sub.shape[0] - 1)
        ib = np.clip(np.floor(gb + 0.5).astype(int), 0, sub.shape[1] - 1)
        return to_display(sl[pi0:, pj0:][ia, ib])
    up = ndimage.map_coordinates(sub, [ga, gb], order=order, mode="nearest")
    return to_display(np.clip(up, 0.0, max(vmax, float(sl.max()))).astype(np.float32))


def crop_index(coord, lo, hi):
    i0 = int(np.searchsorted(coord, lo))
    i1 = int(np.searchsorted(coord, hi, side="right"))
    return i0, i1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--inputs", default=None, help="processed inputs for the affine mean (default <src>/inputs)")
    ap.add_argument("--summaries", default=None, help="eval_mindboggle root with <eval dir>/summary.json")
    ap.add_argument("--methods", default=",".join(m[0] for m in METHODS))
    ap.add_argument("--ours-eval", default=None, help="override the evaluation dir of 'ours'")
    ap.add_argument("--z-mm", type=float, default=10.0, help="axial slice (world mm)")
    ap.add_argument("--y-mm", type=float, default=-20.0, help="coronal slice (world mm)")
    ap.add_argument("--x-mm", type=float, default=0.0, help="sagittal slice (world mm)")
    ap.add_argument("--planes", default="axial,coronal",
                    help="full-slice rows, in order: plane[:image|labels], plane in axial, coronal, sagittal; "
                         "labels = majority-vote atlas_seg in FreeSurfer LUT colours (Figure 2 'plain' style)")
    ap.add_argument("--input-labels", default=None,
                    help="processed input label maps for the affine majority vote (default <src>/input_labels)")
    ap.add_argument("--lut", default=None, help="FreeSurferColorLUT.txt (default <src>/FreeSurferColorLUT.txt)")
    ap.add_argument("--row-names", default=None,
                    help="'|'-separated row titles, one per row in order (full rows, then zooms); overrides --row-style")
    ap.add_argument("--row-style", default="long", choices=["long", "short"],
                    help="row titles: long = 'Axial z = 10 mm (labels)', short = 'Axial atlas' / 'Axial labels'")
    ap.add_argument("--zoom", action="append", default=None, help="plane:a0,a1,b0,b1 in mm; repeatable")
    ap.add_argument("--metric", default="std", choices=["std", "grad", "lncc", "none"])
    ap.add_argument("--vmax", type=float, default=0.95)
    ap.add_argument("--crop-thr", type=float, default=0.05, help="crop full slices to affine mean > thr")
    ap.add_argument("--margin-mm", type=float, default=3.0)
    ap.add_argument("--width-in", type=float, default=6.5)
    ap.add_argument("--fontsize", type=float, default=6.0)
    ap.add_argument("--font", default="DejaVu Sans")
    ap.add_argument("--box-lw", type=float, default=0.8)
    ap.add_argument("--upsample", type=int, default=8,
                    help="display upsampling factor per voxel (1 = raw voxels)")
    ap.add_argument("--upsample-order", type=int, default=3, help="spline order (1 linear, 3 cubic)")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", required=True, help=".png; a .pdf is written next to it too")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec, patches
    plt.rcParams["font.family"] = a.font

    table = {m[0]: m for m in METHODS}
    keys = [k.strip() for k in a.methods.split(",") if k.strip()]
    for k in keys:
        if k not in table:
            raise SystemExit("unknown method %r (known: %s)" % (k, ", ".join(table)))
    affine_mean(a.src, a.inputs or os.path.join(a.src, "inputs"))
    vols, origin = {}, None
    for k in keys:
        v, aff = _load(os.path.join(a.src, k, "atlas.nii.gz"))
        if not np.allclose(aff[:3, :3], np.eye(3)):
            raise SystemExit("%s: expected a 1 mm axis-aligned grid" % k)
        if origin is None:
            origin, shape = aff[:3, 3], v.shape
        elif v.shape != shape or not np.allclose(aff[:3, 3], origin):
            raise SystemExit("%s: grid differs from %s" % (k, keys[0]))
        vols[k] = v
    ref = vols.get("affine_mean")
    if ref is None:
        ref, _ = _load(os.path.join(a.src, "affine_mean", "atlas.nii.gz"))

    # ---- annotations
    vals = {}
    if a.metric in ("std", "lncc"):
        if not a.summaries:
            raise SystemExit("--metric %s needs --summaries" % a.metric)
        for k in keys:
            ed = a.ours_eval if (k == "ours" and a.ours_eval) else table[k][2]
            s = json.load(open(os.path.join(a.summaries, ed, "summary.json")))
            vals[k] = (s["group"]["atlas_intensity_std"] if a.metric == "std"
                       else s["subject_metrics"]["template_lncc"]["mean"])
    elif a.metric == "grad":
        mask = ref > 0.15
        for k in keys:
            vals[k] = grad_sharpness(vols[k], mask)
    fmt = {"std": "SD %.3f", "grad": "|\u2207| %.3f", "lncc": "LNCC %.3f"}.get(a.metric)
    if vals:
        print("metric %s: %s" % (a.metric, ", ".join("%s %.4f" % (k, vals[k]) for k in keys)))
    if a.metric == "std":
        best = min(vals, key=vals.get)
    elif vals:
        best = max(vals, key=vals.get)
    else:
        best = None

    # ---- panels: list of rows, each row = (kind, height/width ratio, per-method images, zooms drawn)
    pos = {"axial": a.z_mm, "coronal": a.y_mm, "sagittal": a.x_mm}
    zooms = [parse_zoom(z) for z in (a.zoom or DEFAULT_ZOOMS) if z != "none"]
    rows = []
    specs = [p.strip() for p in a.planes.split(",") if p.strip()]
    segs, lut = {}, None
    if any(s.endswith(":labels") for s in specs):
        affine_majority(a.src, a.input_labels or os.path.join(a.src, "input_labels"))
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from figure_offmni import label_colors, label_rgb
        for k in keys:
            segs[k] = np.asarray(_load_int(os.path.join(a.src, k, "atlas_seg.nii.gz")))
        ids = np.unique(np.concatenate([np.unique(s) for s in segs.values()])).astype(np.int32)
        lut_path = a.lut or os.path.join(a.src, "FreeSurferColorLUT.txt")
        lut_rgb, lut_src = label_colors(ids, lut_path if os.path.exists(lut_path) else None)
        lut = (ids, lut_rgb)
        print("label colours: %s (%d ids)" % (lut_src, len(ids)))
    for spec in specs:
        plane, _, what = spec.partition(":")
        what = what or "image"
        if plane not in pos or what not in ("image", "labels"):
            raise SystemExit("bad --planes entry %r (plane[:image|labels])" % spec)
        sl_ref, ca, cb = get_slice(ref, origin, plane, pos)
        m = sl_ref > a.crop_thr
        ia = np.where(m.any(1))[0]
        ib = np.where(m.any(0))[0]
        a0, a1 = ca[ia[0]] - a.margin_mm, ca[ia[-1]] + a.margin_mm
        b0, b1 = cb[ib[0]] - a.margin_mm, cb[ib[-1]] + a.margin_mm
        imgs = []
        i0, i1 = crop_index(ca, a0, a1)
        j0, j1 = crop_index(cb, b0, b1)
        for k in keys:
            if what == "image":
                sl, _, _ = get_slice(vols[k], origin, plane, pos)
                imgs.append(crop_up(sl, i0, i1, j0, j1, a.upsample, a.upsample_order, a.vmax))
            else:
                sl, _, _ = get_slice(segs[k], origin, plane, pos)
                lab = crop_up(sl, i0, i1, j0, j1, a.upsample, 0, None)     # nearest: exact label voxels
                imgs.append(label_rgb(lab, np.ones(lab.shape, np.float32), lut[0], lut[1], "plain"))
        ext = (ca[i1 - 1] + 0.5, ca[i0] - 0.5, cb[j0] - 0.5, cb[j1 - 1] + 0.5)  # radiological: x decreases
        boxes = [(zi, z) for zi, z in enumerate(zooms) if z[0] == plane] if what == "image" else []
        rows.append(("full", plane, imgs, ext, boxes, what))
    for zi, (plane, za0, za1, zb0, zb1) in enumerate(zooms):
        imgs = []
        for k in keys:
            sl, ca, cb = get_slice(vols[k], origin, plane, pos)
            i0, i1 = crop_index(ca, za0, za1)
            j0, j1 = crop_index(cb, zb0, zb1)
            imgs.append(crop_up(sl, i0, i1, j0, j1, a.upsample, a.upsample_order, a.vmax))
        rows.append(("zoom", plane, imgs, zi, None, "image"))

    ncol = len(keys)
    label_w, head_h, gap = 0.16, 0.16, 0.025
    col_w = (a.width_in - label_w - gap * (ncol - 1)) / ncol
    heights = [col_w * r[2][0].shape[0] / r[2][0].shape[1] for r in rows]
    fig_h = head_h + sum(heights) + gap * (len(rows) - 1) + 0.02
    fig = plt.figure(figsize=(a.width_in, fig_h))
    gs = gridspec.GridSpec(len(rows), ncol, figure=fig, height_ratios=heights,
                           left=label_w / a.width_in, right=1.0, bottom=0.02 / fig_h, top=1.0 - head_h / fig_h,
                           wspace=gap / col_w, hspace=gap / np.mean(heights))
    fs = a.fontsize
    row_names = {"axial": "Axial z = %g mm" % a.z_mm, "coronal": "Coronal y = %g mm" % a.y_mm,
                 "sagittal": "Sagittal x = %g mm" % a.x_mm}
    for ri, (kind, plane, imgs, info, boxes, what) in enumerate(rows):
        for ci, k in enumerate(keys):
            ax = fig.add_subplot(gs[ri, ci])
            ax.imshow(imgs[ci], cmap="gray", vmin=0.0, vmax=a.vmax, interpolation="nearest",
                      extent=info if kind == "full" else None)
            ax.set_xticks([]); ax.set_yticks([])
            if kind == "full":
                for sp in ax.spines.values():
                    sp.set_visible(False)
                for zi, (_, za0, za1, zb0, zb1) in boxes:
                    ax.add_patch(patches.Rectangle((za1 + 0.5, zb0 - 0.5), -(za1 - za0 + 1), zb1 - zb0 + 1,
                                                   fill=False, lw=a.box_lw, ec=ZOOM_COLORS[zi % len(ZOOM_COLORS)]))
                if ri == 0 and k in vals:
                    ax.text(0.03, 0.98, fmt % vals[k], transform=ax.transAxes, color="white", fontsize=fs,
                            ha="left", va="top", fontweight="bold" if k == best else "normal",
                            bbox=dict(boxstyle="square,pad=0.15", facecolor="black", edgecolor="none", alpha=0.65))
            else:
                for sp in ax.spines.values():
                    sp.set_color(ZOOM_COLORS[info % len(ZOOM_COLORS)])
                    sp.set_linewidth(a.box_lw * 1.5)
            if ri == 0:
                ax.set_title(table[k][1], fontsize=fs + 1, pad=2,
                             fontweight="bold" if k == "ours" else "normal")
            if ci == 0:
                if a.row_names:
                    names = a.row_names.split("|")
                    if len(names) != len(rows):
                        raise SystemExit("--row-names has %d names for %d rows" % (len(names), len(rows)))
                    name = names[ri]
                elif kind != "full":
                    name = "Zoom %d" % (info + 1)
                elif a.row_style == "short":
                    name = "%s %s" % (plane.capitalize(), "atlas" if what == "image" else "labels")
                else:
                    name = row_names[plane] if what == "image" else "Label map"
                ax.set_ylabel(name, fontsize=fs, labelpad=2)
    fig.savefig(a.out, dpi=a.dpi, facecolor="white")
    pdf = os.path.splitext(a.out)[0] + ".pdf"
    fig.savefig(pdf, facecolor="white")
    print("wrote %s and %s (%.2f x %.2f in, %d dpi); zooms %s" % (a.out, pdf, a.width_in, fig_h, a.dpi, zooms))


if __name__ == "__main__":
    main()
