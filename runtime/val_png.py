"""Validation PNGs: qualitative alignment check of one fixed validation group.

Figure (one per validated task and validation step, ``<run_dir>/val_png/<task>_step<step:07d>.png``):

    row 1  input       the K selected validation images as rendered with their fixed transforms
    row 2  predicted   the same images resampled into the predicted central space,
                       transform_image(images, A_i^-1) with the model's centered A_i (eval mode,
                       the matrices of the validation forward pass); per-image centered RMS (mm)
                       under each panel
    row 3  truth       resampled with the barycentric targets A*_i^-1 (optional, default on)
    last column        voxelwise mean over the K images of each row (optional, default on):
                       sharp = aligned, blurry = misaligned

Display convention: arrays are [X, Y, Z] RAS-like (design section 2: +X right, +Y anterior,
+Z superior). Each panel is the axial slice at the middle Z index (z = Z // 2), shown
radiologically: anterior at the top, the subject's right on the LEFT of the panel. Grayscale,
fixed intensity window [0, 1] for every panel.

The K images are a fixed seeded choice (`select_indices`), identical at every step. Everything
runs eagerly outside tf.function (the resampling uses `models.layers.transform_image`);
matplotlib is imported lazily (Agg canvas, no pyplot state) so training does not need it when
the PNGs are off.

Deformable figure (milestone 6, ``<run_dir>/val_png/deform_step<step:07d>.png``, same slice and
orientation; built from arrays the Trainer computes with `models.diffeo`):

    row 1  input       the K selected validation images (central / MNI space, A = I)
    row 2  warped      I_i o phi_i (eval-mode u_i)
    row 3  labels      warped hard labels argmax(onehot_i o phi_i) (tab20, background black)
    row 4  det J       Jacobian determinant of phi_i, window [0, 2] (diverging, 1 = white);
                       folding % (J <= 0) under each panel; the 1-voxel border has no J and
                       shows 1
    last column        rows 1-2 voxelwise mean, row 3 the label template (argmax of the mean
                       warped one-hot over the whole group), row 4 the mean J
"""
import numpy as np

ROW_NAMES = ("input", "predicted", "ground truth")
DEFORM_ROW_NAMES = ("input", "warped", "labels", "det J")
JAC_WINDOW = (0.0, 2.0)
JAC_CMAP = "RdBu"
LABEL_CMAP = "tab20"
PANEL_IN = 1.6       # panel width / height in inches
DPI = 100
DISPLAY = "R on left, anterior up"


def select_indices(n, k, seed=0):
    """Fixed seeded choice of min(k, n) of n images (sorted)."""
    k = min(int(k), int(n))
    return np.sort(np.random.default_rng(int(seed)).choice(int(n), k, replace=False))


def axial_display(vol):
    """[X, Y, Z(, 1)] -> middle axial slice as a display array [Y, X]: anterior up, R on left."""
    v = np.asarray(vol)
    if v.ndim == 4:
        v = v[..., 0]
    s = v[:, :, v.shape[2] // 2]           # [X, Y]
    return s.T[::-1, ::-1]                 # rows: Y descending (anterior top); cols: X descending


def resample_central(images, A, spacing):
    """images [K, X, Y, Z, 1] resampled into central space: out_i(y) = in_i(A_i^-1 y). numpy."""
    import tensorflow as tf
    from models import layers as L
    inv = np.linalg.inv(np.asarray(A, np.float64)).astype(np.float32)
    return L.transform_image(tf.constant(np.asarray(images, np.float32)), tf.constant(inv),
                             spacing).numpy()


def build_panels(images, A, A_star, spacing, gt_row=True, mean_col=True):
    """Display arrays of the figure: list of rows, each a list of [Y, X] panels.

    Args: images [K, X, Y, Z, 1] (the selected inputs), A / A_star [K, 4, 4] predicted centered
    matrices and barycentric targets. Rows: input, predicted (, ground truth); each row has K
    panels plus the mean over the K panels when ``mean_col``.
    """
    vols = [np.asarray(images, np.float32), resample_central(images, A, spacing)]
    if gt_row:
        vols.append(resample_central(images, A_star, spacing))
    rows = []
    for v in vols:
        row = [axial_display(x) for x in v]
        if mean_col:
            row.append(np.mean(np.stack(row), axis=0))
        rows.append(row)
    return rows


def make_figure(panels, title, image_errors_mm=None, mean_col=True, labels=None):
    """matplotlib Figure (Agg canvas) of `build_panels` output. Size: PANEL_IN per panel + title.

    ``labels``: column titles of the K image columns (default '#0'..); the mean column is 'mean'.
    ``image_errors_mm``: K values written under the row-2 (predicted) panels.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    n_rows, n_cols = len(panels), len(panels[0])
    k = n_cols - 1 if mean_col else n_cols
    labels = list(labels) if labels is not None else ["#%d" % i for i in range(k)]
    fig = Figure(figsize=figure_size_in(n_rows, n_cols), dpi=DPI)
    FigureCanvasAgg(fig)
    for r, row in enumerate(panels):
        for c, img in enumerate(row):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1)
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(ROW_NAMES[r], fontsize=8)
            if r == 0:
                ax.set_title(labels[c] if c < k else "mean", fontsize=8)
            if r == 1 and image_errors_mm is not None and c < k:
                ax.set_xlabel("%.2f mm" % float(image_errors_mm[c]), fontsize=8)
    fig.suptitle(title, fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.08, top=1.0 - 0.45 / fig.get_figheight(),
                        wspace=0.04, hspace=0.25)
    return fig


def figure_size_in(n_rows, n_cols):
    return (PANEL_IN * n_cols + 0.3, PANEL_IN * n_rows + 0.6)


def figure_size_px(n_rows, n_cols):
    w, h = figure_size_in(n_rows, n_cols)
    return int(round(w * DPI)), int(round(h * DPI))


def figure_title(task, step, center_rms_mm, pair_rms_mm, z):
    return "%s step %d | val center %.2f mm, pair %.2f mm | axial z=%d, %s, window [0, 1]" % (
        task, int(step), float(center_rms_mm), float(pair_rms_mm), int(z), DISPLAY)


def write_val_png(path, images, A, A_star, spacing, title, image_errors_mm, gt_row=True,
                  mean_col=True, labels=None):
    """Build the panels and save the figure to `path`. Returns (panels, figure)."""
    panels = build_panels(images, A, A_star, spacing, gt_row=gt_row, mean_col=mean_col)
    fig = make_figure(panels, title, image_errors_mm, mean_col=mean_col, labels=labels)
    fig.savefig(path, dpi=DPI)
    return panels, fig


# ---------------------------------------------------------------------------- deformable (m6)
def _colormap(name):
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        from matplotlib import cm
        return cm.get_cmap(name)


def label_colors():
    """[21, 3] RGB lookup table: 0 (background) black, then the 20 tab20 colours."""
    tab = np.asarray(_colormap(LABEL_CMAP).colors, np.float64)[:, :3]
    return np.concatenate([np.zeros((1, 3)), tab], axis=0)


def label_display(seg):
    """[X, Y, Z(, 1)] integer labels -> RGB display slice [Y, X, 3] (`axial_display` orientation)."""
    s = np.asarray(axial_display(np.asarray(seg)), np.int64)
    lut = label_colors()
    return lut[np.clip(s, 0, len(lut) - 1)]


def pad_jacobian(jac):
    """Interior det J [K, X-2, Y-2, Z-2] -> [K, X, Y, Z], border = 1 (the slice index matches)."""
    j = np.asarray(jac, np.float64)
    return np.pad(j, [(0, 0), (1, 1), (1, 1), (1, 1)], mode="constant", constant_values=1.0)


def build_deform_panels(images, warped, seg_warped, seg_template, jac, mean_col=True):
    """Display arrays of the deformable figure: 4 rows of [Y, X] panels (row 3: [Y, X, 3] RGB).

    Args: images / warped [K, X, Y, Z(, 1)] float, seg_warped [K, X, Y, Z] int, seg_template
    [X, Y, Z] int (label template of the whole group), jac [K, X-2, Y-2, Z-2] (interior det J).
    Each row has K panels plus, with ``mean_col``, the mean (rows 1, 2, 4) or the template (row 3).
    """
    rows = []
    for v in (np.asarray(images, np.float32), np.asarray(warped, np.float32)):
        row = [axial_display(x) for x in v]
        if mean_col:
            row.append(np.mean(np.stack(row), axis=0))
        rows.append(row)
    row = [label_display(s) for s in np.asarray(seg_warped)]
    if mean_col:
        row.append(label_display(np.asarray(seg_template).reshape(np.asarray(seg_warped).shape[1:4])))
    rows.append(row)
    jf = pad_jacobian(jac)
    row = [axial_display(j) for j in jf]
    if mean_col:
        row.append(np.mean(np.stack(row), axis=0))
    rows.append(row)
    return rows


def make_deform_figure(panels, title, fold_pct=None, mean_col=True, labels=None):
    """matplotlib Figure (Agg canvas) of `build_deform_panels` output (size as `make_figure`).
    ``fold_pct``: K folding percentages written under the row-4 panels."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    n_rows, n_cols = len(panels), len(panels[0])
    k = n_cols - 1 if mean_col else n_cols
    labels = list(labels) if labels is not None else ["#%d" % i for i in range(k)]
    fig = Figure(figsize=figure_size_in(n_rows, n_cols), dpi=DPI)
    FigureCanvasAgg(fig)
    jac_cmap = _colormap(JAC_CMAP)
    for r, row in enumerate(panels):
        for c, img in enumerate(row):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1)
            if r == 2:
                ax.imshow(img, interpolation="nearest")
            elif r == 3:
                ax.imshow(img, cmap=jac_cmap, vmin=JAC_WINDOW[0], vmax=JAC_WINDOW[1], interpolation="nearest")
            else:
                ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(DEFORM_ROW_NAMES[r], fontsize=8)
            if r == 0:
                ax.set_title(labels[c] if c < k else ("mean" if mean_col else ""), fontsize=8)
            if r == 2 and c == k and mean_col:
                ax.set_xlabel("template", fontsize=8)
            if r == 3 and fold_pct is not None and c < k:
                ax.set_xlabel("fold %.3f %%" % float(fold_pct[c]), fontsize=8)
    fig.suptitle(title, fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.08, top=1.0 - 0.45 / fig.get_figheight(),
                        wspace=0.04, hspace=0.25)
    return fig


def deform_figure_title(step, template_dice, pair_dice, neg_jac_pct, z):
    return ("deform step %d | val template Dice %.3f, pair Dice %.3f, J<=0 %.3f %% | axial z=%d, "
            "%s, J window [0, 2]" % (int(step), float(template_dice), float(pair_dice),
                                     float(neg_jac_pct), int(z), DISPLAY))


def write_deform_png(path, images, warped, seg_warped, seg_template, jac, title, fold_pct,
                     labels=None, mean_col=True):
    """Build the deformable panels and save the figure to `path`. Returns (panels, figure)."""
    panels = build_deform_panels(images, warped, seg_warped, seg_template, jac, mean_col=mean_col)
    fig = make_deform_figure(panels, title, fold_pct, mean_col=mean_col, labels=labels)
    fig.savefig(path, dpi=DPI)
    return panels, fig
