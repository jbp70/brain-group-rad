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
"""
import numpy as np

ROW_NAMES = ("input", "predicted", "ground truth")
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
