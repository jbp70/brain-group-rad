#!/usr/bin/env python
"""Visual QC of the synthetic-contrast generator (synth-contrast design 7.2, plan deviation D-d).

    python scripts/synth_preview.py --data <npy_v2 export root> --level {4,2,1} --n 8 --seed 0 \
        --out <png> [--groups 2] [--split train] [--label-set synthfine] [--prob 1]

Reads the first ``--n`` cases of ``--split`` (images and ``labels_<label-set><suffix>.npy`` via
the export manifest, `data_module.manifest_files`), draws ``--groups`` groups with the TRAINING
SynthConfig defaults (one contrast per group, small per-image jitter; ``data_loading.synth.
draw_synth_params`` + ``synthesize_group``) and writes a PNG grid (matplotlib Agg): row 0 = the
real images, then one row per synthetic group; columns = the cases; middle axial slice (third
array axis). Row titles give the group's blur family and mean FWHM. ``--prob`` is the per-group
synthetic probability (default 1: every group synthetic; a group drawn as real shows the real
images). No training, no checkpoint, CPU TensorFlow: the GPU is hidden with
``tf.config.set_visible_devices([], 'GPU')`` (no memory limit is set). Prints the PNG path and
exits 0; a missing label set exits 2 with a message naming the case / column.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="npy_v2 export root (export_manifest.csv)")
    ap.add_argument("--level", type=int, choices=(4, 2, 1), required=True)
    ap.add_argument("--n", type=int, default=8, help="number of cases (columns), default %(default)s")
    ap.add_argument("--seed", type=int, default=0, help="synthesis generator seed, default %(default)s")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--groups", type=int, default=2, help="synthetic groups (rows), default %(default)s")
    ap.add_argument("--split", default="train", help="manifest split, default %(default)s")
    ap.add_argument("--label-set", default=None, help="fine label set (default: SynthConfig.label_set, synthfine)")
    ap.add_argument("--prob", type=float, default=1.0, help="per-group synthetic probability, default %(default)s")
    return ap


def _hide_gpu(tf):
    """CPU only; a process that already initialised the GPU (e.g. a unittest) keeps it."""
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as e:
        print("synth_preview: GPU already initialised, running on the visible devices (%s)" % e)


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.n < 1 or a.groups < 1:
        ap.error("--n and --groups must be >= 1")
    if not 0.0 <= a.prob <= 1.0:
        ap.error("--prob must be in [0, 1], got %r" % a.prob)
    import numpy as np
    import tensorflow as tf
    _hide_gpu(tf)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from data_loading import data_module as dm
    from data_loading import params as P
    from data_loading import synth

    cfg = P.SynthConfig(prob=float(a.prob))
    if a.label_set:
        cfg.label_set = a.label_set
    rows = dm.read_manifest(os.path.join(a.data, dm.MANIFEST_FILE))
    try:
        img_files = dm.manifest_files(rows, a.level, a.split)[:a.n]
    except (KeyError, ValueError) as e:
        ap.error("no level-%d images in %s: %s" % (a.level, a.data, e))
    try:
        lab_files = dm.manifest_files(rows, a.level, a.split, kind="labels", label_set=cfg.label_set)[:a.n]
    except (KeyError, ValueError) as e:
        ap.error("no '%s' labels at level %d in %s (run export-add-labels first): %s"
                 % (cfg.label_set, a.level, a.data, e))
    if not img_files:
        ap.error("no %s cases at level %d in %s" % (a.split, a.level, a.data))
    n = len(img_files)
    images = np.stack([np.load(os.path.join(a.data, f)).reshape(P.LEVEL_SHAPE[a.level])
                       for f in img_files]).astype(np.float32)
    labels = np.stack([np.load(os.path.join(a.data, f)).reshape(P.LEVEL_SHAPE[a.level])
                       for f in lab_files]).astype(np.uint8)
    gen = tf.random.Generator.from_seed(int(a.seed))
    grid, titles = [images], ["real"]
    lab_t = tf.constant(labels)
    for g in range(a.groups):
        prm = synth.draw_synth_params(gen, n, cfg)
        if bool(prm["flag"].numpy()):
            out = synth.synthesize_group(lab_t, prm, cfg, a.level).numpy()[..., 0]
            fam = "PET" if bool(prm["pet_family"].numpy()) else "MR"
            fwhm = float(np.mean(prm["fwhm_mm"].numpy()))
            titles.append("group %d\n%s blur, FWHM %.1f mm%s" % (
                g, fam, fwhm, ", remask" if bool(prm["remask"].numpy()) else ""))
        else:
            out = images
            titles.append("group %d\nreal (flag off, p %g)" % (g, a.prob))
        grid.append(out)
    z = images.shape[3] // 2
    fig, axes = plt.subplots(len(grid), n, figsize=(1.8 * n, 1.9 * len(grid)), squeeze=False)
    for r, (vol, title) in enumerate(zip(grid, titles)):
        for c in range(n):
            ax = axes[r][c]
            ax.imshow(vol[c, :, :, z].T, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(os.path.basename(os.path.dirname(img_files[c])), fontsize=6)
        axes[r][0].set_ylabel(title, fontsize=6)
    fig.suptitle("synthetic contrast preview: level %d, seed %d, label set %s, split %s"
                 % (a.level, a.seed, cfg.label_set, a.split), fontsize=8)
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(a.out, dpi=100)
    plt.close(fig)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
