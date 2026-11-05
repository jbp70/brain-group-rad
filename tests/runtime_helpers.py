"""Shared helpers of the milestone-4 runtime tests (not a test module).

Import `tests.tf_setup` before this module in TF tests.
"""
import gc

import numpy as np
import tensorflow as tf

from data_loading import params as P
from models import transforms as tfm
from models.config import ModelConfig

# Narrow model (same structure, small widths) so trainer tests run fast on the 8 GB GTX 1080 and
# avoid the full-width cuDNN autotune stalls seen in milestone 3.
NARROW = dict(filters=(16, 32, 32, 48, 64, 96), reductions=(64, 48, 24, 24), dense=(512, 256))
NARROW_ARGS = ["--filters", "16,32,32,48,64,96", "--reductions", "64,48,24,24", "--dense", "512,256"]


def narrow_config(**kw):
    d = dict(NARROW)
    d.update(kw)
    return ModelConfig(**d)


# Tiny model (milestone 8, design §16): small enough for level-2 cascade tests (a frozen level-4
# stage plus a trainable level-2 stage) under the 6000 MB GTX 1080 cap. The deform decoder is
# still off unless ``deform_decoder=True`` is passed (decoder_widths only apply then).
TINY = dict(filters=(8, 16, 16, 16, 16, 16), reductions=(16, 8), dense=(64, 32),
            decoder_widths=(8, 8, 8, 8, 8), dropout=0.0)
TINY_ARGS = ["--filters", "8,16,16,16,16,16", "--reductions", "16,8", "--dense", "64,32",
             "--decoder-widths", "8,8,8,8,8"]


def tiny_config(**kw):
    d = dict(TINY)
    d.update(kw)
    return ModelConfig(**d)


def free_memory():
    tf.keras.backend.clear_session()
    gc.collect()


def fine_labels_from_train18(lab, rng):
    """train18-class label map (uint8, e.g. `fd.synthetic_label_phantom` labels 1..8 read as
    train18 classes) -> synthfine ids (uint8, same shape) with
    ``luts.synthfine_to_train18_table()[fine] == lab`` (synth-contrast tests).

    ``lab``: [X, Y, Z] (x = axis 0). Every class voxel gets a fine id of its class: class 2
    (cortex) is split into 4 spatially coherent parcels (4 equal x slabs of the class's own x
    extent, parcel ids drawn from the 72 cortical ids with ``rng``), classes
    with two (hemispheric) fine ids are split left/right at the x mid-plane, single-id classes
    map directly."""
    from preprocessing import luts
    lab = np.asarray(lab)
    if lab.size and int(lab.max()) >= P.N_TRAIN_LABELS:
        raise ValueError("train18 labels must be < %d" % P.N_TRAIN_LABELS)
    table = luts.synthfine_to_train18_table()[:len(luts.LUTS["synthfine"]["map"])]
    x = np.arange(lab.shape[0]).reshape((-1,) + (1,) * (lab.ndim - 1))
    x = np.broadcast_to(x, lab.shape)
    out = np.zeros(lab.shape, np.uint8)
    for c in np.unique(lab):
        c = int(c)
        if c == 0:
            continue
        ids = np.nonzero(table == c)[0]
        sel = lab == c
        if c == 2:
            parcels = np.sort(rng.choice(ids, size=4, replace=False))
            xs = x[sel]
            lo, span = int(xs.min()), int(xs.max()) - int(xs.min()) + 1
            out[sel] = parcels[np.minimum((xs - lo) * 4 // span, 3)]
        elif ids.size == 2:
            out[sel] = np.where(x[sel] < lab.shape[0] // 2, ids[0], ids[1])
        else:
            out[sel] = ids[0]
    return out


def write_fake_export(tmp, n_train=3, n_val=2, level=4, seed=0, labels=False, fine_labels=False):
    """Tiny export (manifest + level .npy images + template mask) under <tmp>/export.

    ``labels=True``: images come from `fd.synthetic_label_phantom` and each case also gets its
    aligned ``labels_train18<suffix>.npy`` (uint8) and the ``labels_l<k>`` column.
    ``fine_labels=True`` (with ``labels=True``): also ``labels_synthfine<suffix>.npy``
    (`fine_labels_from_train18`, own rng so the images are unchanged), listed in ``labels_l<k>``
    as ``labels_train18<suffix>.npy;labels_synthfine<suffix>.npy``."""
    import csv
    import os
    from runtime import fixed_data as fd
    root = os.path.join(tmp, "export")
    shape = P.LEVEL_SHAPE[level]
    suffix = P.level_suffix(level)
    rows = []
    rng = np.random.default_rng(seed)
    fine_rng = np.random.default_rng(seed + 1) if (labels and fine_labels) else None
    for split, n in (("train", n_train), ("val", n_val)):
        for i in range(n):
            sub = "%s_%02d" % (split, i)
            d = os.path.join(root, "LUMIR", sub)
            os.makedirs(d)
            row = {"dataset": "LUMIR", "subject_dir": sub, "split": split, "contrast": "T1w",
                   "image_l%d" % level: "image%s.npy" % suffix,
                   "shape_l%d" % level: "x".join(map(str, shape))}
            if labels:
                img, lab = fd.synthetic_label_phantom(shape, rng)
                lname = "labels_train18%s.npy" % suffix
                np.save(os.path.join(d, lname), lab)
                if fine_rng is not None:
                    fname = "labels_synthfine%s.npy" % suffix
                    np.save(os.path.join(d, fname), fine_labels_from_train18(lab, fine_rng))
                    lname = lname + ";" + fname
                row["labels_l%d" % level] = lname
            else:
                img = fd.smooth_phantom(shape, rng)[0]
            np.save(os.path.join(d, "image%s.npy" % suffix), img)
            rows.append(row)
    os.makedirs(os.path.join(root, "template"))
    mask = np.zeros(shape, np.uint8)
    mask[10:30, 15:40, 12:36] = 1
    np.save(os.path.join(root, "template", "template_mask%s.npy" % suffix), mask)
    with open(os.path.join(root, "export_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return root


def synthetic_group(level=4, n=4, task="rigid", seed=0, ranges=None, nuisance=True):
    """Phantom group rendered by random transforms -> dict(batch, T, T_target, images, S)."""
    from runtime import fixed_data as fd
    rng = np.random.default_rng(seed)
    vol, mask = fd.smooth_phantom(P.LEVEL_SHAPE[level], rng)
    T, tgt = fd.random_linear_transforms(rng, n, task, ranges, nuisance=nuisance)
    imgs = fd.render_group(vol, T, P.LEVEL_SPACING[level])
    return {"batch": fd.to_dali_batch(imgs, T, tgt, task, level), "T": T, "T_target": tgt,
            "images": imgs, "S": tfm.brain_second_moment(mask, P.LEVEL_SPACING[level]),
            "mask": mask}
