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


def free_memory():
    tf.keras.backend.clear_session()
    gc.collect()


def write_fake_export(tmp, n_train=3, n_val=2, level=4, seed=0):
    """Tiny export (manifest + level .npy images + template mask) under <tmp>/export."""
    import csv
    import os
    from runtime import fixed_data as fd
    root = os.path.join(tmp, "export")
    shape = P.LEVEL_SHAPE[level]
    suffix = P.level_suffix(level)
    rows = []
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("val", n_val)):
        for i in range(n):
            sub = "%s_%02d" % (split, i)
            d = os.path.join(root, "LUMIR", sub)
            os.makedirs(d)
            np.save(os.path.join(d, "image%s.npy" % suffix), fd.smooth_phantom(shape, rng)[0])
            rows.append({"dataset": "LUMIR", "subject_dir": sub, "split": split, "contrast": "T1w",
                         "image_l%d" % level: "image%s.npy" % suffix,
                         "shape_l%d" % level: "x".join(map(str, shape))})
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
