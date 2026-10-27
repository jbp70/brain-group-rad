"""Local reproduction of the m9 level-1 deform NaN signature (read-only investigation tool).

Tiny level-2 deform model under mixed_float16, frozen BN (eval statistics), synthetic phantoms.
A decoder conv kernel is scaled by k to push its float16 output past 65504 at some voxels; for
every k we report where the first non-finite tensor appears and what the downstream metrics look
like (the job-10518952 pattern: forward loss NaN, hard Dice finite, LNCC / inverse consistency /
centrality NaN, folding finite). The same weights in a float32 model are run as control.

Run (WSL, tf_gpu):  python .agents/tasks/m9-level1-nan/repro_fp16_overflow.py [stage]
stage = decoder stage whose conv kernel is scaled (d1..d5, default d2).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import tests.tf_setup  # noqa: F401,E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from models import diffeo  # noqa: E402
from models.group_net import GroupRegLevelModel  # noqa: E402
from runtime import fixed_data as fd  # noqa: E402
from runtime import losses  # noqa: E402
from runtime import metrics as M  # noqa: E402
from tests.runtime_helpers import tiny_config  # noqa: E402
from data_loading import params as P  # noqa: E402

LEVEL, N = 2, 3
STAGE = sys.argv[1] if len(sys.argv) > 1 else "d2"


def data():
    rng = np.random.default_rng(0)
    shape = P.LEVEL_SHAPE[LEVEL]
    imgs, labs = zip(*[fd.synthetic_label_phantom(shape, rng) for _ in range(N)])
    return np.stack(imgs)[..., None].astype(np.float32), np.stack(labs).astype(np.int32)


def build(policy):
    tf.keras.mixed_precision.set_global_policy(policy)
    m = GroupRegLevelModel(LEVEL, tiny_config(deform_decoder=True, seed=1, integration_steps=7)).build_model(N)
    m.set_bn_stats_frozen(True)
    tf.keras.mixed_precision.set_global_policy("float32")
    return m


def nf(t):
    t = tf.cast(t, tf.float32)
    return int(tf.reduce_sum(tf.cast(tf.logical_not(tf.math.is_finite(t)), tf.int32)).numpy())


def amax(t):
    t = tf.cast(t, tf.float32)
    t = tf.where(tf.math.is_finite(t), tf.abs(t), tf.zeros_like(t))
    return float(tf.reduce_max(t).numpy())


def forward(model, x):
    """Manual deform forward (DeformDecoder.call order) with per-stage records."""
    rec = []
    xc = tf.cast(x, model.compute_dtype)
    _, skips = model.encoder(xc, training=False)
    for i, s in enumerate(skips):
        rec.append(("enc%d" % i, s.dtype.name, amax(s), nf(s), int(np.prod(s.shape))))
    dec = model.deform_decoder
    h = skips[5]
    for name, skip in zip(("d5", "d4", "d3", "d2", "d1"), (skips[4], skips[3], skips[2], skips[1], skips[0])):
        up, block = getattr(dec, "up_" + name), getattr(dec, name)
        if up is not None:
            h = up(h)
        h = block(tf.concat([h, tf.cast(skip, h.dtype)], axis=-1), training=False)
        rec.append((name, h.dtype.name, amax(h), nf(h), int(np.prod(h.shape))))
    na = dec.svf_norm_act(h, training=False)
    rec.append(("svf_norm_act", na.dtype.name, amax(na), nf(na), int(np.prod(na.shape))))
    v_raw = tf.cast(dec.svf_head(na), tf.float32)
    rec.append(("v_raw", v_raw.dtype.name, amax(v_raw), nf(v_raw), int(np.prod(v_raw.shape))))
    v = diffeo.center_velocities(v_raw)
    u, u_inv = diffeo.exp_and_inverse(v, 7)
    return rec, v_raw, v, u, u_inv


def report(tag, model, x, lab):
    rec, v_raw, v, u, u_inv = forward(model, x)
    first = next((r[0] for r in rec if r[3] > 0), None)
    print("  [%s] first non-finite: %s" % (tag, first))
    for r in rec:
        print("    %-13s %-8s max|.| %-11.4g non-finite %d / %d" % r)
    vox = int(np.prod(v.shape[1:4]))
    per_img = lambda t: [int(c) for c in tf.reduce_sum(tf.cast(tf.reduce_any(  # noqa: E731
        tf.logical_not(tf.math.is_finite(t)), axis=-1), tf.int32), axis=[1, 2, 3]).numpy()]
    print("    non-finite voxels per image: v_raw %s | v (centred) %s | u %s of %d"
          % (per_img(v_raw), per_img(v), per_img(u), vox))
    onehot = tf.one_hot(lab, int(lab.max()) + 1, dtype=tf.float32)
    out = {"u": u, "u_inv": u_inv}
    batch = {"onehot": onehot, "pair_i": tf.constant([0, 1, 2]), "pair_j": tf.constant([1, 2, 0]),
             "images": tf.constant(x)}
    t = losses.deform_loss(out, batch, lambda_ncc_tmpl=1.0, lambda_ncc_pair=1.0, ncc_window=7)
    seg_w = diffeo.warp(tf.constant(lab[..., None]), u, interpolation="nearest")[..., 0].numpy()
    hd = [M.mean_fg_dice(seg_w[i], lab[i], int(lab.max()) + 1) for i in range(N)]
    ic = tf.reduce_mean(tf.norm(diffeo.compose(u, u_inv), axis=-1)).numpy()
    fold = diffeo.folding_fraction(u).numpy()
    cen = np.mean(np.linalg.norm(v.numpy().mean(axis=0), axis=-1))
    print("    loss %.4g | soft dice tmpl %.4g | ncc_tmpl %.4g | hard dice(warped vs own) %s | "
          "inv-cons %.4g | centrality %.4g | folding %s"
          % (float(t["loss"]), float(t["soft_dice_tmpl"]), float(t["ncc_tmpl"]),
             np.round(hd, 3).tolist(), ic, cen, np.round(fold, 5).tolist()))


def main():
    x, lab = data()
    m16 = build("mixed_float16")
    m32 = build("float32")
    rng = np.random.default_rng(1)
    head = m16.deform_decoder.svf_head
    head.kernel.assign(rng.normal(0, 0.02, head.kernel.shape).astype(np.float32))
    kern = getattr(m16.deform_decoder, STAGE).conv.kernel
    k0 = kern.numpy().copy()
    m32.set_weights(m16.get_weights())
    print("policy compute dtypes: m16 %s / m32 %s; scaled kernel %s %s"
          % (m16.compute_dtype, m32.compute_dtype, kern.name, kern.shape))
    for k in (1.0, 1e2, 1e3, 3e3, 1e4, 3e4, 1e5):
        kern.assign(k0 * k)
        m32.set_weights(m16.get_weights())
        print("== kernel x %g" % k)
        report("fp16", m16, x, lab)
        report("fp32", m32, x, lab)


if __name__ == "__main__":
    main()
