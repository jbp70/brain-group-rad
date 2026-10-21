"""LNCC loss item 8a: gradient magnitude of each deform loss term (synthetic group, level 4).

Narrow model with the deform decoder, synthetic deform group N = 4, 50 Dice-only training steps
(Adam 1e-3, so the SVF head is non-zero), then for K postprocessed batches (flips on, pairs
p_pdice 0.1) one training-mode forward and, per term, the total and encoder-only gradient norms
(persistent tape): template soft Dice (tmpl), pairwise soft Dice (pdice), smoothness (smooth,
unweighted), ncc_tmpl_loss, ncc_pair_loss, and the objectives
  dice_only  = tmpl + pdice + 0.5 smooth                     (current default)
  dice_lncc  = dice_only + ncc_tmpl_loss + ncc_pair_loss     (recommended 1/1)
  lncc_only  = 0.5 smooth + ncc_tmpl_loss + ncc_pair_loss    (ablation)
Writes grad_probe.json next to this file (means and per-batch values).

Usage: python .agents/tasks/lncc-loss/grad_probe.py [--steps 50] [--batches 8]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)

from tests import tf_setup  # noqa: E402,F401  (TF32 off, GPU limit)
from tests.runtime_helpers import narrow_config, synthetic_group  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from data_loading import params as P  # noqa: E402
from models.group_net import GroupRegLevelModel  # noqa: E402
from runtime import fixed_data as fd  # noqa: E402
from runtime import losses  # noqa: E402
from runtime import optimizers as O  # noqa: E402
from runtime.run import Trainer, TrainConfig  # noqa: E402
from runtime.schedulers import ScheduleConfig  # noqa: E402

N = 4
TERMS = ("tmpl", "pdice", "smooth", "ncc_tmpl_loss", "ncc_pair_loss")


def objectives(t):
    dice = t["tmpl"] + t["pdice"] + 0.5 * t["smooth"]
    ncc = t["ncc_tmpl_loss"] + t["ncc_pair_loss"]
    return {"dice_only": dice, "dice_lncc": dice + ncc, "lncc_only": 0.5 * t["smooth"] + ncc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batches", type=int, default=8)
    a = ap.parse_args()
    dgroup = fd.synthetic_deform_group(4, N, seed=0)
    S = synthetic_group(4, N, "rigid", seed=0)["S"]
    model = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1, deform_decoder=True)).build_model(N)
    cfg = TrainConfig(level=4, group_size=N, steps=a.steps, val_every=0, log_every=0,
                      task_ratio={"deform": 1.0}, seed=0, flip_prob=0.5,
                      schedule=ScheduleConfig("constant", base_lr=1e-3, total_steps=a.steps, warmup_steps=0))
    tr = Trainer(model, O.make_optimizer("adam", lr=1e-3), {"deform": fd.FixedGroupSource(dgroup["batch"])}, S, cfg)
    tr.fit(a.steps)
    last = tr.history[-1]
    print("after %d Dice-only steps: loss %.4f soft_dice_tmpl %.4f u_max %.3f vox grad_norm %.4f"
          % (a.steps, last["loss"], last["soft_dice_tmpl"], last["diag_u_max_vox"], last["grad_norm"]), flush=True)
    window = P.ncc_window_for_level(4)
    enc_refs = {v.ref() for v in model.encoder.trainable_variables}
    tvars = model.trainable_variables
    @tf.function
    def probe(images, onehot, pair_i, pair_j):
        # Graph mode: an eager persistent tape over the deform forward does not fit in 6 GB.
        batch = {"images": images, "onehot": onehot, "pair_i": pair_i, "pair_j": pair_j}
        with tf.GradientTape(persistent=True) as tape:
            out = model(images, task="deform", training=True)
            t = losses.deform_loss(out, batch, lambda_ncc_tmpl=1.0, lambda_ncc_pair=1.0, ncc_window=window)
            objs = objectives(t)
        res = {}
        for k, v in list(((k, t[k]) for k in TERMS)) + list(objs.items()):
            g = tape.gradient(v, tvars)
            pairs = [(gi, vi) for gi, vi in zip(g, tvars) if gi is not None]
            res[k] = v
            res[k + "/grad_norm"] = tf.linalg.global_norm([gi for gi, _ in pairs])
            enc = [gi for gi, vi in pairs if vi.ref() in enc_refs]
            res[k + "/enc_grad_norm"] = tf.linalg.global_norm(enc) if enc else tf.constant(0.0)
        for k in ("ncc_tmpl", "ncc_pair", "soft_dice_tmpl", "soft_dice_pair"):
            res[k] = t[k]
        return res

    rows = []
    for b in range(a.batches):
        batch = tr.postprocess["deform"](*dgroup["batch"])
        out = probe(batch["images"], batch["onehot"], batch["pair_i"], batch["pair_j"])
        row = {"batch": b, "n_pairs": int(batch["pair_i"].shape[0])}
        row.update({k: float(v.numpy()) for k, v in out.items()})
        rows.append(row)
        print("batch %d: %s" % (b, {k: round(v, 5) for k, v in row.items() if k.endswith("grad_norm")}), flush=True)
    keys = [k for k in rows[0] if k != "batch"]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    res = {"setup": {"n": N, "level": 4, "widths": "narrow", "dice_only_steps": a.steps, "lr": 1e-3,
                     "ncc_window": window, "batches": a.batches, "after_training": {
                         k: last[k] for k in ("loss", "soft_dice_tmpl", "soft_dice_pair", "diag_u_max_vox",
                                              "grad_norm")}},
           "mean": mean, "rows": rows}
    path = os.path.join(HERE, "grad_probe.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print("mean:", json.dumps({k: round(v, 5) for k, v in mean.items()}, indent=1))
    print("written", path)


if __name__ == "__main__":
    main()
