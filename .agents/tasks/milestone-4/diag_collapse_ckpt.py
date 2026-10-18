"""Diagnose the training collapse from a checkpoint's variables only (no model build, CPU).

For every norm layer: gamma/beta statistics and the fraction of channels whose ReLU is dead for
standard-normal inputs (beta + 3*|gamma| <= 0 -> ReLU(gamma*z + beta) == 0 for |z| <= 3).
For every kernel: norm. For the heads: kernel/bias norms.

Usage: python diag_collapse_ckpt.py <ckpt_prefix_or_dir>
"""
import sys

import numpy as np
import tensorflow as tf


def main(path):
    prefix = tf.train.latest_checkpoint(path) or path
    print("checkpoint:", prefix)
    r = tf.train.load_checkpoint(prefix)
    names = sorted(n for n in r.get_variable_to_shape_map() if "/.OPTIMIZER_SLOT/" not in n
                   and not n.startswith("optimizer") and "save_counter" not in n)
    groups = {}
    for n in names:
        if not n.endswith(".ATTRIBUTES/VARIABLE_VALUE"):
            continue
        key = n.replace("/.ATTRIBUTES/VARIABLE_VALUE", "")
        groups[key] = r.get_tensor(n)
    gammas = {k: v for k, v in groups.items() if k.endswith("gamma")}
    print("\n== norm layers: gamma/beta (dead = beta + 3|gamma| <= 0) ==")
    for g in sorted(gammas):
        b = g[:-len("gamma")] + "beta"
        gv = np.asarray(gammas[g], np.float64).ravel()
        bv = np.asarray(groups.get(b, np.zeros_like(gv)), np.float64).ravel()
        dead = np.mean(bv + 3.0 * np.abs(gv) <= 0)
        print("%-90s n=%4d |gamma| mean %.3g min %.3g  beta mean %+.3g min %+.3g  dead %.2f"
              % (g[-90:], gv.size, np.abs(gv).mean(), np.abs(gv).min(), bv.mean(), bv.min(), dead))
    print("\n== BN moving statistics ==")
    for k in sorted(groups):
        if k.endswith("moving_variance"):
            v = np.asarray(groups[k], np.float64).ravel()
            m = np.asarray(groups.get(k.replace("moving_variance", "moving_mean")), np.float64).ravel()
            print("%-90s var mean %.3g min %.3g  mean |m| %.3g" % (k[-90:], v.mean(), v.min(), np.abs(m).mean()))
    print("\n== kernels and biases (norms) ==")
    for k in sorted(groups):
        if k.endswith("kernel") or k.endswith("bias"):
            v = np.asarray(groups[k], np.float64)
            print("%-90s shape %-22s norm %.4g max|.| %.4g" % (k[-90:], str(v.shape), np.linalg.norm(v), np.abs(v).max()))
    for k in sorted(groups):
        if k.endswith("step") or k.endswith("iterations"):
            print(k, groups[k])


if __name__ == "__main__":
    main(sys.argv[1])
