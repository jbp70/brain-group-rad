"""Post-hoc group re-centering of the written deform fields (`predict.py run --recenter K`; option
C1 of .agents/tasks/inverse-centrality-investigation/report.md section 4).

The model centres every cascade level's SVF in log space, but the group mean of the final composed
forward displacements is not zero (second-order and composition cross terms). This module removes
it after prediction with ONE common diffeomorphism psi applied to every subject:

  fwd_i' = phi_i o psi        (atlas point x -> psi(x) -> phi_i(psi(x)))
  inv_i' = psi^-1 o phi_i^-1

so the subject-to-subject maps phi_i' o phi_j'^-1 = phi_i o phi_j^-1 do not change (up to one extra
trilinear resampling and the exp(-w) / exp(+w) discretisation at 1 mm).

Fields are canonical voxel displacements [X, Y, Z, 3] on the processed 1 mm grid, as the evaluator
reads them (evaluation/fields.py: phi_i(x) = x + fwd_i(x) maps atlas -> subject i). In
`models.diffeo` terms fwd_i' = compose(fwd_i, p) = p + fwd_i(x + p) and inv_i' = compose(q, inv_i) =
inv_i + q(y + inv_i(y)), with p, q the displacements of psi, psi^-1 (border clamping, trilinear;
the same resampling as `evaluation.fields.compose`).

Rounds (K >= 1), psi = exp(-w) and psi^-1 = exp(+w) by scaling and squaring (T = 7 steps, on the
1 mm grid; `diffeo.exp_and_inverse`), so the pair (fwd', inv') keeps exact-by-construction inverse
semantics:

  m_0 = mean_i fwd_i (float64 sum over all N subjects, accumulated during the model pass)
  w_1 = m_0
  round r: psi_r = exp(-w_r); fwd_i^(r) = phi_i o psi_r, always formed from the ORIGINAL fwd_i
           (each output is resampled once, whatever K is); m_r = mean_i fwd_i^(r) (the residual);
           w_{r+1} = w_r + m_r (first-order update; the probe's `center` mode).
  The last round also forms inv_i^(K) = psi_K^-1 o phi_i^-1 and hands (fwd', inv') to the writer
  (moved images, atlas, atlas segmentation and the fwd / inv files all use them).

The original fields are kept as uncompressed .npy files in a temporary directory (one subject in
memory at a time), so host RAM grows only by the float64 sum and a few single-subject fields.

Centrality is the evaluator's definition: mean / max over the atlas mask of |mean_i fwd_i| (mm);
atlas mask = at least half of the moved images > 0. The per-round values use the mask of the
uncorrected moved images (``mask0``); the final residual is also given on the mask of the corrected
moved images (what `python -m evaluation run` will see). Inverse consistency (``inv_err``, the
evaluator's inv_err_fwd_mm on mask0 and inv_err_inv_mm on the subject mask image > 0) is recorded
before / after for the first `DIAG_SUBJECTS` subjects.
"""
import logging
import os
import shutil
import tempfile
import time

import numpy as np
import tensorflow as tf

from evaluation import fields as F
from models import diffeo

logger = logging.getLogger(__name__)

RECENTER_BRANCHES = ("deform", "affine+deform")
INTEGRATION_STEPS = 7
DIAG_SUBJECTS = 4
METHOD = ("psi = exp(-w), psi^-1 = exp(+w) (scaling and squaring, T = 7, 1 mm grid); fwd_i' = phi_i o psi, "
          "inv_i' = psi^-1 o phi_i^-1, formed from the original fields every round; w_1 = mean_i fwd_i, "
          "w_{r+1} = w_r + mean_i fwd_i^(r)")
MASK_NOTE = ("atlas mask = >= 50 % of the moved images > 0 (evaluation.evaluate); per-round values on the "
             "mask of the uncorrected moved images, final_mask = mask of the corrected moved images")


def exp_pair(w, steps=INTEGRATION_STEPS):
    """(p, q) = displacements of (exp(-w), exp(+w)) for w [X, Y, Z, 3] voxels; TF float32 [1, X, Y, Z, 3]."""
    v = tf.constant(np.asarray(w, np.float32)[None])
    return diffeo.exp_and_inverse(-v, steps, lean=True)


def correct_fwd(fwd, p):
    """phi o psi: p + fwd(x + p) -> numpy float32 [X, Y, Z, 3]."""
    return diffeo.compose(tf.constant(np.asarray(fwd, np.float32)[None]), p, lean=True).numpy()[0]


def correct_inv(inv, q):
    """psi^-1 o phi^-1: inv + q(y + inv(y)) -> numpy float32 [X, Y, Z, 3]."""
    return diffeo.compose(q, tf.constant(np.asarray(inv, np.float32)[None]), lean=True).numpy()[0]


def inverse_errors(fwd, inv, atlas_mask, subject_mask, spacing):
    """Evaluator inv_err_fwd_mm / inv_err_inv_mm (means, mm) and their maxima: |F.compose(fwd, inv)|
    on the atlas mask, |F.compose(inv, fwd)| on the subject mask (TF compose, same resampling)."""
    f = tf.constant(np.asarray(fwd, np.float32)[None])
    i = tf.constant(np.asarray(inv, np.float32)[None])
    e1 = F.norm_mm(diffeo.compose(i, f, lean=True).numpy()[0], spacing)     # inv after fwd
    e2 = F.norm_mm(diffeo.compose(f, i, lean=True).numpy()[0], spacing)     # fwd after inv
    out = {}
    for key, e, m in (("inv_err_fwd", e1, atlas_mask), ("inv_err_inv", e2, subject_mask)):
        v = e[m] if m.any() else e.ravel()
        out[key + "_mm"], out[key + "_max_mm"] = float(v.mean()), float(v.max())
    return out


def centrality(mean_field, mask, spacing):
    """{mean_mm, max_mm} of |mean field| (voxels -> mm) over ``mask`` (all voxels if it is empty)."""
    c = F.norm_mm(np.asarray(mean_field, np.float32), spacing)
    v = c[mask] if mask.any() else c.ravel()
    return {"mean_mm": float(v.mean()), "max_mm": float(v.max())}


def atlas_mask(count, n):
    """count of moved images > 0 per voxel -> evaluator atlas mask (mean >= 0.5, exact in integers)."""
    return 2 * count >= int(n)


class Recenterer(object):
    """Collects the original fields of the model pass (`add`), then runs the rounds (`run`)."""

    def __init__(self, rounds, n, shape, spacing, tmp_root, steps=INTEGRATION_STEPS, n_diag=DIAG_SUBJECTS):
        if int(rounds) < 1:
            raise ValueError("Recenterer needs rounds >= 1, got %r" % rounds)
        self.rounds, self.n, self.steps = int(rounds), int(n), int(steps)
        self.shape, self.spacing = tuple(int(s) for s in shape), tuple(float(s) for s in spacing)
        self.n_diag = min(int(n_diag), self.n)
        os.makedirs(tmp_root, exist_ok=True)
        self.dir = tempfile.mkdtemp(prefix="recenter_", dir=tmp_root)
        self.sum = np.zeros(self.shape + (3,), np.float64)
        self.count = np.zeros(self.shape, np.int32)
        self.seen = np.zeros(self.n, bool)
        self.nbytes = 0
        self.t_add = 0.0

    def _path(self, i, kind):
        return os.path.join(self.dir, "%05d_%s.npy" % (int(i), kind))

    def add(self, i, fwd, inv, moved):
        """Original canonical fields (voxels) and the moved image of subject i (model pass)."""
        t = time.time()
        for kind, a in (("fwd", fwd), ("inv", inv)):
            a = np.asarray(a, np.float32)
            np.save(self._path(i, kind), a)
            self.nbytes += a.nbytes
        self.sum += fwd
        self.count += np.asarray(moved) > 0
        self.seen[int(i)] = True
        self.t_add += time.time() - t

    def load(self, i, kind):
        return np.load(self._path(i, kind))

    def tmp_gb(self):
        return self.nbytes / 1e9

    def write_original(self, write_fn):
        """No correction (non-finite model outputs): write_fn(i, fwd, inv) with the original fields."""
        for i in range(self.n):
            write_fn(i, self.load(i, "fwd"), self.load(i, "inv"))

    def run(self, write_fn, images):
        """K rounds; write_fn(i, fwd', inv') -> moved image is called once per subject in the last
        round. ``images`` [N, X, Y, Z] (subject masks of the inverse diagnostics). Returns the record."""
        if not self.seen.all():
            raise RuntimeError("re-centering: %d of %d subjects missing" % (int((~self.seen).sum()), self.n))
        n, sp = self.n, self.spacing
        mask0 = atlas_mask(self.count, n)
        m = (self.sum / n).astype(np.float32)
        hist = [dict(round=0, **centrality(m, mask0, sp))]
        logger.info("re-centering: %d rounds over %d subjects; centrality before %.4f mm (max %.3f mm, "
                    "atlas mask %d voxels)", self.rounds, n, hist[0]["mean_mm"], hist[0]["max_mm"], int(mask0.sum()))
        w = m.astype(np.float64)
        timing, max_w, diag = {}, [], {"subjects": [], "before": [], "after": []}
        count_final = np.zeros(self.shape, np.int32)
        for r in range(1, self.rounds + 1):
            t0 = time.time()
            last = r == self.rounds
            max_w.append(float(np.max(F.norm_mm(w.astype(np.float32), sp))))
            p, q = exp_pair(w, self.steps)
            s = np.zeros(self.shape + (3,), np.float64)
            for i in range(n):
                fwd = self.load(i, "fwd")
                fwd_c = correct_fwd(fwd, p)
                s += fwd_c
                if not last:
                    continue
                inv = self.load(i, "inv")
                inv_c = correct_inv(inv, q)
                if i < self.n_diag:
                    sm = np.asarray(images[i]) > 0
                    diag["subjects"].append(i)
                    diag["before"].append(inverse_errors(fwd, inv, mask0, sm, sp))
                    diag["after"].append(inverse_errors(fwd_c, inv_c, mask0, sm, sp))
                del fwd, inv
                moved = write_fn(i, fwd_c, inv_c)
                count_final += np.asarray(moved) > 0
                del inv_c
            del p, q
            res = (s / n).astype(np.float32)
            hist.append(dict(round=r, **centrality(res, mask0, sp)))
            timing["round_%d" % r] = time.time() - t0
            logger.info("re-centering round %d / %d: centrality %.4f -> %.4f mm (max %.3f mm), |w| max %.3f mm, "
                        "%.1f s%s", r, self.rounds, hist[r - 1]["mean_mm"], hist[r]["mean_mm"], hist[r]["max_mm"],
                        max_w[-1], timing["round_%d" % r], " (fields written)" if last else "")
            w = w + res
        final_mask = atlas_mask(count_final, n)
        rec = {"rounds": self.rounds, "status": "ok", "method": METHOD, "integration_steps": self.steps,
               "mask": MASK_NOTE, "atlas_mask_voxels": int(mask0.sum()),
               "centrality_by_round": hist,
               "centrality_final_mask": dict(atlas_mask_voxels=int(final_mask.sum()),
                                             **centrality(res, final_mask, sp)),
               "max_abs_w_mm": max_w, "timing_s": dict(timing, collect=self.t_add),
               "tmp_dir": self.dir, "tmp_gb": self.tmp_gb()}
        if diag["subjects"]:
            keys = sorted(diag["before"][0])
            rec["inv_err"] = {"subjects": diag["subjects"],
                              "before": {k: float(np.mean([d[k] for d in diag["before"]])) for k in keys},
                              "after": {k: float(np.mean([d[k] for d in diag["after"]])) for k in keys},
                              "per_subject": {"before": diag["before"], "after": diag["after"]}}
            b, a = rec["inv_err"]["before"], rec["inv_err"]["after"]
            logger.info("re-centering inverse consistency (first %d subjects): inv_err_fwd %.4f -> %.4f mm, "
                        "inv_err_inv %.4f -> %.4f mm", len(diag["subjects"]), b["inv_err_fwd_mm"],
                        a["inv_err_fwd_mm"], b["inv_err_inv_mm"], a["inv_err_inv_mm"])
        return rec

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)
