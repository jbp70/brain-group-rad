"""Paired comparison of two prediction arms on the same subjects (TTA design 9.3, milestone 10).

    python scripts/compare_tta.py --a <arm A dir> --b <arm B dir> --out <dir> [--kind eval|linear]
        [--label-a NAME --label-b NAME] [--n-boot 10000] [--seed 0]

Arm A is the reference (``--tta none``), arm B the alternative (``--tta flip8``); every difference
is B - A. numpy + scipy only (no TF), Python 3.8, scipy >= 1.4.

Inputs read from each arm directory:
  --kind eval (``python -m evaluation run`` output):
    per_subject.csv        required; key column ``subject``, one numeric column per metric
    per_pair.csv           optional; keys ``moving``, ``fixed`` (written only when pairs were evaluated)
    per_subject_label.csv  optional; ``subject``, ``template_dice_<label>`` ... (per-label endpoint)
    summary.json           optional; ``group`` values centrality_mean_mm, mean_logjac_abs,
                           atlas_intensity_std (reported as values only)
  --kind linear (``predict.py linear-eval`` output):
    per_subject.csv        required; ``subject``, ``subject_dir``, center_rms_mm, center_rot_deg,
                           center_trans_mm[, center_scale_pct, center_shear_pct], pair_rms_mm_mean
    per_pair.csv           optional; ``moving``, ``fixed``, pair_rms_mm, pair_rot_deg, ...
Non-numeric columns are ignored. Subject sets (and pair sets) of the two arms must be identical,
else exit 2 (also for a missing per_subject.csv).

Statistics per metric (numeric in both arms; subjects with a non-finite value in either arm are
dropped for that metric): n, mean / median / SD of d = B - A, improved / worse / tied by the
metric's direction (higher is better: dice, ncc, lncc; lower is better: fold, logjac, inv_err,
rms, rot, trans, scale, shear, centrality, disp, hd95), 95 % percentile CI of the mean difference
(paired bootstrap over subjects, --n-boot resamples, seed --seed), two-sided Wilcoxon signed-rank
test (zero_method pratt), sign test, d_z = mean / SD and the matched-pairs rank-biserial r.
Pair metrics: per-subject d_i = mean over the pairs with i as the moving image (description and
Wilcoxon), CI by the subject-cluster bootstrap c^T (W * D) c / c^T W c (subjects drawn with
replacement, all ordered pairs among them, self-pairs excluded).

Families and decisions (eval): primary P1 template_dice, P2 pair_dice (needs the all-pairs
evaluation), P3 fold_fwd_pct, Holm over the available primaries; secondary = every other
per-subject and pair metric (Holm within); per-label template Dice (Wilcoxon per label, Holm over
labels) when both arms have per_subject_label.csv. Linear: all metrics in one family,
pair_rms_mm_mean is the linear endpoint. SESOI: Dice metrics 0.002, fold_fwd_pct
max(0.25 x mean_A, 0.0002), linear pair_rms_mm_mean / pair_rms_mm 0.5 mm. "measurable impact" =
Holm p < 0.05 and the CI excludes 0; "practically relevant" additionally |mean d| >= SESOI;
"no practically relevant effect" when the CI lies inside +-SESOI.

Scope: all subjects were registered as ONE group (one atlas, one central space), so the per-subject
outcomes are not independent draws from a population of groups. The analysis is conditional on this
fixed group; p values and CIs refer to resampling subjects within it and do not generalise to new
groups (stated in compare.md).

Outputs in --out: compare.json (all numbers, inputs, n_boot, seed, versions, scope), compare.csv
(one row per metric), compare.md (tables for the report). Exit 0 ok, 2 input error.
"""
import argparse
import csv
import json
import math
import os
import sys
import warnings

import numpy as np
import scipy
import scipy.stats

SCOPE = ("Scope of inference: all subjects were registered as one group (one atlas, one central "
         "space), so the per-subject outcomes are not independent draws from a population of groups. "
         "The analysis is conditional on this fixed group: it measures the effect of the change on "
         "these subjects' registrations. P values and CIs refer to resampling of subjects within "
         "this group and do not generalise to new groups.")
HIGHER_BETTER = ("dice", "ncc", "lncc")
LOWER_BETTER = ("fold", "logjac", "inv_err", "rms", "rot", "trans", "scale", "shear", "centrality", "disp",
                "hd95")
PRIMARY = ("template_dice", "pair_dice", "fold_fwd_pct")
LINEAR_ENDPOINT = "pair_rms_mm_mean"
GROUP_KEYS = ("centrality_mean_mm", "mean_logjac_abs", "atlas_intensity_std")
ALL_ZERO = "all differences zero"
ALPHA = 0.05
BLOCK = 500


class InputError(Exception):
    """Exit code 2."""


# ------------------------------------------------------------------ reading
def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_table(path, keys):
    """CSV -> (ordered key tuples, {column: float array}) for the numeric columns (non-key)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise InputError("%s has no rows" % path)
    for k in keys:
        if k not in rows[0]:
            raise InputError("%s has no column %r" % (path, k))
    ids = [tuple(r[k] for k in keys) for r in rows]
    if len(set(ids)) != len(ids):
        raise InputError("%s has duplicate %s rows" % (path, "/".join(keys)))
    cols = {}
    for c in rows[0]:
        if c in keys:
            continue
        vals = [_float(r.get(c)) for r in rows]
        if all(v is not None for v in vals):
            cols[c] = np.asarray(vals, np.float64)
    return ids, cols


def _diff_msg(a, b, what):
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    return "%s differ: %d only in A %s, %d only in B %s" % (what, len(only_a), only_a[:5], len(only_b),
                                                            only_b[:5])


def align(ids_a, cols_a, ids_b, cols_b, what):
    """Columns of B reordered to A's key order; InputError when the key sets differ."""
    if set(ids_a) != set(ids_b):
        raise InputError(_diff_msg(ids_a, ids_b, what))
    pos = {k: i for i, k in enumerate(ids_b)}
    order = np.asarray([pos[k] for k in ids_a], np.int64)
    common = [c for c in cols_a if c in cols_b]
    return common, {c: cols_a[c] for c in common}, {c: cols_b[c][order] for c in common}


# ------------------------------------------------------------------ statistics
def direction(name):
    """+1 higher is better, -1 lower is better, 0 unknown."""
    n = name.lower()
    if any(k in n for k in HIGHER_BETTER):
        return 1
    if any(k in n for k in LOWER_BETTER):
        return -1
    return 0


def sesoi(name, kind, mean_a):
    if kind == "linear":
        return 0.5 if name in (LINEAR_ENDPOINT, "pair_rms_mm") else None
    if "dice" in name:
        return 0.002
    if name == "fold_fwd_pct":
        return max(0.25 * float(mean_a), 0.0002)
    return None


def holm(pvals):
    """Holm step-down adjustment; None entries stay None and are not counted."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    m = len(idx)
    out = [None] * len(pvals)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda k: pvals[k])):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        out[i] = running
    return out


def wilcoxon_p(d):
    """(p, note): two-sided Wilcoxon signed-rank, zero_method pratt (scipy 1.4 signature)."""
    d = np.asarray(d, np.float64)
    if d.size == 0:
        return None, "no data"
    if not np.any(d != 0):
        return 1.0, ALL_ZERO
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(scipy.stats.wilcoxon(d, zero_method="pratt")[1])
    except ValueError as e:
        return None, "wilcoxon failed: %s" % e
    if not math.isfinite(p):
        return None, "wilcoxon p not finite"
    return p, None


def sign_p(d):
    """(p, note): two-sided sign test over the untied subjects."""
    d = np.asarray(d, np.float64)
    k, n = int(np.sum(d > 0)), int(np.sum(d != 0))
    if n == 0:
        return 1.0, ALL_ZERO
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bt = getattr(scipy.stats, "binom_test", None)
        p = float(bt(k, n, 0.5)) if bt is not None else float(scipy.stats.binomtest(k, n).pvalue)
    return min(1.0, p), None


def rank_biserial(d):
    """Matched-pairs rank-biserial r = (R+ - R-) / (R+ + R-) over the non-zero |d|."""
    d = np.asarray(d, np.float64)
    d = d[d != 0]
    if d.size == 0:
        return None
    r = scipy.stats.rankdata(np.abs(d))
    tot = float(r.sum())
    return float((r[d > 0].sum() - r[d < 0].sum()) / tot)


def bootstrap_mean_ci(d, n_boot, seed):
    """95 % percentile CI of mean(d) by the paired bootstrap over subjects."""
    d = np.asarray(d, np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, np.float64)
    for s in range(0, n_boot, BLOCK):
        b = min(BLOCK, n_boot - s)
        means[s:s + b] = d[rng.integers(0, d.size, (b, d.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def cluster_bootstrap_ci(D, W, n_boot, seed):
    """(estimate, lo, hi): subject-cluster bootstrap of the mean pair difference.

    D [n, n] pair differences (row = moving, column = fixed), W [n, n] 1 for evaluated pairs (0 on
    the diagonal and for missing / non-finite pairs). Per resample: counts c of n subjects drawn
    with replacement, estimate c^T (W * D) c / c^T W c; percentile CI over the finite resamples."""
    D = np.where(W > 0, np.asarray(D, np.float64), 0.0)
    W = np.asarray(W, np.float64).copy()
    np.fill_diagonal(W, 0.0)
    WD = W * D
    n = D.shape[0]
    est = float(WD.sum() / W.sum()) if W.sum() > 0 else float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, np.float64)
    for s in range(0, n_boot, BLOCK):
        b = min(BLOCK, n_boot - s)
        idx = rng.integers(0, n, (b, n))
        C = np.zeros((b, n), np.float64)
        np.add.at(C, (np.repeat(np.arange(b), n), idx.reshape(-1)), 1.0)
        num = np.einsum("bi,bi->b", C @ WD, C)
        den = np.einsum("bi,bi->b", C @ W, C)
        with np.errstate(invalid="ignore", divide="ignore"):
            vals[s:s + b] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return est, None, None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return est, float(lo), float(hi)


def describe(name, a, b, n_boot, seed, kind, d=None, ci=None, ci_method="paired bootstrap"):
    """Paired statistics of one metric; a / b per-subject values (or d given directly)."""
    if d is None:
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        d = b - a
    d = np.asarray(d, np.float64)
    sgn = direction(name)
    r = {"metric": name, "n": int(d.size), "mean_a": float(np.mean(a)) if np.size(a) else None,
         "mean_b": float(np.mean(b)) if np.size(b) else None, "direction": {1: "higher", -1: "lower", 0: None}[sgn],
         "notes": []}
    if d.size == 0:
        r.update(mean_diff=None, median_diff=None, sd_diff=None, improved=None, worse=None, tied=None,
                 ci_low=None, ci_high=None, ci_method=ci_method, wilcoxon_p=None, sign_p=None, d_z=None,
                 rank_biserial=None, sesoi=None)
        r["notes"].append("no finite paired values")
        return r
    sd = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
    r["mean_diff"], r["median_diff"], r["sd_diff"] = float(d.mean()), float(np.median(d)), sd
    if sgn:
        r["improved"], r["worse"] = int(np.sum(sgn * d > 0)), int(np.sum(sgn * d < 0))
    else:
        r["improved"] = r["worse"] = None
        r["notes"].append("direction unknown (increased %d, decreased %d)" % (int(np.sum(d > 0)), int(np.sum(d < 0))))
    r["tied"] = int(np.sum(d == 0))
    if ci is None:
        ci = bootstrap_mean_ci(d, n_boot, seed)
    r["ci_low"], r["ci_high"], r["ci_method"] = ci[0], ci[1], ci_method
    r["wilcoxon_p"], note = wilcoxon_p(d)
    if note:
        r["notes"].append("wilcoxon: " + note)
    r["sign_p"], note = sign_p(d)
    if note:
        r["notes"].append("sign test: " + note)
    r["d_z"] = float(d.mean() / sd) if sd > 0 else None
    r["rank_biserial"] = rank_biserial(d)
    r["sesoi"] = sesoi(name, kind, r["mean_a"]) if r["mean_a"] is not None else None
    return r


def decide(r):
    """Decision text from holm_p, the CI and the SESOI (design 9.3)."""
    if r.get("mean_diff") is None:
        return "no data"
    p, lo, hi, s = r.get("holm_p"), r.get("ci_low"), r.get("ci_high"), r.get("sesoi")
    have_ci = lo is not None and hi is not None
    measurable = p is not None and p < ALPHA and have_ci and (lo > 0 or hi < 0)
    sgn = direction(r["metric"])
    if measurable:
        txt = "measurable impact"
        if sgn:
            txt += " (B %s)" % ("better" if sgn * r["mean_diff"] > 0 else "worse")
    else:
        txt = "no measurable impact"
    if s is not None:
        if measurable and abs(r["mean_diff"]) >= s:
            txt += "; practically relevant (|mean d| >= SESOI %g)" % s
        elif have_ci and -s < lo and hi < s:
            txt += "; no practically relevant effect (CI inside +-%g)" % s
        else:
            txt += "; practical relevance not established (SESOI %g)" % s
    return txt


def finish_family(rows, family):
    ps = holm([r.get("wilcoxon_p") for r in rows])
    for r, p in zip(rows, ps):
        r["family"] = family
        r["holm_p"] = p
        r["decision"] = decide(r)


# ------------------------------------------------------------------ comparison
def _pair_stats(name, ids_s, ids_p, pa, pb, n_boot, seed, kind):
    """Pair metric: per-subject means over moving pairs (Wilcoxon) + subject-cluster bootstrap CI."""
    n = len(ids_s)
    pos = {k[0]: i for i, k in enumerate(ids_s)}
    D = np.zeros((n, n), np.float64)
    W = np.zeros((n, n), np.float64)
    A = np.full((n, n), np.nan)
    B = np.full((n, n), np.nan)
    for (mv, fx), x, y in zip(ids_p, pa, pb):
        i, j = pos[mv], pos[fx]
        if i != j and np.isfinite(x) and np.isfinite(y):
            D[i, j], W[i, j], A[i, j], B[i, j] = y - x, 1.0, x, y
    has = W.sum(axis=1) > 0
    with np.errstate(invalid="ignore"):
        d_i = (D * W).sum(axis=1)[has] / W.sum(axis=1)[has]
    a_i = np.nanmean(np.where(W > 0, A, np.nan)[has], axis=1) if has.any() else np.zeros(0)
    b_i = np.nanmean(np.where(W > 0, B, np.nan)[has], axis=1) if has.any() else np.zeros(0)
    est, lo, hi = cluster_bootstrap_ci(D, W, n_boot, seed)
    r = describe(name, a_i, b_i, n_boot, seed, kind, d=d_i, ci=(lo, hi), ci_method="subject-cluster bootstrap")
    r["source"] = "pair"
    r["n_pairs"] = int(W.sum())
    r["pair_mean_diff"] = est
    return r


def compare(a_dir, b_dir, kind, n_boot, seed):
    res = {"kind": kind, "notes": []}
    pa, pb = (os.path.join(x, "per_subject.csv") for x in (a_dir, b_dir))
    for p in (pa, pb):
        if not os.path.isfile(p):
            raise InputError("missing %s" % p)
    ids_a, cols_a = read_table(pa, ("subject",))
    ids_b, cols_b = read_table(pb, ("subject",))
    names, ca, cb = align(ids_a, cols_a, ids_b, cols_b, "subject sets")
    res["n_subjects"] = len(ids_a)
    res["subjects"] = [k[0] for k in ids_a]
    subj_rows = {}
    for c in names:
        r = describe(c, ca[c], cb[c], n_boot, seed, kind)
        r["source"] = "subject"
        subj_rows[c] = r
    # pairs
    pair_rows = {}
    qa, qb = (os.path.join(x, "per_pair.csv") for x in (a_dir, b_dir))
    have = [os.path.isfile(qa), os.path.isfile(qb)]
    if all(have):
        pid_a, pcols_a = read_table(qa, ("moving", "fixed"))
        pid_b, pcols_b = read_table(qb, ("moving", "fixed"))
        pnames, pca, pcb = align(pid_a, pcols_a, pid_b, pcols_b, "pair sets")
        subj = set(k[0] for k in ids_a)
        bad = sorted({x for k in pid_a for x in k if x not in subj})
        if bad:
            raise InputError("per_pair.csv names subjects not in per_subject.csv: %s" % bad[:5])
        res["n_pairs"] = len(pid_a)
        for c in pnames:
            pair_rows[c] = _pair_stats(c, ids_a, pid_a, pca[c], pcb[c], n_boot, seed, kind)
    else:
        res["n_pairs"] = 0
        if any(have):
            res["notes"].append("per_pair.csv only in arm %s: pair metrics skipped" % ("A" if have[0] else "B"))
        else:
            res["notes"].append("per_pair.csv absent: pair metrics need the all-pairs evaluation")
    res["primary"], res["secondary"], res["per_label"] = [], [], []
    if kind == "linear":
        fam = list(subj_rows.values()) + list(pair_rows.values())
        for r in fam:
            if r["metric"] == LINEAR_ENDPOINT:
                r["notes"].append("linear endpoint")
        finish_family(fam, "linear")
        res["secondary"] = fam
    else:
        prim = []
        for name in PRIMARY:
            r = subj_rows.pop(name, None) or pair_rows.pop(name, None)
            if r is None:
                note = ("needs all-pairs evaluation" if name == "pair_dice" else "metric absent")
                res["notes"].append("primary %s: %s" % (name, note))
                res["primary"].append({"metric": name, "family": "primary", "missing": True, "decision": note})
                continue
            prim.append(r)
        finish_family(prim, "primary")
        res["primary"] = sorted(res["primary"] + prim, key=lambda r: PRIMARY.index(r["metric"]))
        sec = list(subj_rows.values()) + list(pair_rows.values())
        finish_family(sec, "secondary")
        res["secondary"] = sec
        res["per_label"] = _per_label(a_dir, b_dir, ids_a, n_boot, seed, res["notes"])
        res["group"] = _group(a_dir, b_dir)
    return res


def _per_label(a_dir, b_dir, ids_a, n_boot, seed, notes):
    la, lb = (os.path.join(x, "per_subject_label.csv") for x in (a_dir, b_dir))
    if not (os.path.isfile(la) and os.path.isfile(lb)):
        notes.append("per-label data absent")
        return []
    ida, colsa = read_table(la, ("subject",))
    idb, colsb = read_table(lb, ("subject",))
    names, ca, cb = align(ida, colsa, idb, colsb, "per-label subject sets")
    if set(ida) != set(ids_a):
        raise InputError(_diff_msg(ids_a, ida, "per_subject_label.csv vs per_subject.csv subjects"))
    rows = []
    for c in names:
        if not c.startswith("template_dice_"):
            continue
        r = describe(c, ca[c], cb[c], n_boot, seed, "eval")
        r["source"] = "label"
        r["label"] = c[len("template_dice_"):]
        rows.append(r)
    finish_family(rows, "per_label")
    return rows


def _group(a_dir, b_dir):
    out = {}
    for arm, d in (("a", a_dir), ("b", b_dir)):
        p = os.path.join(d, "summary.json")
        g = {}
        if os.path.isfile(p):
            with open(p) as f:
                g = (json.load(f) or {}).get("group") or {}
        for k in GROUP_KEYS:
            out.setdefault(k, {})[arm] = g.get(k)
    return out


# ------------------------------------------------------------------ outputs
CSV_COLS = ("family", "source", "metric", "n", "n_pairs", "direction", "mean_a", "mean_b", "mean_diff",
            "median_diff", "sd_diff", "improved", "worse", "tied", "ci_low", "ci_high", "ci_method", "wilcoxon_p",
            "holm_p", "sign_p", "d_z", "rank_biserial", "sesoi", "decision", "notes")


def _fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        if not math.isfinite(x):
            return "nan"
        return "%.3g" % x if (x != 0 and (abs(x) < 10 ** -nd or abs(x) >= 1e5)) else ("%." + str(nd) + "f") % x
    return str(x)


def _pfmt(p):
    return "-" if p is None else ("%.2e" % p if p < 1e-3 else "%.4f" % p)


def _table(rows):
    head = ("| metric | n | mean A | mean B | mean d (B-A) | 95 % CI | Wilcoxon p | Holm p | sign p | d_z | r_rb | "
            "B better/worse/tied | decision |")
    out = [head, "|" + "---|" * 13]
    for r in rows:
        if r.get("missing"):
            out.append("| %s | - | - | - | - | - | - | - | - | - | - | - | %s |" % (r["metric"], r["decision"]))
            continue
        ci = "-" if r.get("ci_low") is None else "[%s, %s]" % (_fmt(r["ci_low"], 5), _fmt(r["ci_high"], 5))
        bwt = ("%s/%s/%s" % (r["improved"], r["worse"], r["tied"]) if r.get("improved") is not None
               else "?/?/%s" % r.get("tied"))
        n = "%d" % r["n"] + (" (%d pairs)" % r["n_pairs"] if r.get("n_pairs") is not None else "")
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["metric"], n, _fmt(r.get("mean_a")), _fmt(r.get("mean_b")), _fmt(r.get("mean_diff"), 5), ci,
            _pfmt(r.get("wilcoxon_p")), _pfmt(r.get("holm_p")), _pfmt(r.get("sign_p")), _fmt(r.get("d_z"), 3),
            _fmt(r.get("rank_biserial"), 3), bwt, r.get("decision", "")))
    return out


def write_outputs(res, out, meta):
    os.makedirs(out, exist_ok=True)
    doc = dict(meta)
    doc.update(res)
    doc["scope"] = SCOPE
    with open(os.path.join(out, "compare.json"), "w") as f:
        json.dump(doc, f, indent=1)
    rows = [r for r in res["primary"] if not r.get("missing")] + res["secondary"] + res["per_label"]
    with open(os.path.join(out, "compare.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for r in rows:
            w.writerow(["" if r.get(c) is None else ("; ".join(r[c]) if c == "notes" else r[c]) for c in CSV_COLS])
    md = ["# Paired comparison: %s (B) vs %s (A)" % (meta["label_b"], meta["label_a"]), "",
          SCOPE, "",
          "- kind: %s; A = `%s`; B = `%s`" % (res["kind"], meta["a"], meta["b"]),
          "- subjects: %d; pairs: %d; differences d = B - A; bootstrap %d resamples, seed %d; scipy %s"
          % (res["n_subjects"], res["n_pairs"], meta["n_boot"], meta["seed"], meta["scipy_version"]),
          "- decision: measurable impact = Holm p < 0.05 and the 95 % CI excludes 0; practically relevant = "
          "additionally |mean d| >= SESOI; CI inside +-SESOI = no practically relevant effect"]
    md += ["- note: %s" % n for n in res["notes"]]
    if res["kind"] == "linear":
        md += ["", "## Linear errors (one Holm family; endpoint %s)" % LINEAR_ENDPOINT, ""] + _table(res["secondary"])
    else:
        md += ["", "## Primary endpoints (Holm over the available primaries)", ""] + _table(res["primary"])
        md += ["", "## Secondary metrics (exploratory, Holm within the family)", ""] + _table(res["secondary"])
        md += ["", "## Per-label template Dice (Wilcoxon per label, Holm over labels)", ""]
        pl = res["per_label"]
        if not pl:
            md.append("per-label data absent")
        else:
            sig = [r for r in pl if r.get("holm_p") is not None and r["holm_p"] < ALPHA]
            better = [r for r in sig if r["mean_diff"] > 0]
            md.append("%d labels; Holm p < 0.05: %d (B better %d, B worse %d); labels with mean d > 0: %d"
                      % (len(pl), len(sig), len(better), len(sig) - len(better),
                         sum(1 for r in pl if (r.get("mean_diff") or 0) > 0)))
            if sig:
                md += [""] + _table(sorted(sig, key=lambda r: r["holm_p"]))
        md += ["", "## Group metrics (one value per arm)", "", "| metric | A | B |", "|---|---|---|"]
        for k, v in res.get("group", {}).items():
            md.append("| %s | %s | %s |" % (k, _fmt(v.get("a"), 5), _fmt(v.get("b"), 5)))
    with open(os.path.join(out, "compare.md"), "w") as f:
        f.write("\n".join(md) + "\n")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="arm A directory (reference, e.g. --tta none)")
    p.add_argument("--b", required=True, help="arm B directory (e.g. --tta flip8)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--kind", choices=("eval", "linear"), default="eval",
                   help="eval = `python -m evaluation run` outputs (default), linear = `predict.py linear-eval`")
    p.add_argument("--label-a", default="A", help="name of arm A in compare.md (default %(default)s)")
    p.add_argument("--label-b", default="B", help="name of arm B in compare.md (default %(default)s)")
    p.add_argument("--n-boot", type=int, default=10000, help="bootstrap resamples (default %(default)s)")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed (default %(default)s)")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.n_boot < 1:
        print("ERROR: --n-boot must be >= 1", file=sys.stderr)
        return 2
    try:
        res = compare(a.a, a.b, a.kind, a.n_boot, a.seed)
    except InputError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    meta = {"a": os.path.abspath(a.a), "b": os.path.abspath(a.b), "label_a": a.label_a, "label_b": a.label_b,
            "n_boot": a.n_boot, "seed": a.seed, "scipy_version": scipy.__version__,
            "numpy_version": np.__version__, "alpha": ALPHA}
    write_outputs(res, a.out, meta)
    for r in res["primary"] + (res["secondary"] if a.kind == "linear" else []):
        if not r.get("missing"):
            print("%-22s n %3d  mean d %+.5g  CI [%s, %s]  Holm p %s  %s"
                  % (r["metric"], r["n"], r["mean_diff"] if r["mean_diff"] is not None else float("nan"),
                     _fmt(r.get("ci_low"), 5), _fmt(r.get("ci_high"), 5), _pfmt(r.get("holm_p")), r["decision"]))
    print("wrote %s/compare.{json,csv,md}" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
