"""Target grid: the LUMIR 160x224x192 voxel grid with a world-consistent
(MNI152) affine, estimated by `build-target-grid`.

The LUMIR header is untrusted (it is an identity matrix). The working
hypothesis is that LUMIR arrays are MNI152 voxel order cropped/padded per axis,
possibly with the x axis flipped relative to what the header claims. We find
the integer voxel mapping

    mni_idx = P @ lumir_idx + o        (P: signed axis permutation incl. flip)

by maximizing the Pearson NCC between the mean of N normalized LUMIR images
and MNI152_T1_1mm_brain over the overlapping box. Default candidates are
"no flip" and "x flip" (all 48 signed permutations with allow_permutations).
Offsets are searched exhaustively on a block-mean pyramid (factor 4 -> 2 -> 1)
over every plausible crop/pad plus a margin. The corrected affine is
``MNI_affine @ T`` with T the 4x4 voxel mapping.

If the best candidate does not beat the runner-up by ``min_margin`` NCC, the
search refuses (GridAmbiguityError; the CLI exits non-zero) and writes only
target_grid_candidates.json; verify independently, then use force_flip.
The MNI brain is nearly symmetric, so the x flip is often ambiguous (LUMIR:
NCC 0.97107 no-flip vs 0.97127 flip). Verify with organizer labels that know
the sides (LUMIR2025 SanityCheckLabelsTr, SLANT ids: 47/48 = Right/Left
hippocampus) or with the population left/right SynthSeg asymmetry check of
`qc-summary`. For LUMIR both showed that the flip is REQUIRED (subject left =
low voxel x): use --force-flip yes. Voxel data is never changed; only the
affine stamped on the outputs differs.

Outputs in out_dir: target_grid.json, template_lumirgrid.nii.gz (MNI brain on
the corrected grid), template_lumirgrid_mask.nii.gz, lumir_mean.nii.gz.

    python -m preprocessing build-target-grid --config cfg.yaml \\
        --lumir-root /autofs/space/crater_001/datasets/public/LUMIR2025 --out-dir /data/target_grid
"""
import itertools
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize
from scipy import ndimage

from . import nifti
from .intensity import normalize_intensity
from .masks import derive_mask, largest_component

log = logging.getLogger(__name__)
FG_FRACTION_MIN = 0.9


class GridAmbiguityError(RuntimeError):
    """Best and runner-up candidates are too close to choose safely."""


# ---------------------------------------------------------------- candidates
def candidate_list(allow_permutations: bool = False) -> List[Tuple[Tuple[int, int, int], Tuple[bool, bool, bool]]]:
    """(perm, flips) pairs; default: identity with and without an x flip."""
    if not allow_permutations:
        return [((0, 1, 2), (False, False, False)), ((0, 1, 2), (True, False, False))]
    out = []
    for perm in itertools.permutations((0, 1, 2)):
        for flips in itertools.product((False, True), repeat=3):
            out.append((tuple(perm), tuple(flips)))
    return out  # type: ignore[return-value]


def candidate_name(perm, flips) -> str:
    axes = "xyz"
    parts = [("-" if f else "+") + axes[p] for p, f in zip(perm, flips)]
    return "".join(parts)


def apply_candidate(arr: np.ndarray, perm, flips) -> np.ndarray:
    """arr2[j] = arr[i] with j_k = i_perm[k] (or n-1-i_perm[k] when flipped)."""
    out = np.transpose(arr, perm)
    for k, f in enumerate(flips):
        if f:
            out = np.flip(out, axis=k)
    return np.ascontiguousarray(out)


def candidate_matrix(shape: Sequence[int], perm, flips) -> np.ndarray:
    """4x4 voxel map lumir_idx -> candidate-array idx (see apply_candidate)."""
    p = np.zeros((4, 4))
    p[3, 3] = 1.0
    for k in range(3):
        if flips[k]:
            p[k, perm[k]] = -1.0
            p[k, 3] = int(shape[perm[k]]) - 1
        else:
            p[k, perm[k]] = 1.0
    return p


# ---------------------------------------------------------------- scoring
def _overlap(na: Sequence[int], nb: Sequence[int], o: Sequence[int]):
    sa, sb = [], []
    for k in range(3):
        a0 = max(0, -o[k])
        a1 = min(na[k], nb[k] - o[k])
        if a1 <= a0:
            return None
        sa.append(slice(a0, a1))
        sb.append(slice(a0 + o[k], a1 + o[k]))
    return tuple(sa), tuple(sb)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else -1.0


def overlap_score(a: np.ndarray, b: np.ndarray, o: Sequence[int], fg: np.ndarray, fg_total: float
                  ) -> Optional[float]:
    """NCC over the overlap of a placed at offset o inside b; None if too little overlap."""
    ov = _overlap(a.shape, b.shape, o)
    if ov is None:
        return None
    sa, sb = ov
    if fg_total > 0 and fg[sa].sum() / fg_total < FG_FRACTION_MIN:
        return None
    return pearson(a[sa], b[sb])


def block_mean(arr: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return arr.astype(np.float32)
    s = [(n // f) * f for n in arr.shape]
    a = arr[: s[0], : s[1], : s[2]].astype(np.float32)
    return a.reshape(s[0] // f, f, s[1] // f, f, s[2] // f, f).mean(axis=(1, 3, 5))


def offset_range(na: Sequence[int], nb: Sequence[int], margin: int) -> List[Tuple[int, int]]:
    """Inclusive offset range per axis covering all crops/pads plus a margin."""
    return [(min(0, nb[k] - na[k]) - margin, max(0, nb[k] - na[k]) + margin) for k in range(3)]


def search_offsets(a: np.ndarray, b: np.ndarray, margin: int = 10, levels: Sequence[int] = (4, 2, 1)
                   ) -> Tuple[Optional[Tuple[int, int, int]], float, int]:
    """Exhaustive coarse search then local refinement. Returns (offset, ncc, n_evaluated)."""
    rng = offset_range(a.shape, b.shape, margin)
    best = None
    best_s = -np.inf
    n_eval = 0
    prev_f = None
    for f in levels:
        af, bf = block_mean(a, f), block_mean(b, f)
        fg = af > 0.05 * float(af.max()) if af.max() > 0 else af > 0
        fg_total = float(fg.sum())
        if best is None:
            ranges = [range(int(np.floor(lo / f)), int(np.ceil(hi / f)) + 1) for lo, hi in rng]
        else:
            r = int(prev_f // f) + 1
            c = [int(round(best[k] / f)) for k in range(3)]
            ranges = [range(c[k] - r, c[k] + r + 1) for k in range(3)]
        lvl_best, lvl_s = None, -np.inf
        for o in itertools.product(*ranges):
            s = overlap_score(af, bf, o, fg, fg_total)
            n_eval += 1
            if s is not None and s > lvl_s:
                lvl_best, lvl_s = tuple(int(x) * f for x in o), s
        if lvl_best is None:
            return None, -1.0, n_eval
        best, best_s = lvl_best, lvl_s
        prev_f = f
    return best, float(best_s), n_eval  # type: ignore[return-value]


def subvoxel_refine(a: np.ndarray, b: np.ndarray, o: Sequence[int], stride: int = 2) -> Tuple[np.ndarray, float]:
    """Translation-only Powell refinement around integer offset o (report only)."""
    fg = a > 0.05 * float(a.max())
    idx = np.argwhere(fg)
    lo, hi = idx.min(0), idx.max(0) + 1
    grid = np.mgrid[lo[0]:hi[0]:stride, lo[1]:hi[1]:stride, lo[2]:hi[2]:stride].reshape(3, -1).astype(float)
    av = a[tuple(grid.astype(int))]

    def cost(t):
        c = grid + (np.asarray(o, float) + t)[:, None]
        bv = ndimage.map_coordinates(b, c, order=1, mode="constant", cval=0.0)
        return -pearson(av, bv)

    res = optimize.minimize(cost, np.zeros(3), method="Powell",
                            options={"xtol": 1e-2, "ftol": 1e-6, "maxiter": 2000})
    t = np.clip(res.x, -1.5, 1.5)
    return t, float(-cost(t))


# ---------------------------------------------------------------- LUMIR mean
def check_headers(paths: Sequence[str], tol: float = 1e-3) -> Tuple[Tuple[int, ...], np.ndarray, List[Dict[str, Any]]]:
    """Majority (shape, affine) and the list of mismatching files."""
    keys: Dict[Any, List[str]] = {}
    hdr: Dict[Any, Tuple[Tuple[int, ...], np.ndarray]] = {}
    for p in paths:
        img = nifti.load(p)
        shape = nifti.spatial_shape(img)
        k = (shape, tuple(np.round(img.affine, 3).ravel()))
        keys.setdefault(k, []).append(p)
        hdr[k] = (shape, img.affine)
    major = max(keys, key=lambda k: len(keys[k]))
    ref_shape, ref_aff = hdr[major]
    mismatches = []
    for k, ps in keys.items():
        if k == major:
            continue
        for p in ps:
            mismatches.append({"path": p, "shape": list(hdr[k][0]), "affine": hdr[k][1].tolist()})
    return ref_shape, ref_aff, mismatches


def lumir_mean(paths: Sequence[str], n: int, seed: int, cfg: Dict[str, Any]
               ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Mean of a seeded subset of normalized LUMIR images (header-consistent ones)."""
    if not paths:
        raise ValueError("no LUMIR images found")
    shape, aff, mismatches = check_headers(paths, cfg["grid_tolerance_mm"])
    if mismatches:
        log.warning("%d LUMIR files disagree with the majority header and are excluded", len(mismatches))
    bad = {m["path"] for m in mismatches}
    cands = [p for p in paths if p not in bad]
    rng = np.random.default_rng(seed)
    k = min(int(n), len(cands))
    sel = sorted(cands[i] for i in rng.choice(len(cands), size=k, replace=False))
    acc = np.zeros(shape, np.float64)
    freq = np.zeros(shape, np.float64)
    mcfg, ncfg = cfg["mask"], cfg["normalization"]
    for i, p in enumerate(sel):
        data, _ = nifti.load_data(p)
        m = derive_mask(data, mcfg["threshold"], mcfg["fill_holes"], mcfg["largest_component"])
        y, _ = normalize_intensity(data, m, ncfg["upper_percentile"], ncfg["lower_percentile"])
        acc += y
        freq += m
        if (i + 1) % 25 == 0:
            log.info("mean: %d/%d images", i + 1, k)
    info = {"n_found": len(paths), "n_used": k, "seed": int(seed), "selected": sel,
            "header_mismatches": mismatches, "lumir_header_affine": np.asarray(aff).tolist(),
            "shape": list(shape)}
    return (acc / k).astype(np.float32), (freq / k).astype(np.float32), info


# ---------------------------------------------------------------- main entry
def estimate_mapping(mean: np.ndarray, mni: np.ndarray, allow_permutations: bool = False,
                     search_margin: int = 10, subvoxel: bool = True) -> List[Dict[str, Any]]:
    """Score every candidate; returns a list sorted by NCC (best first)."""
    results = []
    for perm, flips in candidate_list(allow_permutations):
        arr = apply_candidate(mean, perm, flips)
        o, s, n_eval = search_offsets(arr, mni, search_margin)
        rec: Dict[str, Any] = {"name": candidate_name(perm, flips), "perm": list(perm), "flips": list(flips),
                               "offset": None if o is None else list(o), "ncc": s, "n_evaluated": n_eval}
        if o is not None:
            p = candidate_matrix(mean.shape, perm, flips)
            t = np.eye(4)
            t[:3, 3] = o
            rec["voxel_map"] = (t @ p).tolist()
            half = [(mni.shape[k] - arr.shape[k]) / 2.0 for k in range(3)]
            rec["symmetric_crop_pad"] = [bool(abs(o[k] - half[k]) <= 1.0) for k in range(3)]
            rec["centered_offset"] = half
            if subvoxel:
                sv, ssv = subvoxel_refine(arr, mni, o)
                rec["subvoxel_shift"] = [float(x) for x in sv]
                rec["subvoxel_ncc"] = ssv
        results.append(rec)
        log.info("candidate %s: offset=%s ncc=%.5f", rec["name"], rec["offset"], s)
    return sorted(results, key=lambda r: r["ncc"], reverse=True)


def build_target_grid(lumir_paths: Sequence[str], mni_path: str, out_dir: str, cfg: Dict[str, Any],
                      mni_mask_path: Optional[str] = None, n: int = 200, seed: int = 0,
                      allow_permutations: bool = False, search_margin: int = 10, min_margin: float = 0.01,
                      force_flip: Optional[str] = None, subvoxel: bool = True, use_subvoxel: bool = False
                      ) -> Dict[str, Any]:
    """Estimate the LUMIR voxel -> MNI world affine and write the target grid files."""
    os.makedirs(out_dir, exist_ok=True)
    mean, freq, info = lumir_mean(lumir_paths, n, seed, cfg)
    mni, mni_img = nifti.load_data(mni_path)
    hdr_aff = np.asarray(info["lumir_header_affine"])
    expected = tuple(cfg["lumir_expected_shape"])
    if tuple(mean.shape) != expected:
        log.warning("LUMIR shape %s != lumir_expected_shape %s", mean.shape, expected)
    # Header check (report only; the header is untrusted).
    hdr_resampled = nifti.resample_world(mni, mni_img.affine, mean.shape, hdr_aff, order=1)
    info["header_ncc"] = pearson(mean, hdr_resampled)
    results = estimate_mapping(mean, mni, allow_permutations, search_margin, subvoxel)
    valid = [r for r in results if r["offset"] is not None]
    if not valid:
        raise GridAmbiguityError("no candidate overlapped the MNI template; check inputs")
    margin = valid[0]["ncc"] - valid[1]["ncc"] if len(valid) > 1 else float("inf")
    info.update({"candidates": results, "ncc_margin": margin, "min_margin": min_margin,
                 "mni_path": os.path.abspath(mni_path), "mni_affine": mni_img.affine.tolist(),
                 "mni_shape": list(mni.shape)})
    chosen = valid[0]
    if force_flip is not None:
        want = force_flip == "yes"
        forced = [r for r in valid if r["perm"] == [0, 1, 2] and r["flips"] == [want, False, False]]
        if not forced:
            raise GridAmbiguityError("force_flip=%s: candidate not evaluated" % force_flip)
        chosen = forced[0]
        info["forced"] = force_flip
        log.warning("using forced flip=%s (ncc %.5f; best was %s %.5f)", force_flip, chosen["ncc"],
                    valid[0]["name"], valid[0]["ncc"])
    elif margin < min_margin:
        cand_path = os.path.join(out_dir, "target_grid_candidates.json")
        with open(cand_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(info), f, indent=2)
        raise GridAmbiguityError(
            "ambiguous LUMIR orientation: best %s (ncc %.5f) beats %s (ncc %.5f) by only %.5f < min_margin %.5f. "
            "Candidates written to %s. The image correlation cannot tell left from right here; verify "
            "independently before choosing --force-flip yes|no: (a) organizer labels with known sides (LUMIR2025 "
            "SanityCheckLabelsTr: Right-* structures must land on the subject's right in the corrected affine), "
            "or (b) process ~20+ cases with each candidate and run qc-summary: its left/right SynthSeg "
            "asymmetry check warns when the population pattern is reversed. For LUMIR this was settled on "
            "2026-10-15: --force-flip yes."
            % (valid[0]["name"], valid[0]["ncc"], valid[1]["name"], valid[1]["ncc"], margin, min_margin, cand_path))
    t = np.asarray(chosen["voxel_map"], float)
    if use_subvoxel and "subvoxel_shift" in chosen:
        # Sub-voxel shift is in candidate-array index units == MNI voxel units.
        t[:3, 3] += np.asarray(chosen["subvoxel_shift"])
    affine = mni_img.affine @ t
    template = nifti.resample_world(mni, mni_img.affine, mean.shape, affine, order=1).astype(np.float32)
    if mni_mask_path:
        mdata, mimg = nifti.load_data(mni_mask_path)
        tmask = nifti.resample_world((mdata > 0).astype(np.uint8), mimg.affine, mean.shape, affine, order=0) > 0
    else:
        tmask = derive_mask(template, 0.0)
    template[~tmask] = 0.0
    paths = {
        "template": os.path.join(out_dir, "template_lumirgrid.nii.gz"),
        "template_mask": os.path.join(out_dir, "template_lumirgrid_mask.nii.gz"),
        "lumir_mean": os.path.join(out_dir, "lumir_mean.nii.gz"),
        "grid": os.path.join(out_dir, "target_grid.json"),
    }
    nifti.save_like(template, affine, paths["template"], np.float32, xform_code=4)
    nifti.save_like(tmask.astype(np.uint8), affine, paths["template_mask"], np.uint8, xform_code=4)
    nifti.save_like(mean, affine, paths["lumir_mean"], np.float32, xform_code=4)
    lumir_mask = largest_component(ndimage.binary_fill_holes(freq >= 0.5))
    info.update({
        "shape": list(mean.shape), "affine": affine.tolist(), "voxel_map": t.tolist(),
        "chosen": chosen["name"], "flip_x": bool(chosen["flips"][0]) if chosen["perm"] == [0, 1, 2] else None,
        "offset": chosen["offset"], "ncc": chosen["ncc"], "used_subvoxel": bool(use_subvoxel),
        "integer_mapping": not use_subvoxel,
        "dice_lumir_mean_mask_vs_template_mask": float(
            2 * np.logical_and(lumir_mask, tmask).sum() / max(1, lumir_mask.sum() + tmask.sum())),
        "outputs": paths,
    })
    with open(paths["grid"], "w", encoding="utf-8") as f:
        json.dump(_jsonable(info), f, indent=2)
    return info


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


class Target:
    """The build-target-grid outputs used by `run`/`synthseg`/QC (lazy data loading)."""

    def __init__(self, cfg: Dict[str, Any], env=None):
        from .config import target_paths
        tp = target_paths(cfg, env)
        for k in ("grid", "template", "template_mask"):
            if not os.path.isfile(tp[k]):
                raise FileNotFoundError("target %s not found: %s (run build-target-grid and set target_dir)"
                                        % (k, tp[k]))
        self.tol = float(cfg["grid_tolerance_mm"])
        self.grid = load_target_grid(tp["grid"])
        self.template_path = tp["template"]
        self.mask_path = tp["template_mask"]
        self.img = nifti.load(self.template_path)
        self.shape = nifti.spatial_shape(self.img)
        self.affine = np.asarray(self.img.affine, float)
        if not nifti.grid_matches(self.shape, self.affine, self.grid["shape"], self.grid["affine"], self.tol):
            raise ValueError("%s does not match target_grid.json (shape/affine)" % self.template_path)
        self._data = None
        self._mask = None
        self._sha = None

    def data(self) -> np.ndarray:
        if self._data is None:
            self._data = nifti.load_data(self.template_path)[0]
        return self._data

    def mask_data(self) -> np.ndarray:
        if self._mask is None:
            self._mask = nifti.load_data(self.mask_path, np.uint8)[0] > 0
        return self._mask

    def sha256(self) -> str:
        if self._sha is None:
            from .meta import sha256_file
            self._sha = sha256_file(self.template_path)
        return self._sha

    def state(self, shape: Sequence[int], affine) -> str:
        """'target' (already corrected), 'header' (raw LUMIR header -> stamp), or 'mismatch'."""
        if tuple(shape) != tuple(self.shape):
            return "mismatch"
        if nifti.affines_match(affine, self.affine, self.tol):
            return "target"
        if nifti.affines_match(affine, self.grid["lumir_header_affine"], self.tol):
            return "header"
        return "mismatch"


def load_target_grid(path: str) -> Dict[str, Any]:
    """Read target_grid.json -> dict with numpy 'affine' / 'lumir_header_affine' and tuple 'shape'."""
    with open(path, "r", encoding="utf-8") as f:
        g = json.load(f)
    for k in ("shape", "affine", "lumir_header_affine"):
        if k not in g:
            raise ValueError("%s: missing key %r" % (path, k))
    g["shape"] = tuple(int(s) for s in g["shape"])
    g["affine"] = np.asarray(g["affine"], float)
    g["lumir_header_affine"] = np.asarray(g["lumir_header_affine"], float)
    return g
