"""ANTs groupwise template construction (rigid / affine / deformable SyGN) in the MultiMorph format.

Runs ``antsMultivariateTemplateConstruction2.sh`` (Avants et al. 2010, SyGN; ANTs 2.4.4) on all
subjects of a processed dataset (one common grid), then converts every subject's final
transforms into the evaluation format shared with baselines/brainmorph and baselines/synthmorph
(``--components xyz --signs +++ --units mm``, see evaluation/fields.py):

    <out>/<type>/<subject_dir>_fwd.nii.gz   [X,Y,Z,3] mm, atlas grid: x + fwd(x) = atlas -> subject
    <out>/<type>/<subject_dir>_inv.nii.gz   [X,Y,Z,3] mm, subject grid: y + inv(y) = subject -> atlas
    <out>/<type>/<subject_dir>_moved.nii.gz / _moved_seg.nii.gz, atlas.nii.gz, atlas_seg.nii.gz
    <out>/<type>/matrices.npz               rigid / affine: 4x4 fwd / inv (voxel and world mm)
    <out>/<type>/ants_template0.nii.gz      ANTs' own final template (one shape update later)
    <out>/<type>/ants_check.json            our moved image vs ANTs' WarpedToTemplate, field checks
    <out>/<type>/work/                      ANTs working folder (transforms, logs)

Transforms: the script registers every subject (moving) to the template (fixed). fwd is the
composite ``-t <s>1Warp.nii.gz -t <s>0GenericAffine.mat`` (ANTs' own command for warping the
subject into template space; affine only for rigid / affine), inv is
``-t [<s>0GenericAffine.mat,1] -t <s>1InverseWarp.nii.gz``; both are written as dense
displacement fields by antsApplyTransforms (``-o [field,1]``, physical LPS mm) on the processed
grid and converted to grid-axis mm. For rigid / affine inv is the exact inverse; for SyN it is
ANTs' inverse warp. The transforms of the last iteration map to the template BEFORE the final
shape update, so atlas.nii.gz is the mean of our moved images (consistent with the fields);
ANTs' final template is kept for reference.

Defaults per type (override with --ants-args, appended last so they win):
    rigid       -t Rigid  -m MI -q 1000x500x250x100 -f 6x4x2x1 -s 4x2x1x0vox
    affine      -t Affine -m MI -q 1000x500x250x100 -f 6x4x2x1 -s 4x2x1x0vox  (Rigid then Affine)
    deformable  -t SyN    -m CC -q 100x100x70x20    -f 6x4x2x1 -s 3x2x1x0vox  (Rigid + Affine (MI) + SyN)
    all         -i 4 template iterations, -r 1 (rigid init to the initial average), -n 0 (inputs are
                already bias corrected), -c 2 -j <cpus> local parallel, ITK threads 1 per registration

Run with numpy/scipy/nibabel python (e.g. tf10_nv.sif) and ANTs reachable via --ants-path:
    python baselines/ants/run_ants_groupwise.py --self-test --ants-path <ANTs bin>
    python baselines/ants/run_ants_groupwise.py --type deformable --processed-dir P --out O --cpus 16
See slurm/ants_groupwise.sh.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import nibabel as nib
import numpy as np
from scipy import io as sio

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from evaluation import fields as F  # noqa: E402
from evaluation.metrics import MajorityVote  # noqa: E402

DEFAULT_ANTS = "/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/ANTS/2.4.4/bin"
PRESETS = {
    "rigid": ["-t", "Rigid", "-m", "MI", "-q", "1000x500x250x100", "-f", "6x4x2x1", "-s", "4x2x1x0vox"],
    "affine": ["-t", "Affine", "-m", "MI", "-q", "1000x500x250x100", "-f", "6x4x2x1", "-s", "4x2x1x0vox"],
    "deformable": ["-t", "SyN", "-m", "CC", "-q", "100x100x70x20", "-f", "6x4x2x1", "-s", "3x2x1x0vox"],
}
PREFIX = "T_"
LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ data
def find_subjects(processed_dir, label_name, subjects_file=None):
    dirs = sorted(d for d in os.listdir(processed_dir)
                  if os.path.isfile(os.path.join(processed_dir, d, "image.nii.gz")))
    if subjects_file:
        with open(subjects_file) as f:
            want = [ln.strip() for ln in f if ln.strip()]
        keep = []
        for w in want:
            m = [d for d in dirs if d == w or d == w + "_T1w"]
            if not m:
                raise FileNotFoundError("subject %s not in %s" % (w, processed_dir))
            keep.append(m[0])
        dirs = keep
    subs = []
    for d in dirs:
        if "." in d:
            raise ValueError("subject folder %r contains '.', which ANTs' file naming truncates" % d)
        lab = os.path.join(processed_dir, d, "labels_%s.nii.gz" % label_name)
        subs.append({"dir": d, "image": os.path.join(processed_dir, d, "image.nii.gz"),
                     "labels": lab if os.path.isfile(lab) else None})
    return subs


def save_nii(arr, affine, path):
    nib.save(nib.Nifti1Image(arr, affine), path)


def apply(m, x):
    return x @ m[:3, :3].T + m[:3, 3]


# ------------------------------------------------------------------ ANTs
class Ants:
    def __init__(self, path, threads=1):
        self.path = os.path.abspath(path)
        for b in ("antsRegistration", "antsApplyTransforms", "antsMultivariateTemplateConstruction2.sh",
                  "ANTSpexec.sh"):
            if not os.access(os.path.join(self.path, b), os.X_OK):
                raise SystemExit("ANTs: %s not found / not executable in %s" % (b, self.path))
        self.env = dict(os.environ)
        self.env["ANTSPATH"] = self.path + "/"
        self.env["PATH"] = self.path + os.pathsep + self.env.get("PATH", "")
        self.env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(int(threads))

    def run(self, args, cwd=None, logfile=None):
        cmd = [os.path.join(self.path, args[0])] + [str(a) for a in args[1:]]
        if logfile:
            with open(logfile, "a") as f:
                f.write("\n$ %s\n" % " ".join(cmd))
                f.flush()
                r = subprocess.run(cmd, cwd=cwd, env=self.env, stdout=f, stderr=subprocess.STDOUT)
        else:
            r = subprocess.run(cmd, cwd=cwd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            tail = "" if logfile else r.stdout.decode(errors="replace")[-3000:]
            raise RuntimeError("ANTs command failed (%d): %s\n%s" % (r.returncode, " ".join(cmd), tail))

    def field(self, ref, transforms, out):
        """Composite displacement field (physical LPS mm) of ``transforms`` on the grid of ``ref``."""
        args = ["antsApplyTransforms", "-d", 3, "-r", ref, "-o", "[%s,1]" % out, "--float", 0]
        for t in transforms:
            args += ["-t", t]
        self.run(args)


def itk_field_to_grid_mm(path, shape, affine):
    """ITK displacement field (LPS physical mm, [X,Y,Z,1,3]) -> [X,Y,Z,3] mm along the grid axes."""
    d = F.to_xyz3(F.load_array(path), shape, path).astype(np.float64)
    d_ras = d * np.array([-1.0, -1.0, 1.0])
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(0))
    R = affine[:3, :3] / spacing  # columns: grid-axis directions in RAS
    return (d_ras @ R).astype(np.float32)


def read_itk_affine(path):
    """ITK .mat (AffineTransform_{float,double}_3_3 + fixed) -> 4x4 point map in physical LPS
    (fixed/template point -> moving/subject point): y = A (x - c) + c + t."""
    m = sio.loadmat(path)
    key = [k for k in m if k.startswith("AffineTransform")][0]
    p = np.asarray(m[key], np.float64).ravel()
    c = np.asarray(m["fixed"], np.float64).ravel()
    A, t = p[:9].reshape(3, 3), p[9:12]
    out = np.eye(4)
    out[:3, :3] = A
    out[:3, 3] = t + c - A @ c
    return out


def write_itk_affine(path, M):
    """4x4 LPS point map -> ITK .mat (fixed centre 0)."""
    sio.savemat(path, {"AffineTransform_double_3_3": np.concatenate([M[:3, :3].ravel(), M[:3, 3]])[:, None],
                       "fixed": np.zeros((3, 1))}, format="4")


def template_construction(ants, subs, work, ttype, iterations, cpus, extra, log_path):
    """Run antsMultivariateTemplateConstruction2.sh in ``work`` (inputs symlinked there)."""
    os.makedirs(work, exist_ok=True)
    names = []
    for s in subs:
        dst = os.path.join(work, s["dir"] + ".nii.gz")
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(s["image"]), dst)
        names.append(s["dir"] + ".nii.gz")
    par = ["-c", 2, "-j", cpus] if cpus > 1 else ["-c", 0]
    args = (["antsMultivariateTemplateConstruction2.sh", "-d", 3, "-o", PREFIX, "-i", iterations,
             "-r", 1, "-n", 0] + par + PRESETS[ttype] + list(extra) + names)
    log("ANTs template construction (%s, %d subjects, log %s)" % (ttype, len(subs), log_path))
    ants.run(args, cwd=work, logfile=log_path)


def transforms_of(work, ttype, j, name):
    base = os.path.join(work, "%s%s%d" % (PREFIX, name, j))
    aff = base + "0GenericAffine.mat"
    if ttype == "deformable":
        warp, iwarp = base + "1Warp.nii.gz", base + "1InverseWarp.nii.gz"
        return [warp, aff], ["[%s,1]" % aff, iwarp], aff, base
    return [aff], ["[%s,1]" % aff], aff, base


def convert(ants, subs, work, d, ttype, log_every=10):
    """Dense fields, moved images/labels, atlas, matrices, checks."""
    img0 = nib.load(subs[0]["image"])
    shape, affine = tuple(int(v) for v in img0.shape[:3]), img0.affine
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(0))
    tmp = os.path.join(work, "fields_tmp")
    os.makedirs(tmp, exist_ok=True)
    atlas_sum = np.zeros(shape, np.float64)
    labs, mats, checks = [], [], {"subjects": []}
    g = np.indices(shape, dtype=np.float64).reshape(3, -1).T
    V, Vi = affine, np.linalg.inv(affine)
    t0 = time.time()
    for j, s in enumerate(subs):
        im = nib.load(s["image"])
        if tuple(im.shape[:3]) != shape or not np.allclose(im.affine, affine, atol=1e-4):
            raise ValueError("%s: grid differs from the first subject" % s["image"])
        fwd_t, inv_t, aff_path, base = transforms_of(work, ttype, j, s["dir"])
        for p in fwd_t + [aff_path]:
            p = p.strip("[]").split(",")[0]
            if not os.path.isfile(p):
                raise FileNotFoundError("missing ANTs output %s" % p)
        ants.field(s["image"], fwd_t, os.path.join(tmp, "fwd.nii.gz"))
        ants.field(s["image"], inv_t, os.path.join(tmp, "inv.nii.gz"))
        fwd_mm = itk_field_to_grid_mm(os.path.join(tmp, "fwd.nii.gz"), shape, affine)
        inv_mm = itk_field_to_grid_mm(os.path.join(tmp, "inv.nii.gz"), shape, affine)
        fwd, inv = fwd_mm / spacing.astype(np.float32), inv_mm / spacing.astype(np.float32)
        img = np.asarray(im.dataobj, np.float32)
        moved = F.warp(img, fwd, order=1)
        atlas_sum += moved
        save_nii(fwd_mm, affine, os.path.join(d, s["dir"] + "_fwd.nii.gz"))
        save_nii(inv_mm, affine, os.path.join(d, s["dir"] + "_inv.nii.gz"))
        save_nii(moved, affine, os.path.join(d, s["dir"] + "_moved.nii.gz"))
        if s["labels"]:
            lab = np.rint(np.asarray(nib.load(s["labels"]).dataobj, np.float32)).astype(np.int32)
            ml = F.warp(lab, fwd, order=0)
            labs.append(ml)
            save_nii(ml, affine, os.path.join(d, s["dir"] + "_moved_seg.nii.gz"))
        row = {"subject": s["dir"], "fwd_mean_mm": float(F.norm_mm(fwd, spacing).mean())}
        if ttype != "deformable":
            M = Vi @ LPS @ read_itk_affine(aff_path) @ LPS @ V  # voxel: atlas -> subject
            mats.append(M)
            row["matrix_vs_field_max_vox"] = float(np.abs(apply(M, g) - g - fwd.reshape(-1, 3)).max())
        warped = os.path.join(work, "%stemplate0%s%dWarpedToTemplate.nii.gz" % (PREFIX, s["dir"], j))
        if os.path.isfile(warped):
            w = np.asarray(nib.load(warped).dataobj, np.float32)
            m = (w != 0) | (moved != 0)
            row["ncc_vs_ants_warped"] = float(np.corrcoef(w[m], moved[m])[0, 1])
        if j < 3:
            ident = F.norm_mm(F.compose(fwd, inv), spacing)
            row["inv_consistency_mean_mm"] = float(ident.mean())
        checks["subjects"].append(row)
        if j % log_every == 0 or j == len(subs) - 1:
            log("convert %d/%d %s |fwd| %.2f mm, ncc vs ANTs %s (%.0f s)" % (
                j + 1, len(subs), s["dir"], row["fwd_mean_mm"], row.get("ncc_vs_ants_warped"), time.time() - t0))
    shutil.rmtree(tmp, ignore_errors=True)
    save_nii((atlas_sum / len(subs)).astype(np.float32), affine, os.path.join(d, "atlas.nii.gz"))
    if len(labs) == len(subs):
        save_nii(MajorityVote(np.stack(labs)).full().astype(np.int32), affine, os.path.join(d, "atlas_seg.nii.gz"))
    if mats:
        fv = np.stack(mats)
        iv = np.linalg.inv(fv)
        np.savez(os.path.join(d, "matrices.npz"), subjects=np.array([s["dir"] for s in subs]), fwd_vox=fv,
                 inv_vox=iv, fwd_world=V @ fv @ Vi, inv_world=V @ iv @ Vi,
                 note="fwd: atlas -> subject point map (x_subject = M x_atlas), inv = fwd^-1; "
                      "vox = processed-grid voxel coords, world = scanner (RAS) mm of the grid affine")
    tpl = os.path.join(work, PREFIX + "template0.nii.gz")
    if os.path.isfile(tpl):
        shutil.copy(tpl, os.path.join(d, "ants_template0.nii.gz"))
    nccs = [r["ncc_vs_ants_warped"] for r in checks["subjects"] if "ncc_vs_ants_warped" in r]
    checks["min_ncc_vs_ants_warped"] = min(nccs) if nccs else None
    if mats:
        checks["max_matrix_vs_field_vox"] = max(r["matrix_vs_field_max_vox"] for r in checks["subjects"])
    ok = (not nccs or min(nccs) > 0.99) and (not mats or checks["max_matrix_vs_field_vox"] < 0.05)
    checks["pass"] = bool(ok)
    with open(os.path.join(d, "ants_check.json"), "w") as f:
        json.dump(checks, f, indent=2)
    log("checks: min NCC(our moved, ANTs WarpedToTemplate) %s, matrix vs field %s -> %s" % (
        checks["min_ncc_vs_ants_warped"], checks.get("max_matrix_vs_field_vox"), "PASS" if ok else "FAIL"))
    return checks


def run_type(ants, subs, out, ttype, iterations, cpus, extra, skip_template=False, info=None):
    d = os.path.join(out, ttype)
    work = os.path.join(d, "work")
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    if not skip_template:
        template_construction(ants, subs, work, ttype, iterations, cpus, extra,
                              os.path.join(work, "ants_template_construction.log"))
    t_tpl = time.time() - t0
    log("template construction done (%.1f h)" % (t_tpl / 3600))
    checks = convert(ants, subs, work, d, ttype)
    meta = dict(info or {})
    meta.update(type=ttype, n_subjects=len(subs), iterations=iterations, cpus=cpus,
                ants_args=PRESETS[ttype] + list(extra), template_construction_s=t_tpl,
                total_s=time.time() - t0, checks_pass=checks["pass"],
                convention="--components xyz --signs +++ --units mm")
    with open(os.path.join(d, "run_info.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return checks


# ------------------------------------------------------------------ self-test
def self_test(ants, keep=False):
    """ANTs conversions on synthetic data: a known affine through antsApplyTransforms, then small
    real affine and SyN template constructions (4 subjects, 40x48x36 voxels)."""
    rng = np.random.RandomState(0)
    ok = True

    def check(name, cond, val):
        nonlocal ok
        ok &= bool(cond)
        log("%-60s %s  (%s)" % (name, "PASS" if cond else "FAIL", val))

    tmp = tempfile.mkdtemp(prefix="ants_selftest_")
    shape = (40, 48, 36)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = (-20, -30, -10)
    xx = np.indices(shape).astype(np.float32)
    blob = np.zeros(shape, np.float32)
    for _ in range(30):
        ctr = rng.uniform(6, np.array(shape) - 6)
        blob += np.exp(-((xx - ctr[:, None, None, None]) ** 2).sum(0) / (2 * rng.uniform(2, 5) ** 2))
    blob *= 1000.0 / blob.max()
    ref = os.path.join(tmp, "ref.nii.gz")
    save_nii(blob, affine, ref)

    # 1. known LPS affine -> antsApplyTransforms field -> our convention
    M = np.eye(4)
    M[:3, :3] += rng.uniform(-0.08, 0.08, (3, 3))
    M[:3, 3] = rng.uniform(-4, 4, 3)
    mat = os.path.join(tmp, "known.mat")
    write_itk_affine(mat, M)
    check("ITK .mat write/read round trip", np.abs(read_itk_affine(mat) - M).max() < 1e-9, "")
    ants.field(ref, [mat], os.path.join(tmp, "known_field.nii.gz"))
    got = itk_field_to_grid_mm(os.path.join(tmp, "known_field.nii.gz"), shape, affine)
    Mv = np.linalg.inv(affine) @ LPS @ M @ LPS @ affine
    g = np.indices(shape, dtype=np.float64).reshape(3, -1).T
    exp = (apply(Mv, g) - g).reshape(shape + (3,))
    check("antsApplyTransforms field == analytic (grid mm)", np.abs(got - exp).max() < 1e-3,
          "%.1e" % np.abs(got - exp).max())

    # 2. synthetic processed dataset: subject = base pulled through a known affine about the centre
    proc = os.path.join(tmp, "processed")
    lab0 = np.digitize(blob, np.quantile(blob, [0.5, 0.7, 0.85, 0.95])).astype(np.int32)
    ctr = (np.array(shape) - 1) / 2.0
    subs = []
    for i in range(4):
        A = np.eye(4)
        A[:3, :3] += rng.uniform(-0.06, 0.06, (3, 3))
        A[:3, 3] = rng.uniform(-3, 3, 3) + ctr - A[:3, :3] @ ctr
        disp = np.moveaxis(np.einsum("ij,j...->i...", A[:3, :3], xx) + A[:3, 3, None, None, None] - xx, 0, -1)
        dd = os.path.join(proc, "S%d_T1w" % i)
        os.makedirs(dd)
        save_nii(F.warp(blob, disp, 1), affine, os.path.join(dd, "image.nii.gz"))
        save_nii(F.warp(lab0, disp, 0), affine, os.path.join(dd, "labels_dkt31aseg.nii.gz"))
    subs = find_subjects(proc, "dkt31aseg")
    out = os.path.join(tmp, "out")
    small = {"affine": ["-q", "100x50x20", "-f", "4x2x1", "-s", "2x1x0vox"],
             "deformable": ["-q", "20x10x5", "-f", "4x2x1", "-s", "2x1x0vox"]}
    for ttype in ("affine", "deformable"):
        t = time.time()
        c = run_type(ants, subs, out, ttype, 2, 2, small[ttype])
        d = os.path.join(out, ttype)
        check("%s: ANTs ran + %d subjects converted (%.0f s)" % (ttype, len(subs), time.time() - t),
              len(c["subjects"]) == 4, "")
        check("%s: our moved == ANTs WarpedToTemplate (min NCC)" % ttype,
              (c["min_ncc_vs_ants_warped"] or 0) > 0.995, c["min_ncc_vs_ants_warped"])
        if ttype == "affine":
            check("affine: .mat matrix == dense field", c["max_matrix_vs_field_vox"] < 0.01,
                  "%.1e" % c["max_matrix_vs_field_vox"])
        ic = max(r["inv_consistency_mean_mm"] for r in c["subjects"] if "inv_consistency_mean_mm" in r)
        check("%s: mean |compose(fwd, inv)| [mm]" % ttype, ic < (0.01 if ttype == "affine" else 0.2), "%.3f" % ic)
        mv = [np.asarray(nib.load(os.path.join(d, "S%d_T1w_moved.nii.gz" % i)).dataobj) for i in (0, 2)]
        m = (mv[0] > 0.2 * mv[0].max()) & (mv[1] > 0)
        ncc = float(np.corrcoef(mv[0][m], mv[1][m])[0, 1])
        check("%s: moved S0 vs moved S2 NCC (registration works)" % ttype, ncc > 0.97, round(ncc, 4))
        r = subprocess.run([sys.executable, "-m", "evaluation", "check-convention", "--processed-dir", proc,
                            "--n-subjects", "4", "--fwd", d + "/{subject_dir}_fwd.nii.gz",
                            "--inv", d + "/{subject_dir}_inv.nii.gz", "--moved", d + "/{subject_dir}_moved.nii.gz"],
                           cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        txt = r.stdout.decode(errors="replace")
        # top-ranked convention; with small synthetic transforms the negated/role-swapped variant can
        # come close (RESULT: AMBIGUOUS), so test the ranking itself
        lines = txt.splitlines()
        top = [ln.strip() for i, ln in enumerate(lines) if i > 0 and lines[i - 1].startswith("top conventions")]
        check("%s: evaluation check-convention ranks xyz +++ mm first" % ttype,
              bool(top) and "components=xyz signs=+++ units=mm swap_roles=False" in top[0],
              top[0] if top else txt[-500:])
    log("self-test files: %s" % tmp)
    if not keep and ok:
        shutil.rmtree(tmp, ignore_errors=True)
    log("SELF-TEST %s" % ("PASSED" if ok else "FAILED"))
    return ok


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true", help="synthetic checks (needs ANTs, ~5-15 min)")
    ap.add_argument("--keep-self-test", action="store_true")
    ap.add_argument("--type", choices=("rigid", "affine", "deformable"))
    ap.add_argument("--processed-dir", help="<subject>_T1w/image.nii.gz folders on one common grid")
    ap.add_argument("--label-name", default="dkt31aseg", help="labels_<name>.nii.gz (optional)")
    ap.add_argument("--subjects-file", default=None)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--out", help="output folder; results in <out>/<type>/")
    ap.add_argument("--iterations", type=int, default=4, help="template iterations (-i)")
    ap.add_argument("--cpus", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                    help="parallel registrations (-c 2 -j), 1 = serial")
    ap.add_argument("--ants-path", default=os.environ.get("ANTSPATH", DEFAULT_ANTS))
    ap.add_argument("--ants-args", default="", help="extra template-construction flags, appended last")
    ap.add_argument("--skip-template", action="store_true",
                    help="reuse the transforms in <out>/<type>/work (only redo the conversion)")
    a = ap.parse_args()

    ants = Ants(a.ants_path, threads=1)
    if a.self_test:
        sys.exit(0 if self_test(ants, a.keep_self_test) else 1)
    for k in ("type", "processed_dir", "out"):
        if not getattr(a, k):
            ap.error("--%s is required" % k.replace("_", "-"))
    subs = find_subjects(a.processed_dir, a.label_name, a.subjects_file)
    if a.max_subjects:
        subs = subs[:a.max_subjects]
    if len(subs) < 2:
        raise SystemExit("need at least 2 subjects")
    if any(not s["labels"] for s in subs):
        log("WARNING: labels_%s missing for some subjects -> no moved_seg / atlas_seg" % a.label_name)
    extra = a.ants_args.split()
    info = {"processed_dir": os.path.abspath(a.processed_dir), "ants_path": ants.path}
    c = run_type(ants, subs, a.out, a.type, a.iterations, max(1, a.cpus), extra, a.skip_template, info)
    log("done: %s/%s (checks %s)" % (a.out, a.type, "PASS" if c["pass"] else "FAIL"))
    sys.exit(0 if c["pass"] else 4)


if __name__ == "__main__":
    main()
