"""Behaviour of the fake external tools used in tests (see fake_tools.py).

Each fake logs {"tool", "argv"} to $FAKE_TOOL_LOG and writes plausible outputs
with nibabel/numpy/scipy. FAKE_FAIL=<tool> makes that tool exit 3.
FLIRT semantics are re-implemented here independently of preprocessing.nifti.
"""
import json
import os
import sys
import tempfile

import nibabel as nib
import numpy as np
from scipy import ndimage


def _log(tool, argv):
    # Like real tools, create a temp file the default way (honours $TMPDIR, falls
    # back to /tmp when it is unusable) and record where it landed.
    fd, tmp = tempfile.mkstemp(prefix="fake_%s_" % tool)
    os.close(fd)
    os.remove(tmp)
    path = os.environ.get("FAKE_TOOL_LOG")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"tool": tool, "argv": argv, "tmp_dir": os.path.dirname(tmp),
                                "TMPDIR": os.environ.get("TMPDIR")}) + "\n")


def _opt(argv, flag, default=None):
    if flag in argv:
        return argv[argv.index(flag) + 1]
    return default


def _load(path):
    img = nib.load(path)
    return np.asanyarray(img.dataobj).astype(np.float64), img


def _save(data, affine, path, dtype=np.float32):
    img = nib.Nifti1Image(np.asarray(data).astype(dtype), affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, path)


def _largest(m):
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    c = np.bincount(lab.ravel())
    c[0] = 0
    return lab == np.argmax(c)


def _resample(src, src_aff, shape, ref_aff, order, vox2vox=None):
    if vox2vox is None:
        vox2vox = np.linalg.inv(src_aff) @ ref_aff
    ijk = np.indices(shape).reshape(3, -1)
    pts = np.vstack([ijk, np.ones((1, ijk.shape[1]))])
    c = (vox2vox @ pts)[:3]
    if order == 0:
        c = np.rint(c)
    out = ndimage.map_coordinates(src, c, order=order, mode="constant", cval=0.0)
    return out.reshape(shape)


def _fsl(shape, img):
    zooms = img.header.get_zooms()[:3]
    s = np.diag(list(zooms) + [1.0])
    if np.linalg.det(img.affine[:3, :3]) > 0:
        f = np.eye(4)
        f[0, 0] = -1
        f[0, 3] = shape[0] - 1
        s = s @ f
    return s


def synthstrip(argv):
    data, img = _load(_opt(argv, "-i"))
    m = ndimage.binary_fill_holes(_largest(data >= 0.6 * data.max()))
    _save(data * m, img.affine, _opt(argv, "-o"))
    _save(m, img.affine, _opt(argv, "-m"), np.uint8)


def n4(argv):
    out = _opt(argv, "-o")
    if out.startswith("["):
        out = out[1:-1].split(",")[0]
    data, img = _load(_opt(argv, "-i"))
    _save(data, img.affine, out)


def ants_registration(argv):
    if "--version" in argv:
        print("ANTs Version: 2.5.0.post7-g4f7f3c4\nCompiled: Jan 1 2024")
        return
    prefix = _opt(argv, "--output")
    if prefix.startswith("["):
        prefix = prefix[1:-1].split(",")[0]
    with open(prefix + "0GenericAffine.mat", "w") as f:
        f.write("FAKE_IDENTITY\n")


def ants_apply(argv):
    src, simg = _load(_opt(argv, "-i"))
    ref = nib.load(_opt(argv, "-r"))
    order = 1 if _opt(argv, "-n") == "Linear" else 0
    out = _resample(src, simg.affine, ref.shape[:3], ref.affine, order)
    _save(out, ref.affine, _opt(argv, "-o"))


def flirt(argv):
    src, simg = _load(_opt(argv, "-in"))
    ref = nib.load(_opt(argv, "-ref"))
    s_in, s_ref = _fsl(src.shape, simg), _fsl(ref.shape[:3], ref)
    if "-applyxfm" not in argv:
        shift = np.eye(4)
        if os.environ.get("FAKE_FLIRT_SHIFT"):
            shift[:3, 3] = [float(x) for x in os.environ["FAKE_FLIRT_SHIFT"].split(",")]
        # world(in) -> world(ref) = shift; express in FSL coordinates.
        m = s_ref @ np.linalg.inv(ref.affine) @ shift @ simg.affine @ np.linalg.inv(s_in)
        np.savetxt(_opt(argv, "-omat"), m)
        return
    m = np.loadtxt(_opt(argv, "-init"))
    # output voxel -> ref FSL -> input FSL (inverse of M) -> input voxel
    vox2vox = np.linalg.inv(s_in) @ np.linalg.inv(m) @ s_ref
    order = 0 if _opt(argv, "-interp") == "nearestneighbour" else 1
    out = _resample(src, None, ref.shape[:3], None, order, vox2vox=vox2vox)
    dtype = np.float32 if _opt(argv, "-datatype") == "float" else simg.get_data_dtype()
    _save(out, ref.affine, _opt(argv, "-out"), dtype)


def fslreorient2std(argv):
    img = nib.as_closest_canonical(nib.load(argv[0]))
    _save(np.asanyarray(img.dataobj), img.affine, argv[1])


def synthseg(argv):
    inp, out = _opt(argv, "--i"), _opt(argv, "--o")
    vol, qc = _opt(argv, "--vol"), _opt(argv, "--qc")

    def lines(p):
        with open(p) as f:
            return [x.strip() for x in f if x.strip()]

    if inp.endswith(".txt"):
        ins, outs = lines(inp), lines(out)
        vols = lines(vol) if vol else [None] * len(ins)
        qcs = lines(qc) if qc else [None] * len(ins)
    else:
        ins, outs, vols, qcs = [inp], [out], [vol], [qc]
    crop = os.environ.get("FAKE_SYNTHSEG_CROP") == "1"
    for i, o, v, q in zip(ins, outs, vols, qcs):
        data, img = _load(i)
        lab = np.array([0, 24, 3, 2])[np.digitize(data, [0.05, 0.4, 0.7])]
        aff = img.affine.copy()
        if crop:
            lab = lab[2:-2, 2:-2, 2:-2]
            aff[:3, 3] = img.affine[:3, :3] @ np.array([2.0, 2.0, 2.0]) + img.affine[:3, 3]
        _save(lab, aff, o, np.int32)
        for p, text in ((v, "subject,total\nx,1\n"), (q, "subject,general\nx,0.9\n")):
            if p:
                with open(p, "w") as f:
                    f.write(text)


TOOLS = {
    "mri_synthstrip": synthstrip,
    "N4BiasFieldCorrection": n4,
    "antsRegistration": ants_registration,
    "antsApplyTransforms": ants_apply,
    "flirt": flirt,
    "fslreorient2std": fslreorient2std,
    "mri_synthseg": synthseg,
}


def main(tool):
    argv = sys.argv[1:]
    _log(tool, argv)
    if os.environ.get("FAKE_FAIL") == tool:
        sys.stderr.write("fake failure\n")
        sys.exit(3)
    TOOLS[tool](argv)
