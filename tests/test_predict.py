"""Tests of predict.py: canonical fields, writer, atlas, internal check, CLI (predict-script FEAT-004;
design 3.3, 5, 7, 8, 10.1 (b)-(d), AC 2-5).

* TestCanonicalConversion: `canonical.to_canonical` vs independent numpy formulas (known rotation /
  affine, constant u, their composition; inverse consistency of the linear fields) at 1 mm and on
  a synthetic 2 mm grid (AC 5).
* TestCLI / TestGrid / TestGitCommit / TestSetupDevice: CLI validation (exit 2, no TF import),
  check-streaming chunk defaults, --grid-reference, git commit fallbacks, device setup call order.
* TestRunLevel4: predict.json schema and grid record, non-finite (exit 3), internal mismatch
  (exit 5), mirrored grid reference (exit 2), `predict.py run --device cpu` in a subprocess,
  check-streaming --n-subjects 5 on a default-width level-1 config (exit 2, config only).
* TestLevel2Convention: (c) the 5.3 checks from the written files for every branch on a TINY
  level-2 cascade with large randomised heads, a wrong conversion fails them, check-convention
  picks the canonical flags and the evaluator runs; (d) the documented composition per branch;
  check-streaming end to end (N 3, default chunks).
* TestFilesVsTrainerLevel1: (b) the written level-1 _fwd / _inv files vs the trainer's u / u_inv
  (unchunked S_exact, streamed chunk 1 / 3 S_stream), on /CPU:0 inside the GPU process.
Measured maxima are printed to stderr.
"""
import argparse
import contextlib
import io
import json
import logging
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory limit)
from tests.predict_helpers import (GRID_SHAPE, grid_affine, save_cascade_run, save_chain_run,
                                   save_level4_run, write_subjects)
from tests.runtime_helpers import free_memory, tiny_config
from tests.test_cascade import N_CLASSES, STAGE_KW, chain_config, randomize, rotation_deg
from tests.test_predict_runner import _level1
from tests.test_trainer import _OpDeterminism
import nibabel as nib
import numpy as np
import tensorflow as tf
from scipy import ndimage

from evaluation import evaluate as EV
from evaluation import fields as F
from inference import canonical as CN
from inference import cli as CLI
from inference import predict as PR
from inference.loader import LoadedModel
from inference.runner import StreamedRunner, UnchunkedRunner, _pool_to
from inference.store import RamStore
from models import cascade as CS
from models import transforms as tfm
from models.config import ModelConfig
from models.group_net import GroupRegLevelModel
from runtime import checkpoint as C
from runtime import optimizers as O
from runtime.run import Trainer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = 1e-3                                    # AC 3 gates (coordinate vox, moved, inverse mm)
B_STREAM, B_EXACT, B_EXACT_FALLBACK = 1e-4, 1e-6, 1e-5
BRANCH_ORDER = ("rigid", "affine", "deform", "affine+deform")
# predict.json `pipeline` of a default level-2 run, as written before predict-levels (literal).
DEFAULT_PIPELINE_L2 = {
    "rigid": "rigid cascade (L4 -> L2) on the raw images; fwd / inv = linear field of A",
    "affine": "affine cascade (L4 -> L2) on the raw images; fwd / inv = linear field of A",
    "deform": "deform cascade (L4 -> L2) on the raw images (inputs MNI-affine-aligned); fwd = u, "
              "inv = u_inv",
    "affine+deform": "affine cascade (L4 -> L2); images resampled once into the affine central space "
                     "(I o A^-1, trilinear, zero fill); deform cascade (L4 -> L2) on them; fwd = "
                     "A^-1 o (id + u), inv its inverse"}


def _say(msg):
    print("\n[predict] " + msg, file=sys.stderr)


def _parse(*argv):
    return CLI.parse_args([str(a) for a in argv])


def _run(ckpt, proc, out, *extra):
    return _parse("run", "--checkpoint", ckpt, "--processed-dir", proc, "--out", out, *extra)


def _read(path):
    return np.asarray(nib.load(path).dataobj, np.float32)


def _files(out, sdir):
    return tuple(_read(os.path.join(out, "%s_%s.nii.gz" % (sdir, k))) for k in ("fwd", "inv", "moved"))


def _json(path):
    with open(path) as f:
        return json.load(f)


def _diff(a, b, mask=None):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    if mask is not None:
        d = d[mask]
    return float(d.max())


def _collect(loaded, images, branches, streamed=False, plan=None):
    """Internal results (numpy) of the unchunked runner, or of the streamed runner with the `run`
    default chunks (the script's own computation): A [N, 4, 4] or None, U1 / U1_inv or None.
    ``plan``: a predict-levels `LevelPlan` for the runner (None = default)."""
    n = images.shape[0]
    got = {}

    def sink(i0, i1, res):
        for k in ("A", "U1", "U1_inv"):
            if res[k] is not None:
                a = res[k].numpy()
                got.setdefault(k, np.zeros((n,) + a.shape[1:], a.dtype))[i0:i1] = a
    if streamed:
        with RamStore() as store:
            StreamedRunner(loaded, store, chunks=dict(CLI.RUN_CHUNK_DEFAULTS), plan=plan).run(
                images, branches, sink)
    else:
        UnchunkedRunner(loaded, plan=plan).run(images, branches, sink)
    return got.get("A"), got.get("U1"), got.get("U1_inv")


# ============================================================================ numpy references
def _np_v(shape, spacing):
    """Independent index -> centred mm matrix: x = (p - (n - 1) / 2) * s."""
    s = np.asarray(spacing, np.float64)
    v = np.eye(4)
    v[:3, :3] = np.diag(s)
    v[:3, 3] = -s * (np.asarray(shape, np.float64) - 1.0) / 2.0
    return v


def _np_fields(A, shape, spacing, u=None, u_inv=None):
    """fwd / inv voxel displacements of design 5.1 in float64 numpy (u, u_inv constant 3-vectors)."""
    v = _np_v(shape, spacing)
    vi = np.linalg.inv(v)
    m_inv, m = vi @ np.linalg.inv(A) @ v, vi @ A @ v
    p = np.moveaxis(np.indices(shape, dtype=np.float64), 0, -1)
    q = p if u is None else p + np.asarray(u, np.float64)
    fwd = q @ m_inv[:3, :3].T + m_inv[:3, 3] - p
    z = p @ m[:3, :3].T + m[:3, 3]
    inv = z - p if u_inv is None else z + np.asarray(u_inv, np.float64) - p
    return fwd, inv


def _rot(deg, axis):
    a = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def _affine(lin, t):
    a = np.eye(4)
    a[:3, :3], a[:3, 3] = lin, t
    return a


def _inside(fwd, shape, margin=1):
    """Atlas voxels whose target p + fwd(p) lies inside the grid with ``margin`` (numpy)."""
    q = np.moveaxis(np.indices(shape, dtype=np.float64), 0, -1) + fwd
    ok = np.ones(shape, bool)
    for a in range(3):
        ok &= (q[..., a] >= margin) & (q[..., a] <= shape[a] - 1 - margin)
    return ok


class TestCanonicalConversion(unittest.TestCase):
    """AC 5: the converter against independent formulas, at 1 mm and on a 2 mm grid."""

    SHAPE = (20, 24, 18)
    ROT = _rot(10.0, (0.3, 0.5, 0.8))
    T = np.array([3.0, -2.0, 1.5])
    LIN_AFF = ROT @ np.array([[1.1, 0.05, -0.03], [0.0, 0.92, 0.04], [0.0, 0.0, 1.05]])
    U, U_INV = np.array([1.5, -0.5, 0.25]), np.array([-1.5, 0.5, -0.25])

    def _const(self, vec):
        return tf.constant(np.broadcast_to(np.asarray(vec, np.float32), (1,) + self.SHAPE + (3,)).copy())

    def _check(self, branches, A, spacing, u=None, u_inv=None):
        shape = self.SHAPE
        ref_f, ref_i = _np_fields(A, shape, spacing, u, u_inv)
        U1 = None if u is None else self._const(u)
        U1i = None if u_inv is None else self._const(u_inv)
        A_t = None if branches == "deform" else tf.constant(A[None], tf.float32)
        fwd, inv = CN.to_canonical(branches, A_t, U1, U1i, shape, spacing)
        fwd, inv = fwd.numpy()[0], inv.numpy()[0]
        e_f, e_i = _diff(fwd, ref_f), _diff(inv, ref_i)
        mm = CN.to_mm(fwd, spacing)
        np.testing.assert_allclose(mm, fwd * np.asarray(spacing, np.float32), rtol=0, atol=0)
        inside = _inside(fwd, shape)
        comp = F.norm_mm(F.compose(fwd, inv), spacing)[inside]
        _say("canonical %s spacing %s: fwd %.2e, inv %.2e vox vs numpy; |compose(fwd, inv)| max %.2e mm "
             "(%d interior voxels)" % (branches, spacing, e_f, e_i, float(comp.max()), int(inside.sum())))
        self.assertLessEqual(e_f, 1e-4)
        self.assertLessEqual(e_i, 1e-4)
        self.assertGreater(inside.sum(), 0.2 * inside.size)
        self.assertLess(float(comp.max()), 1e-4)
        return fwd, inv

    def test_rigid_and_affine_1mm_and_2mm(self):
        for spacing in ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0)):
            fwd, _ = self._check("rigid", _affine(self.ROT, self.T), spacing)
            self.assertGreater(float(np.abs(fwd).max()), 1.0)
            self._check("affine", _affine(self.LIN_AFF, self.T), spacing)

    def test_deform_constant_field(self):
        fwd, inv = CN.to_canonical("deform", None, self._const(self.U), self._const(self.U_INV),
                                   self.SHAPE, (2.0, 2.0, 2.0))
        np.testing.assert_array_equal(fwd.numpy()[0], np.broadcast_to(self.U.astype(np.float32), self.SHAPE + (3,)))
        np.testing.assert_array_equal(inv.numpy()[0], np.broadcast_to(self.U_INV.astype(np.float32), self.SHAPE + (3,)))
        np.testing.assert_array_equal(CN.to_mm(fwd.numpy()[0], (2.0, 2.0, 2.0))[0, 0, 0], 2.0 * self.U.astype(np.float32))

    def test_affine_deform_composition_1mm_and_2mm(self):
        for spacing in ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0)):
            self._check("affine+deform", _affine(self.LIN_AFF, self.T), spacing, self.U, self.U_INV)


# ============================================================================ CLI
class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_cli_")
        self.ck = os.path.join(self.tmp, "run", "ckpt", "last")
        os.makedirs(self.ck)
        self.proc = os.path.join(self.tmp, "proc")
        os.makedirs(self.proc)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _exit2(self, *argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            _parse(*argv)
        self.assertEqual(cm.exception.code, 2, err.getvalue())
        return err.getvalue()

    def _run_argv(self, *extra):
        return ("run", "--checkpoint", self.ck, "--processed-dir", self.proc,
                "--out", os.path.join(self.tmp, "out")) + extra

    def _check_argv(self, *extra):
        return ("check-streaming", "--checkpoint", self.ck, "--processed-dir", self.proc,
                "--out", os.path.join(self.tmp, "c.json")) + extra

    def test_rejected_branches_exit2_naming_the_four_values(self):
        for bad in ("all", "rigid+deform", "rigid+affine", "deform+affine"):
            for argv in (self._run_argv("--branches", bad), self._check_argv("--branches", bad)):
                msg = self._exit2(*argv)
                for v in CLI.BRANCH_VALUES:
                    self.assertIn(v, msg)
                self.assertIn("rigid and affine are alternatives; the deform branch was trained on "
                              "affine-aligned inputs", msg)
        for good in CLI.BRANCH_VALUES:
            self.assertEqual(_parse(*self._run_argv("--branches", good)).branches, good)
        self.assertEqual(_parse(*self._run_argv()).branches, "affine+deform")
        self.assertEqual(_parse(*self._check_argv()).branches, "deform")

    def test_defaults(self):
        a = _parse(*self._run_argv())
        self.assertEqual((a.chunk_l1, a.chunk_l2, a.chunk_l4), (2, 8, 32))
        self.assertEqual((a.mode, a.device, a.precision, a.store, a.check_internal, a.compress_level, a.tta),
                         ("streamed", "gpu", "checkpoint", "auto", 2, 1, "none"))
        c = _parse(*self._check_argv())
        self.assertEqual((c.chunk_l1, c.chunk_l2, c.chunk_l4, c.n_subjects), (None, None, None, 4))
        self.assertIsNone(c.tol_field_max_mm)
        self.assertEqual(_parse(*self._check_argv("--tol-moved", "0.5")).tol_moved, 0.5)

    def test_validation_errors_exit2(self):
        os.makedirs(os.path.join(self.tmp, "full"))
        open(os.path.join(self.tmp, "full", "predict.json"), "w").close()
        open(os.path.join(self.tmp, "full2_x"), "w").close()
        os.makedirs(os.path.join(self.tmp, "full3"))
        open(os.path.join(self.tmp, "full3", "S0_T1w_fwd.nii.gz"), "w").close()
        cases = [
            ("run", "--checkpoint", os.path.join(self.tmp, "nope"), "--processed-dir", self.proc, "--out", "o"),
            ("run", "--checkpoint", self.ck, "--processed-dir", os.path.join(self.tmp, "nope"), "--out", "o"),
            self._run_argv("--device", "cpu", "--gpu-memory-limit-mb", "1000"),
            self._check_argv("--device", "cpu", "--gpu-memory-limit-mb", "1000"),
            self._run_argv("--gpu-memory-limit-mb", "0"),
            self._run_argv("--tta", "flip"),
            self._run_argv("--chunk-l1", "-1"),
            self._run_argv("--chunk-l2", "x"),
            self._run_argv("--compress-level", "10"),
            self._run_argv("--check-internal", "-1"),
            self._run_argv("--grid-reference", os.path.join(self.tmp, "missing.json")),
            self._run_argv("--subjects-file", os.path.join(self.tmp, "missing.txt")),
            self._check_argv("--n-subjects", "1"),
            self._check_argv("--tol-moved", "0"),
            self._check_argv("--tol-field-max-mm", "-1e-3"),
            self._check_argv("--tol-matrix", "nan"),
            ("check-streaming", "--checkpoint", self.ck, "--processed-dir", self.proc, "--out", self.tmp),
            ("run", "--checkpoint", self.ck, "--processed-dir", self.proc, "--out", os.path.join(self.tmp, "full")),
            ("run", "--checkpoint", self.ck, "--processed-dir", self.proc, "--out", os.path.join(self.tmp, "full2_x")),
            ("run", "--checkpoint", self.ck, "--processed-dir", self.proc, "--out", os.path.join(self.tmp, "full3")),
        ]
        for argv in cases:
            with self.subTest(argv=argv[1:]):
                self._exit2(*argv)
        a = _parse("run", "--checkpoint", self.ck, "--processed-dir", self.proc, "--out",
                   os.path.join(self.tmp, "full"), "--overwrite")
        self.assertTrue(a.overwrite)
        prefix = os.path.join(self.ck, "last-3")
        open(prefix + ".index", "w").close()
        self.assertEqual(_parse("run", "--checkpoint", prefix, "--processed-dir", self.proc, "--out",
                                os.path.join(self.tmp, "o2")).checkpoint, prefix)

    def test_validate_imports_no_tensorflow(self):
        code = ("import sys\nfrom inference import cli\n"
                "a = cli.parse_args(%r)\n"
                "print('TF_IMPORTED' if 'tensorflow' in sys.modules else 'NO_TF', a.branches)\n"
                "try:\n    cli.parse_args(%r)\nexcept SystemExit as e:\n    print('EXIT', e.code)\n"
                "print('TF_IMPORTED' if 'tensorflow' in sys.modules else 'NO_TF')\n"
                % (list(self._run_argv()), list(self._run_argv("--branches", "all"))))
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = out.stdout.split()
        self.assertEqual(lines, ["NO_TF", "affine+deform", "EXIT", "2", "NO_TF"], out.stdout + out.stderr)

    def test_resolve_check_chunks(self):
        caps = {"l1": 4, "l2": 19, "l4": 89}
        r = CLI.resolve_check_chunks(4, {"l1": None, "l2": None, "l4": None}, caps)
        self.assertEqual((r["l1"], r["l2"], r["l4"]), (3, 3, 3))
        self.assertEqual(r["source"], {"l1": "default", "l2": "default", "l4": "default"})
        self.assertEqual(r["capped"], {})
        self.assertEqual([CLI.resolve_check_chunks(n, {}, caps)["l2"] for n in (2, 3, 4)], [1, 2, 3])
        with self.assertLogs("inference.cli", "WARNING") as logs:
            r = CLI.resolve_check_chunks(4, {"l1": 1, "l2": 0, "l4": None}, caps)
        self.assertTrue(any("not chunked" in m for m in logs.output))
        self.assertEqual((r["l1"], r["l2"], r["l4"]), (1, 0, 3))
        self.assertEqual(r["source"], {"l1": "flag", "l2": "flag", "l4": "default"})
        r = CLI.resolve_check_chunks(8, {"l1": 6}, caps)
        self.assertEqual(r["l1"], 4)
        self.assertEqual(r["capped"], {"l1": {"requested": 6, "used": 4}})
        r = CLI.resolve_check_chunks(10, {}, caps)
        self.assertEqual((r["l1"], r["l2"], r["l4"]), (4, 9, 9))
        self.assertEqual(r["capped"], {"l1": {"requested": 9, "used": 4}})
        r = CLI.resolve_check_chunks(30, {"l1": 0}, caps)
        self.assertEqual(r["l1"], 4)


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_grid_")
        self.aff = grid_affine()
        mirrored = self.aff.copy()
        mirrored[0, :] *= -1.0                       # x reversed: x = -(i - c)
        mirrored[0, 3] = (GRID_SHAPE[0] - 1.0) / 2.0
        self.mirrored = mirrored

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _json_ref(self, name, aff, shape=GRID_SHAPE, key="affine"):
        p = os.path.join(self.tmp, name)
        d = {key: np.asarray(aff).tolist()}
        if shape is not None:
            d["shape"] = list(shape)
        with open(p, "w") as f:
            json.dump(d, f)
        return p

    def _nii_ref(self, name, aff, shape=GRID_SHAPE):
        p = os.path.join(self.tmp, name)
        nib.save(nib.Nifti1Image(np.zeros(shape, np.uint8), aff), p)
        return p

    def test_json_and_nifti_references(self):
        for ref in (self._json_ref("t.json", self.aff), self._json_ref("t2.json", self.aff, shape=None),
                    self._nii_ref("t.nii.gz", self.aff)):
            rec = CN.check_grid(self.aff, GRID_SHAPE, (1.0, 1.0, 1.0), ref)
            self.assertTrue(rec["checked"])
            self.assertEqual(rec["reference_path"], os.path.abspath(ref))
            self.assertLessEqual(rec["reference_max_abs_diff"], 1e-6)
            self.assertEqual(rec["affine"], self.aff.tolist())
            self.assertEqual(rec["shape"], list(GRID_SHAPE))
        aff, shape = CN.load_reference_affine(self._json_ref("t3.json", self.aff))
        self.assertEqual((aff.dtype, aff.shape, shape), (np.float64, (4, 4), GRID_SHAPE))

    def test_mirrored_and_bad_references_raise(self):
        bad = [self._json_ref("m.json", self.mirrored), self._nii_ref("m.nii.gz", self.mirrored),
               self._json_ref("noaff.json", self.aff, key="matrix"),
               self._json_ref("shape.json", self.aff, shape=(160, 224, 191)),
               self._nii_ref("shape.nii.gz", self.aff, shape=(16, 22, 19))]
        broken = os.path.join(self.tmp, "broken.nii.gz")
        with open(broken, "wb") as f:
            f.write(b"not a nifti")
        bad.append(broken)
        for ref in bad:
            with self.subTest(ref=os.path.basename(ref)), self.assertRaises(CN.GridError):
                with contextlib.redirect_stderr(io.StringIO()):
                    CN.check_grid(self.aff, GRID_SHAPE, (1.0, 1.0, 1.0), ref)
        with self.assertRaisesRegex(CN.GridError, "no 4x4 affine"):
            CN.check_grid(self.aff, GRID_SHAPE, (1.0, 1.0, 1.0), os.path.join(self.tmp, "noaff.json"))

    def test_absent_reference_and_grid_requirements(self):
        with self.assertLogs("inference.canonical", "INFO") as logs:
            rec = CN.check_grid(self.aff, GRID_SHAPE, (1.0, 1.0, 1.0), None)
        self.assertTrue(any("no --grid-reference" in m for m in logs.output))
        self.assertEqual((rec["checked"], rec["reference_path"], rec["reference_max_abs_diff"]),
                         (False, None, None))
        with self.assertRaises(CN.GridError):
            CN.check_grid(self.aff, (80, 112, 96), (1.0, 1.0, 1.0))
        with self.assertRaises(CN.GridError):
            CN.check_grid(self.aff, GRID_SHAPE, (1.0, 1.0, 1.01))


class TestGitCommit(unittest.TestCase):
    SHA = "0123456789abcdef0123456789abcdef01234567"

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_git_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _repo(self, name, head, loose=None, packed=None):
        root = os.path.join(self.tmp, name)
        g = os.path.join(root, ".git")
        os.makedirs(os.path.join(g, "refs", "heads"))
        with open(os.path.join(g, "HEAD"), "w") as f:
            f.write(head + "\n")
        if loose is not None:
            with open(os.path.join(g, "refs", "heads", "main"), "w") as f:
                f.write(loose + "\n")
        if packed is not None:
            with open(os.path.join(g, "packed-refs"), "w") as f:
                f.write("# pack-refs with: peeled fully-peeled sorted\n%s refs/heads/main\n" % packed)
        return root

    def test_fallbacks_without_git(self):
        other = "f" * 40
        with mock.patch.object(PR.subprocess, "run", side_effect=FileNotFoundError("git")):
            self.assertEqual(PR.git_commit(self._repo("loose", "ref: refs/heads/main", loose=self.SHA)),
                             (self.SHA, "git_dir"))
            self.assertEqual(PR.git_commit(self._repo("packed", "ref: refs/heads/main", packed=self.SHA)),
                             (self.SHA, "git_dir"))
            self.assertEqual(PR.git_commit(self._repo("both", "ref: refs/heads/main", loose=self.SHA,
                                                      packed=other)), (self.SHA, "git_dir"))
            self.assertEqual(PR.git_commit(self._repo("missing", "ref: refs/heads/main")), (None, "none"))
            self.assertEqual(PR.git_commit(self._repo("detached", self.SHA)), (self.SHA, "git_dir"))
            self.assertEqual(PR.git_commit(os.path.join(self.tmp, "nogit")), (None, "none"))
        for exc in (subprocess.CalledProcessError(128, "git"), subprocess.TimeoutExpired("git", 10), OSError()):
            with mock.patch.object(PR.subprocess, "run", side_effect=exc):
                self.assertEqual(PR.git_commit(self._repo("r%d" % id(exc), self.SHA)), (self.SHA, "git_dir"))

    def test_real_repository(self):
        sha, source = PR.git_commit(REPO)
        self.assertIn(source, ("git", "git_dir"))
        self.assertRegex(sha, "^[0-9a-f]{40}$")


class TestSetupDevice(unittest.TestCase):
    """setup_device call order with the TF calls patched (design 3.3; no device change here)."""

    def _setup(self, device, limit=None, gpus=("GPU0",)):
        calls = []
        args = argparse.Namespace(device=device, gpu_memory_limit_mb=limit)
        with mock.patch.object(tf.config, "set_visible_devices",
                               side_effect=lambda *a, **k: calls.append(("hide",) + a)), \
                mock.patch("models.precision.configure_tf_precision",
                           side_effect=lambda **k: calls.append(("precision", k))), \
                mock.patch.object(tf.config, "list_physical_devices", return_value=list(gpus)), \
                mock.patch.object(tf.config, "set_logical_device_configuration",
                                  side_effect=lambda g, c: calls.append(("limit", g, c[0].memory_limit))), \
                mock.patch.object(tf.config, "get_visible_devices", return_value=[]):
            PR.setup_device(args)
        return calls

    def test_cpu_hides_gpus_first(self):
        self.assertEqual(self._setup("cpu"), [("hide", [], "GPU"),
                                              ("precision", {"tf32": False, "memory_growth": False})])

    def test_gpu_growth_or_limit(self):
        self.assertEqual(self._setup("gpu"), [("precision", {"tf32": False, "memory_growth": True})])
        self.assertEqual(self._setup("gpu", 1234), [("precision", {"tf32": False, "memory_growth": False}),
                                                    ("limit", "GPU0", 1234)])

    def test_errors(self):
        with self.assertRaises(PR.PredictError) as cm:
            self._setup("gpu", gpus=())
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("use --device cpu", str(cm.exception))
        with mock.patch.object(tf.config, "set_visible_devices", side_effect=RuntimeError("initialised")), \
                self.assertRaises(PR.PredictError) as cm:
            PR.setup_device(argparse.Namespace(device="cpu", gpu_memory_limit_mb=None))
        self.assertEqual(cm.exception.code, 1)


# ============================================================================ level 4 runs
class TestRunLevel4(unittest.TestCase):
    """predict.json, grid reference, exit codes 2 / 3 / 5, the CPU subprocess (TINY level 4, N 2)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgr_pred4_")
        cfg = tiny_config(**STAGE_KW)
        cls.ckpt, _ = save_level4_run(cls.tmp, "l4", cfg, seed=5)
        cls.ckpt_amp, _ = save_level4_run(cls.tmp, "l4amp", cfg, seed=5, amp=True)
        cls.proc = os.path.join(cls.tmp, "proc")
        cls.dirs, cls.images = write_subjects(cls.proc, 2, textured=True, seed=1)
        cls.sdirs = [os.path.basename(d) for d in cls.dirs]
        aff = grid_affine()
        cls.ref_json = os.path.join(cls.tmp, "target_grid.json")
        with open(cls.ref_json, "w") as f:
            json.dump({"shape": list(GRID_SHAPE), "affine": aff.tolist()}, f)
        cls.ref_nii = os.path.join(cls.tmp, "ref.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros(GRID_SHAPE, np.uint8), aff), cls.ref_nii)
        mirrored = aff.copy()
        mirrored[0, :] *= -1.0
        mirrored[0, 3] = (GRID_SHAPE[0] - 1.0) / 2.0
        cls.ref_mirrored = os.path.join(cls.tmp, "mirrored.json")
        with open(cls.ref_mirrored, "w") as f:
            json.dump({"shape": list(GRID_SHAPE), "affine": mirrored.tolist()}, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        free_memory()

    def _out(self, name):
        return os.path.join(self.tmp, name)

    def test_predict_json_schema_and_grid_reference(self):
        for name, extra in (("o_json", ("--branches", "rigid", "--grid-reference", self.ref_json)),
                            ("o_nii", ("--branches", "deform", "--mode", "unchunked",
                                       "--grid-reference", self.ref_nii))):
            out = self._out(name)
            self.assertEqual(PR.run_predict(_run(self.ckpt, self.proc, out, *extra), configure_device=False), 0)
            rec = _json(os.path.join(out, "predict.json"))
            self.assertEqual(set(rec), set(PR.PREDICT_KEYS))
            self.assertEqual(rec["status"], "ok")
            self.assertIs(rec["configure_device"], False)
            self.assertEqual(rec["subjects"], ["S0", "S1"])
            self.assertEqual(rec["subject_dirs"], self.sdirs)
            self.assertEqual(rec["n_subjects"], 2)
            g = rec["grid"]
            self.assertTrue(g["checked"])
            self.assertEqual(g["reference_path"], os.path.abspath(extra[-1]))
            self.assertLessEqual(g["reference_max_abs_diff"], 1e-6)
            self.assertEqual(g["shape"], list(GRID_SHAPE))
            np.testing.assert_allclose(np.asarray(g["affine"]), grid_affine(), atol=1e-6)
            self.assertEqual(rec["convention"], {"components": "xyz", "signs": "+++", "units": "mm",
                                                 "swap_roles": False})
            self.assertEqual(rec["global_transform"], "none (training augmentation only)")
            self.assertEqual(rec["nonfinite"]["total"], 0)
            self.assertTrue(rec["internal_check"]["passed"])
            self.assertEqual(rec["internal_check"]["k"], 2)
            self.assertEqual(rec["checkpoint"]["level"], 4)
            self.assertEqual(rec["checkpoint"]["step"], 7)
            self.assertEqual(rec["knobs"]["lean_sampler"], {"checkpoint": True, "used": True})
            self.assertEqual(rec["precision"]["trainable"], "float32")
            self.assertIn(rec["git_commit_source"], ("git", "git_dir", "none"))
            self.assertEqual(rec["tta"], "none")
            self.assertEqual(rec["compress_level"], 1)
            for k in ("total", "read", "load", "write", "atlas"):
                self.assertIn(k, rec["timing_s"])
            for f in rec["outputs"]:
                self.assertTrue(os.path.isfile(os.path.join(out, f)), f)
            self.assertEqual(len(rec["outputs"]), 3 * 2 + 3)
            fwd, inv, moved = _files(out, self.sdirs[0])
            self.assertEqual((fwd.shape, fwd.dtype, moved.shape), (GRID_SHAPE + (3,), np.float32, GRID_SHAPE))
            np.testing.assert_allclose(nib.load(os.path.join(out, "atlas.nii.gz")).affine, grid_affine(), atol=1e-6)
            seg = np.asarray(nib.load(os.path.join(out, "atlas_seg.nii.gz")).dataobj)
            self.assertEqual(seg.dtype, np.uint16)
            _say("level-4 %s: internal check %s" % (extra[1], {k: {q: round(v, 7) if isinstance(v, float) else v
                                                                  for q, v in r.items()}
                                                              for k, r in rec["internal_check"]["subjects"].items()}))

    def test_absent_and_mirrored_grid_reference(self):
        out = self._out("o_noref")
        with self.assertLogs("inference.canonical", "INFO") as logs:
            self.assertEqual(PR.run_predict(_run(self.ckpt, self.proc, out, "--branches", "affine"),
                                            configure_device=False), 0)
        self.assertTrue(any("no --grid-reference" in m for m in logs.output))
        g = _json(os.path.join(out, "predict.json"))["grid"]
        self.assertEqual((g["checked"], g["reference_path"]), (False, None))
        out = self._out("o_mirror")
        code = PR.run_predict(_run(self.ckpt, self.proc, out, "--branches", "affine",
                                   "--grid-reference", self.ref_mirrored), configure_device=False)
        self.assertEqual(code, 2)
        rec = _json(os.path.join(out, "predict.json"))
        self.assertEqual(rec["status"], "error")
        self.assertIn("grid reference", rec["message"])
        self.assertFalse(os.path.exists(os.path.join(out, self.sdirs[0] + "_fwd.nii.gz")))

    def test_nonfinite_exit3_no_atlas(self):
        orig = CN.to_canonical

        def nan_fwd(*a, **k):
            fwd, inv = orig(*a, **k)
            f = fwd.numpy()
            f[0, 50, 60, 70, 1] = np.nan
            return tf.constant(f), inv
        out = self._out("o_nan")
        with mock.patch.object(CN, "to_canonical", side_effect=nan_fwd):
            code = PR.run_predict(_run(self.ckpt, self.proc, out, "--branches", "rigid"), configure_device=False)
        self.assertEqual(code, 3)
        rec = _json(os.path.join(out, "predict.json"))
        self.assertEqual(rec["status"], "nonfinite")
        self.assertGreaterEqual(rec["nonfinite"]["total"], 2)
        self.assertEqual(rec["nonfinite"]["per_subject"][self.sdirs[0]]["fwd"], 1)
        self.assertFalse(os.path.exists(os.path.join(out, "atlas.nii.gz")))
        self.assertFalse(os.path.exists(os.path.join(out, "atlas_seg.nii.gz")))
        for s in self.sdirs:
            self.assertTrue(os.path.isfile(os.path.join(out, s + "_fwd.nii.gz")))

    def test_internal_mismatch_exit5(self):
        orig = CN.to_canonical

        def shifted(*a, **k):
            fwd, inv = orig(*a, **k)
            return fwd + 0.01, inv
        out = self._out("o_shift")
        with mock.patch.object(CN, "to_canonical", side_effect=shifted):
            code = PR.run_predict(_run(self.ckpt, self.proc, out, "--branches", "affine",
                                       "--check-internal", "1"), configure_device=False)
        self.assertEqual(code, 5)
        rec = _json(os.path.join(out, "predict.json"))
        self.assertEqual(rec["status"], "internal_mismatch")
        ic = rec["internal_check"]
        self.assertFalse(ic["passed"])
        self.assertEqual(ic["k"], 1)
        c = ic["subjects"][self.sdirs[0]]["coord_fwd_max_vox"]
        self.assertAlmostEqual(c, 0.01, delta=1e-3)
        self.assertTrue(os.path.isfile(os.path.join(out, "atlas.nii.gz")))
        _say("internal mismatch (+0.01 vox): coord_fwd_max_vox %.4g -> exit 5" % c)

    def test_check_streaming_n_max_exit2_from_config(self):
        run = os.path.join(self.tmp, "l1cfg")
        os.makedirs(os.path.join(run, "ckpt", "last"))
        prefix = os.path.join(run, "ckpt", "last", "last-1")
        open(prefix + ".index", "w").close()
        C.write_config(os.path.join(run, "config.json"), ModelConfig(deform_decoder=True),
                       {"level": 1, "split_group_conv": True})
        out = os.path.join(self.tmp, "nmax.json")
        code = PR.run_check_streaming(_parse("check-streaming", "--checkpoint", prefix, "--processed-dir",
                                             self.proc, "--out", out, "--n-subjects", "5"),
                                      configure_device=False)
        self.assertEqual(code, 2)
        rec = _json(out)
        self.assertEqual(rec["status"], "error")
        self.assertIn("unchunked mode at level 1 supports at most 4 subjects", rec["message"])
        self.assertIsNone(rec["checkpoint"])                       # failed before the model build
        _say("check-streaming --n-subjects 5 (default widths, level 1): %s" % rec["message"])

    def test_cpu_subprocess(self):
        out = self._out("o_cpu")
        env = dict(os.environ)
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        t0 = time.time()
        p = subprocess.run([sys.executable, "predict.py", "run", "--device", "cpu", "--checkpoint", self.ckpt_amp,
                            "--processed-dir", self.proc, "--out", out, "--branches", "rigid", "-v"],
                           cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
        if p.returncode < 0:
            self.skipTest("tf_gpu env crashes without a visible GPU (rc %d)" % p.returncode)
        self.assertEqual(p.returncode, 0, p.stderr[-4000:])
        dev = [ln for ln in p.stderr.splitlines() if "visible devices:" in ln]
        self.assertTrue(dev, p.stderr[-2000:])
        self.assertNotIn("GPU", dev[-1])
        self.assertIn("CPU", dev[-1])
        self.assertIn("WARNING", p.stderr)
        rec = _json(os.path.join(out, "predict.json"))
        self.assertEqual(rec["device"], "cpu")
        self.assertIs(rec["configure_device"], True)
        self.assertEqual(rec["precision"]["trainable"], "float32 (cpu)")
        self.assertTrue(any("AMP checkpoint" in w for w in rec["warnings"]), rec["warnings"])
        self.assertFalse(any("GPU" in d for d in rec["visible_devices"]))
        self.assertIsNone(rec["gpu_peak_mb"])
        _say("predict.py run --device cpu (AMP level-4 TINY, N 2): rc %d, %.0f s, visible %s"
             % (p.returncode, time.time() - t0, rec["visible_devices"]))


# ============================================================================ level 2: (c), (d)
class TestLevel2Convention(unittest.TestCase):
    """(c) convention evidence and (d) branch compositions on a TINY level-2 cascade (GPU)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgr_pred2_")
        cls.proc = os.path.join(cls.tmp, "proc")
        dirs, cls.images = write_subjects(cls.proc, 3, textured=True, seed=2)
        cls.sdirs = [os.path.basename(d) for d in dirs]
        cfg = tiny_config(**STAGE_KW)
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        randomize(m4, 3, 0.02)
        frozen = CS.FrozenStage.from_model(m4)
        m2 = GroupRegLevelModel(2, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
        m2.copy_weights_from(m4)
        randomize(m2, 4, 0.01)
        cls.loaded = LoadedModel(2, m2, frozen)
        cls._design_rigid_head(frozen.model, cls.images)
        cls.mags, cls.scale_steps = cls._scale_svf_heads(cls.loaded, cls.images, (frozen.model, m2))
        cls.ckpt = save_cascade_run(cls.tmp, "l2", m2, frozen)
        cls.ref = {b: _collect(cls.loaded, cls.images, b, streamed=True) for b in BRANCH_ORDER}
        cls.out, cls.codes = {}, {}
        for b in BRANCH_ORDER:
            out = os.path.join(cls.tmp, "out_" + b.replace("+", "_"))
            cls.codes[b] = PR.run_predict(_run(cls.ckpt, cls.proc, out, "--branches", b,
                                               "--check-internal", "3"), configure_device=False)
            cls.out[b] = out

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        free_memory()

    @staticmethod
    def _magnitudes(loaded, images):
        m = {}
        for b in ("rigid", "affine"):
            A, _, _ = _collect(loaded, images, b)
            m[b + "_rot_deg"] = max(rotation_deg(a[:3, :3]) for a in A)
            m[b + "_trans_mm"] = max(float(np.linalg.norm(a[:3, 3])) for a in A)
        _, U1, _ = _collect(loaded, images, "deform")
        m["u_max_vox"] = float(np.linalg.norm(U1, axis=-1).max())
        return m

    @staticmethod
    def _design_rigid_head(m4, images):
        """Set the frozen level-4 rigid head kernel to the minimum-norm solution of H K = T - b:
        per subject a chosen rotation (12 deg about z, -10 deg about x, 8 deg about y) and
        translation, so the cascade's centred A rotate by several degrees (doubling a random kernel
        does not work: the 10D decoding is scale invariant, the rotation saturates)."""
        x4 = _pool_to(tf.constant(images[..., None]), 4)
        h = m4(x4, task="rigid", training=False)["head_input"].numpy().astype(np.float64)
        head = m4.linear_decoder.rigid_head
        rots = (_rot(12.0, (0, 0, 1)), _rot(-10.0, (1, 0, 0)), _rot(8.0, (0, 1, 0)))
        trans = np.array([[7.0, -4.0, 3.0], [-5.0, 6.0, -2.0], [2.0, -6.0, 6.0]])
        target = np.concatenate([np.stack([tfm.encode_10d(r).numpy() for r in rots]),
                                 trans / float(m4.config.translation_scale)], axis=1)
        delta = target - head.bias.numpy().astype(np.float64)[None]
        k, *_ = np.linalg.lstsq(h, delta, rcond=None)
        head.kernel.assign(k.astype(np.float32))

    @classmethod
    def _scale_svf_heads(cls, loaded, images, models):
        """Double the SVF head kernels until max |U1| >= 2.5 vox (margin over the AC 3 minimum)."""
        steps = {"svf": 0}
        for _ in range(12):
            m = cls._magnitudes(loaded, images)
            if m["u_max_vox"] >= 2.5:
                break
            for model in models:
                k = model.deform_decoder.svf_head.kernel
                k.assign(k * 2.0)
            steps["svf"] += 1
        return m, steps

    def test_0_magnitudes(self):
        m = self.mags
        _say("level-2 cascade, level-4 rigid head designed, SVF heads x2^%d: %s"
             % (self.scale_steps["svf"], {k: round(v, 3) for k, v in m.items()}))
        for b in ("rigid", "affine"):
            self.assertGreaterEqual(m[b + "_rot_deg"], 5.0)
            self.assertGreaterEqual(m[b + "_trans_mm"], 3.0)
        self.assertGreaterEqual(m["u_max_vox"], 2.0)
        self.assertEqual(self.codes, {b: 0 for b in BRANCH_ORDER})

    def test_c_internal_checks_from_files(self):
        for b in BRANCH_ORDER:
            A, U1, U1i = self.ref[b]
            rec = _json(os.path.join(self.out[b], "predict.json"))
            self.assertEqual(rec["status"], "ok", (b, rec["message"], rec["internal_check"]))
            self.assertTrue(rec["internal_check"]["passed"], rec["internal_check"])
            # predict-levels test 6: default runs keep today's pipeline text, levels.default true
            self.assertEqual(rec["pipeline"], DEFAULT_PIPELINE_L2[b])
            self.assertEqual(rec["pipeline"], PR._pipeline(b, (4, 2)))
            lv = rec["levels"]
            self.assertTrue(lv["default"])
            self.assertIsNone(lv["requested"])
            self.assertEqual((lv["selected"], lv["skipped"], lv["truncated"]), ([4, 2], [], []))
            self.assertEqual(lv["exposure"]["entries"], [])
            self.assertFalse(lv["allow_untrained_skip"])
            for i, s in enumerate(self.sdirs):
                fwd, inv, moved = _files(self.out[b], s)
                r = CN.internal_check(self.images[i], moved, fwd, inv, b,
                                      None if A is None else A[i:i + 1],
                                      None if U1 is None else U1[i:i + 1],
                                      None if U1i is None else U1i[i:i + 1])
                _say("(c) %s %s from files: coord fwd %.2e / inv %.2e vox, moved %.2e%s, inverse "
                     "consistency %.2e mm (raw %.2e), interior %.0f %%" % (
                         b, s, r["coord_fwd_max_vox"], r["coord_inv_max_vox"],
                         r["moved_vs_internal_max_abs"], "" if r["moved_gated"] else " (not gated)",
                         r["inv_consistency_max_mm"], r["inv_consistency_raw_max_mm"],
                         100 * r["interior_fraction"]))
                self.assertLessEqual(r["coord_fwd_max_vox"], GATE, b)
                self.assertLessEqual(r["coord_inv_max_vox"], GATE, b)
                if b != "affine+deform":
                    self.assertLessEqual(r["moved_vs_internal_max_abs"], GATE, b)
                    self.assertLessEqual(r["inv_consistency_max_mm"], GATE, b)
                self.assertTrue(r["passed"])
                self.assertGreater(r["interior_fraction"], 0.5)

    def test_c_wrong_conversion_fails(self):
        for b in BRANCH_ORDER:
            A, U1, U1i = self.ref[b]
            a0 = None if A is None else np.linalg.inv(A[0:1].astype(np.float64)).astype(np.float32)
            u0, ui0 = (None, None) if U1 is None else (U1[0:1], U1i[0:1])
            fwd, inv = CN.to_canonical(b, a0, ui0, u0, GRID_SHAPE, (1.0, 1.0, 1.0))   # Mv(A), u <-> u_inv
            fwd, inv = fwd.numpy()[0], inv.numpy()[0]
            moved = F.warp(self.images[0], fwd, order=1)
            r = CN.internal_check(self.images[0], moved, fwd, inv, b, None if A is None else A[0:1], u0, ui0)
            _say("(c) wrong conversion %s: coord fwd %.3g / inv %.3g vox -> passed %s"
                 % (b, r["coord_fwd_max_vox"], r["coord_inv_max_vox"], r["passed"]))
            self.assertGreater(r["coord_fwd_max_vox"], GATE, b)
            self.assertGreater(r["coord_inv_max_vox"], GATE, b)
            self.assertFalse(r["passed"])

    def test_c_check_convention_and_evaluation(self):
        out = self.out["affine+deform"]
        tpl = os.path.join(out, "{subject_dir}_%s.nii.gz")
        conv = os.path.join(self.tmp, "conv.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = EV.main(["check-convention", "--processed-dir", self.proc, "--fwd", tpl % "fwd",
                            "--inv", tpl % "inv", "--moved", tpl % "moved", "--n-subjects", "3", "--out", conv])
        text = buf.getvalue()
        self.assertEqual(code, 0, text)
        self.assertIn("RESULT: use  --components xyz --signs +++ --units mm", text)
        res = _json(conv)
        best, second = res["ranking"][0], res["ranking"][1]
        self.assertEqual((best["components"], best["signs"], best["swap_roles"]), ("xyz", "+++", False))
        _say("(c) check-convention: best %.4f, second %.4f (%s %s swap %s), identity %.4f -> margins %.4f / %.4f"
             % (best["ncc"], second["ncc"], second["components"], second["signs"], second["swap_roles"],
                res["identity_ncc"], best["ncc"] - second["ncc"], best["ncc"] - res["identity_ncc"]))
        self.assertGreater(best["ncc"] - second["ncc"], 0.002)
        self.assertGreater(best["ncc"], res["identity_ncc"] + 0.002)
        ev = os.path.join(self.tmp, "eval")
        with contextlib.redirect_stdout(io.StringIO()):
            code = EV.main(["run", "--processed-dir", self.proc, "--method", "tiny_l2", "--fwd", tpl % "fwd",
                            "--inv", tpl % "inv", "--atlas", os.path.join(out, "atlas.nii.gz"),
                            "--atlas-seg", os.path.join(out, "atlas_seg.nii.gz"), "--components", "xyz",
                            "--signs", "+++", "--units", "mm", "--pairs", "2", "--out", ev])
        self.assertEqual(code, 0)
        summ = _json(os.path.join(ev, "summary.json"))
        self.assertEqual(summ["n_pairs"], 2)
        for sec, keys in (("subject_metrics", ("template_dice", "template_ncc", "fold_fwd_pct", "inv_err_fwd_mm")),
                          ("pair_metrics", ("pair_dice", "pair_ncc"))):
            for k in keys:
                self.assertTrue(np.isfinite(summ[sec][k]["mean"]), (sec, k))
        self.assertTrue(np.isfinite(summ["group"]["centrality_mean_mm"]))
        _say("(c) evaluation run --pairs 2: template_dice %.3f, pair_dice %.3f, inv_err_fwd %.3f mm, "
             "centrality %.3f mm" % (summ["subject_metrics"]["template_dice"]["mean"],
                                     summ["pair_metrics"]["pair_dice"]["mean"],
                                     summ["subject_metrics"]["inv_err_fwd_mm"]["mean"],
                                     summ["group"]["centrality_mean_mm"]))

    def _omega(self, b, A=None, U1=None):
        ones = tf.ones((1,) + GRID_SHAPE + (1,), tf.float32)
        return CN.interior_mask(CN.internal_moved(ones, b, A, U1))

    def test_d_rigid_composition(self):
        A = self.ref["rigid"][0]
        m_inv, _ = CN.voxel_matrices(A, GRID_SHAPE, (1.0, 1.0, 1.0))
        for i, s in enumerate(self.sdirs):
            R = m_inv[i, :3, :3]
            fwd, _, _ = _files(self.out["rigid"], s)
            om = self._omega("rigid", A[i:i + 1])
            om[::2] = False
            om[:, ::2] = False
            p = np.argwhere(om).astype(np.float64)
            q = p + fwd[om].astype(np.float64)
            X, *_ = np.linalg.lstsq(np.concatenate([p, np.ones((len(p), 1))], 1), q, rcond=None)
            M = X[:3].T
            d_fit = float(np.abs(M - R).max())
            dets = (abs(np.linalg.det(R) - 1), abs(np.linalg.det(M) - 1))
            orth = (float(np.abs(R.T @ R - np.eye(3)).max()), float(np.abs(M.T @ M - np.eye(3)).max()))
            _say("(d) rigid %s: |R_fit - R| %.2e, |det - 1| runner %.2e / fit %.2e, |R^T R - I| runner %.2e / "
                 "fit %.2e, rotation %.2f deg" % (s, d_fit, dets[0], dets[1], orth[0], orth[1], rotation_deg(R)))
            self.assertLessEqual(d_fit, 1e-4)
            self.assertLessEqual(dets[0], 1e-5)
            self.assertLessEqual(orth[0], 1e-5)
            self.assertLessEqual(dets[1], 1e-4)
            self.assertLessEqual(orth[1], 1e-4)

    def test_d_affine_and_deform_compositions(self):
        A = self.ref["affine"][0]
        _, U1, U1i = self.ref["deform"]
        _, U1u, U1ui = _collect(self.loaded, self.images, "deform")          # unchunked cascade
        for i, s in enumerate(self.sdirs):
            fwd, inv, _ = _files(self.out["affine"], s)
            ref_f, ref_i = _np_fields(A[i].astype(np.float64), GRID_SHAPE, (1.0, 1.0, 1.0))
            e = (_diff(fwd, ref_f), _diff(inv, ref_i))
            fwd_d, inv_d, _ = _files(self.out["deform"], s)
            e_d = (_diff(fwd_d, U1[i]), _diff(inv_d, U1i[i]))
            e_u = (_diff(fwd_d, U1u[i]), _diff(inv_d, U1ui[i]))
            _say("(d) %s: affine fwd / inv vs Mv(A^-1) p - p, Mv(A) y - y (numpy, the script's A): %.2e / "
                 "%.2e vox; deform fwd / inv vs the deform cascade on the raw images: streamed %.2e / %.2e, "
                 "unchunked %.2e / %.2e vox" % (s, e[0], e[1], e_d[0], e_d[1], e_u[0], e_u[1]))
            self.assertLessEqual(max(e), B_STREAM)
            self.assertLessEqual(max(e_d), B_EXACT_FALLBACK)
            self.assertLessEqual(max(e_u), B_STREAM)

    def test_d_affine_deform_composition(self):
        """affine+deform == the deform cascade on the images resampled with the script's A, composed."""
        A = self.ref["affine+deform"][0]
        np.testing.assert_array_equal(A, self.ref["affine"][0])
        x = CS.prewarp_linear(tf.constant(self.images[..., None]), tf.constant(A), 1.0).numpy()[..., 0]
        _, U1, U1i = _collect(self.loaded, x, "deform")
        del x
        _, m = CN.voxel_matrices(A, GRID_SHAPE, (1.0, 1.0, 1.0))
        m_inv, _ = CN.voxel_matrices(A, GRID_SHAPE, (1.0, 1.0, 1.0))
        p = np.moveaxis(np.indices(GRID_SHAPE, dtype=np.float64), 0, -1)
        for i, s in enumerate(self.sdirs):
            fwd, inv, _ = _files(self.out["affine+deform"], s)
            q = p + U1[i]
            ref_f = q @ m_inv[i, :3, :3].T + m_inv[i, :3, 3] - p
            z = p @ m[i, :3, :3].T + m[i, :3, 3]
            zc = np.moveaxis(z, -1, 0)
            u_inv_z = np.stack([ndimage.map_coordinates(U1i[i][..., c], zc, order=1, mode="nearest")
                                for c in range(3)], -1)
            ref_i = z + u_inv_z - p
            om = self._omega("affine", A[i:i + 1]) & self._omega("affine+deform", A[i:i + 1], U1[i:i + 1])
            e = (_diff(fwd, ref_f), _diff(inv, ref_i, om))
            _say("(d) affine+deform %s: fwd vs Mv(A^-1)(p + u) - p %.2e vox (all voxels), inv vs z + u_inv(z) - y "
                 "%.2e vox (interior)" % (s, e[0], e[1]))
            self.assertLessEqual(max(e), B_STREAM)

    def test_e_check_streaming_end_to_end(self):
        out = os.path.join(self.tmp, "check.json")
        code = PR.run_check_streaming(_parse("check-streaming", "--checkpoint", self.ckpt, "--processed-dir",
                                             self.proc, "--out", out, "--n-subjects", "3"),
                                      configure_device=False)
        rec = _json(out)
        self.assertEqual(code, 0, rec.get("message"))
        self.assertEqual(rec["status"], "pass")
        self.assertEqual(rec["branches"], "deform")
        self.assertEqual(rec["chunks"], {"l2": 2, "l4": 2, "source": {"l2": "default", "l4": "default"},
                                         "capped": {}})
        for lv in ("l2", "l4"):
            self.assertEqual(rec["chunk_bounds"]["streamed"][lv]["deform"], [[0, 2], [2, 3]], lv)
            self.assertEqual(rec["chunk_bounds"]["unchunked"][lv]["deform"], [[0, 3]], lv)
        self.assertEqual(set(rec["gpu_peak_mb"]), {"unchunked", "streamed"})
        self.assertIsNotNone(rec["gpu_peak_mb"]["streamed"])
        self.assertEqual(rec["grid"]["shape"], list(GRID_SHAPE))
        self.assertEqual(rec["precision"]["bound_set"], "float32")
        self.assertEqual(rec["n_subjects"], 3)
        self.assertEqual(rec["checkpoint"]["step"], 5)
        _say("check-streaming L2 N 3 deform: %s; gpu_peak_mb %s"
             % ({k: "%.2e" % v for k, v in rec["differences"]["deform"].items()}, rec["gpu_peak_mb"]))
        # affine+deform on the same cascade: recorded only (the designed rigid head amplifies float
        # rounding of the head input; the cluster gate is deform at float32, design 9).
        out2 = os.path.join(self.tmp, "check_ad.json")
        code2 = PR.run_check_streaming(_parse("check-streaming", "--checkpoint", self.ckpt, "--processed-dir",
                                              self.proc, "--out", out2, "--n-subjects", "3", "--branches",
                                              "affine+deform"), configure_device=False)
        rec2 = _json(out2)
        self.assertIn(code2, (0, 4))
        for lv in ("l2", "l4"):
            for task in ("affine", "deform"):
                self.assertEqual(rec2["chunk_bounds"]["streamed"][lv][task], [[0, 2], [2, 3]], (lv, task))
        _say("check-streaming L2 N 3 affine+deform (recorded): exit %d, %s" % (
            code2, {k: "%.2e" % v for k, v in rec2["differences"]["affine+deform"].items()}))


# ============================================================================ predict-levels
SKIP_ARGS = {"cascade_skip_prob": 0.0,
             "cascade_skip_stage_parsed": {"rigid": {"4": 0.0}, "affine": {"4": 0.0}, "deform": {"4": 0.15}}}


class TestLevelsEndToEnd(unittest.TestCase):
    """predict-levels design 11 (end to end) on a TINY level-2 cascade whose config.json says deform
    level 4 was skipped with p 0.15 and linear never: `run --levels 4 / 2` (files, predict.json
    `levels`, internal check, written fields == the runner under the plan, check-convention),
    refusal / --allow-untrained-skip, level not in the checkpoint, low-exposure WARNING,
    check-streaming --levels."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgr_predlv_")
        cls.proc = os.path.join(cls.tmp, "proc")
        dirs, cls.images = write_subjects(cls.proc, 3, textured=True, seed=2)
        cls.sdirs = [os.path.basename(d) for d in dirs]
        cfg = tiny_config(**STAGE_KW)
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        randomize(m4, 3, 0.02)
        frozen = CS.FrozenStage.from_model(m4)
        m2 = GroupRegLevelModel(2, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
        m2.copy_weights_from(m4)
        randomize(m2, 4, 0.01)
        cls.loaded = LoadedModel(2, m2, frozen)
        TestLevel2Convention._scale_svf_heads(cls.loaded, cls.images, (frozen.model, m2))
        # Each selection keeps only part of the cascade's field: double the SVF heads until both
        # `4` and `2` reach max |U1| >= 4 vox (else check-convention cannot separate the
        # conventions: AMBIGUOUS at max |U1| 2.1 vox, mean |fwd| 0.43 mm).
        from inference import levels as LV
        cls.svf_doublings = 0
        for _ in range(4):
            mags = [float(np.abs(_collect(cls.loaded, cls.images, "deform", streamed=True,
                                          plan=LV.plan((4, 2), sel))[1]).max()) for sel in ((4,), (2,))]
            if min(mags) >= 4.0:
                break
            for model in (frozen.model, m2):
                k = model.deform_decoder.svf_head.kernel
                k.assign(k * 2.0)
            cls.svf_doublings += 1
        cls.mags = mags
        cls.ckpt = save_cascade_run(cls.tmp, "l2skip", m2, frozen, args=SKIP_ARGS)
        low = {"cascade_skip_prob": 0.0, "cascade_skip_stage_parsed": {"deform": {"4": 0.04}}}
        cls.ckpt_low = save_cascade_run(cls.tmp, "l2low", m2, frozen, args=low)
        cls.out, cls.codes = {}, {}
        for sel in ("4", "2"):
            out = os.path.join(cls.tmp, "out_L%s" % sel)
            cls.codes[sel] = PR.run_predict(_run(cls.ckpt, cls.proc, out, "--branches", "deform", "--levels",
                                                 sel, "--check-internal", "3"), configure_device=False)
            cls.out[sel] = out

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        free_memory()

    def test_run_levels_files_and_record(self):
        from inference import levels as LV
        _say("levels e2e: SVF heads doubled %d more times; last measured max |U1| for `4` / `2`: %s"
             % (self.svf_doublings, ["%.2f" % m for m in self.mags]))
        self.assertEqual(self.codes, {"4": 0, "2": 0})
        expect = {"4": ((4,), [], [2], "deform cascade (L4; L2 not run (truncated after level 4)) on the raw images"),
                  "2": ((2,), [4], [], "deform cascade (L4 skipped (identity) -> L2) on the raw images")}
        for sel, (selected, skipped, trunc, pipe) in expect.items():
            out = self.out[sel]
            rec = _json(os.path.join(out, "predict.json"))
            self.assertEqual(rec["status"], "ok", rec["message"])
            self.assertTrue(rec["internal_check"]["passed"], rec["internal_check"])
            self.assertTrue(rec["pipeline"].startswith(pipe), rec["pipeline"])
            lv = rec["levels"]
            self.assertEqual((lv["requested"], lv["selected"], lv["skipped"], lv["truncated"], lv["default"]),
                             (sel, list(selected), skipped, trunc, False))
            self.assertEqual(lv["checkpoint_levels"], [4, 2])
            self.assertEqual(lv["upsampled_from_level"], selected[-1])
            self.assertFalse(lv["exposure"]["forced"])
            self.assertEqual(sorted(rec["chunks"]["bounds"]), ["l%d" % selected[-1]])
            if sel == "2":
                ent = lv["exposure"]["entries"]
                self.assertEqual(len(ent), 1)
                self.assertEqual((ent[0]["task"], ent[0]["level"], ent[0]["status"], ent[0]["p"]),
                                 ("deform", 2, "trained", {"4": 0.15}))
                self.assertEqual(lv["exposure"]["runs"]["2"]["source"], "args.cascade_skip_stage_parsed")
            else:
                self.assertEqual(lv["exposure"]["entries"], [])
            for name in ("atlas.nii.gz", "atlas_seg.nii.gz"):
                self.assertTrue(os.path.isfile(os.path.join(out, name)), name)
            plan = LV.plan((4, 2), selected)
            _, U1, U1i = _collect(self.loaded, self.images, "deform", streamed=True, plan=plan)
            e = 0.0
            for i, s in enumerate(self.sdirs):
                fwd, inv, moved = _files(out, s)
                self.assertEqual(fwd.shape, GRID_SHAPE + (3,))
                e = max(e, _diff(fwd, U1[i]), _diff(inv, U1i[i]))
            _say("--levels %s: written fields vs the streamed runner under the plan %.2e vox; max |U1| %.3f"
                 % (sel, e, float(np.abs(U1).max())))
            self.assertLessEqual(e, B_EXACT_FALLBACK)

    def test_check_convention_on_skip_output(self):
        for sel in ("2", "4"):
            out = self.out[sel]
            tpl = os.path.join(out, "{subject_dir}_%s.nii.gz")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = EV.main(["check-convention", "--processed-dir", self.proc, "--fwd", tpl % "fwd",
                                "--inv", tpl % "inv", "--moved", tpl % "moved", "--n-subjects", "3", "--out",
                                os.path.join(self.tmp, "conv_%s.json" % sel)])
            self.assertEqual(code, 0, buf.getvalue())
            self.assertIn("RESULT: use  --components xyz --signs +++ --units mm", buf.getvalue())
            best = _json(os.path.join(self.tmp, "conv_%s.json" % sel))["ranking"][0]
            self.assertEqual((best["components"], best["signs"], best["swap_roles"]), ("xyz", "+++", False))
            _say("--levels %s check-convention: best %s %s ncc %.4f" % (sel, best["components"], best["signs"],
                                                                      best["ncc"]))

    def test_refusal_and_force(self):
        out = os.path.join(self.tmp, "out_refused")
        with self.assertLogs("inference.predict", "ERROR"):
            code = PR.run_predict(_run(self.ckpt, self.proc, out, "--branches", "affine", "--levels", "2"),
                                  configure_device=False)
        self.assertEqual(code, CLI.EXIT_INPUT)
        rec = _json(os.path.join(out, "predict.json"))
        self.assertEqual(rec["status"], "error")
        self.assertIn("--allow-untrained-skip", rec["message"])
        ent = rec["levels"]["exposure"]["entries"]
        self.assertEqual([(e["task"], e["level"], e["status"]) for e in ent], [("affine", 2, "untrained")])
        self.assertIsNone(rec["n_subjects"])                     # refused before reading subjects
        self.assertFalse([f for f in os.listdir(out) if f.endswith("_fwd.nii.gz")])
        out2 = os.path.join(self.tmp, "out_forced")
        code2 = PR.run_predict(_run(self.ckpt, self.proc, out2, "--branches", "affine", "--levels", "2",
                                    "--allow-untrained-skip"), configure_device=False)
        rec2 = _json(os.path.join(out2, "predict.json"))
        self.assertEqual(code2, 0, rec2["message"])
        self.assertTrue(rec2["levels"]["exposure"]["forced"])
        self.assertTrue(rec2["levels"]["allow_untrained_skip"])
        self.assertTrue(any("forced (--allow-untrained-skip)" in w for w in rec2["warnings"]), rec2["warnings"])
        self.assertTrue(os.path.isfile(os.path.join(out2, self.sdirs[0] + "_fwd.nii.gz")))
        _say("refusal: %s" % rec["message"])

    def test_level_not_in_checkpoint(self):
        out = os.path.join(self.tmp, "out_l1")
        with self.assertLogs("inference.predict", "ERROR"):
            code = PR.run_predict(_run(self.ckpt, self.proc, out, "--branches", "deform", "--levels", "1"),
                                  configure_device=False)
        self.assertEqual(code, CLI.EXIT_INPUT)
        rec = _json(os.path.join(out, "predict.json"))
        self.assertIn("level-2 run with levels [4, 2]", rec["message"])
        self.assertEqual(rec["levels"], {"requested": "1", "checkpoint_levels": [4, 2], "selected": None})

    def test_low_exposure_warning(self):
        args = _run(self.ckpt_low, self.proc, os.path.join(self.tmp, "out_low"), "--branches", "deform",
                    "--levels", "2")
        rec = {}
        with self.assertLogs("inference.predict", "WARNING") as logs:
            plan = PR._setup_levels(args, rec, 2)
        self.assertEqual(plan.selected, (2,))
        self.assertTrue(any("about 4.00 % of training steps" in m for m in logs.output), logs.output)
        self.assertEqual(rec["levels"]["exposure"]["min_joint"], {"deform": 0.04})

    def test_check_streaming_levels(self):
        for sel in ("2", "4"):
            out = os.path.join(self.tmp, "check_L%s.json" % sel)
            code = PR.run_check_streaming(_parse("check-streaming", "--checkpoint", self.ckpt, "--processed-dir",
                                                 self.proc, "--out", out, "--n-subjects", "3", "--levels", sel),
                                          configure_device=False)
            rec = _json(out)
            self.assertEqual(code, 0, rec.get("message"))
            self.assertEqual(rec["status"], "pass")
            key = "l" + sel
            self.assertEqual(rec["chunks"], {key: 2, "source": {key: "default"}, "capped": {}})
            self.assertEqual(rec["chunk_bounds"]["streamed"], {key: {"deform": [[0, 2], [2, 3]]}})
            self.assertEqual(rec["chunk_bounds"]["unchunked"], {key: {"deform": [[0, 3]]}})
            self.assertEqual(rec["levels"]["selected"], [int(sel)])
            _say("check-streaming --levels %s: %s" % (sel, {k: "%.2e" % v for k, v in
                                                          rec["differences"]["deform"].items()}))


# ============================================================================ level 1: (b) files
class TestFilesVsTrainerLevel1(unittest.TestCase):
    """AC 2 level-1 file part: written _fwd / _inv (mm / spacing) vs the trainer's u / u_inv."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_pred1_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        free_memory()

    def test_level1_files_vs_trainer_cpu(self):
        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("tf_gpu env requires a visible GPU (the level-1 case runs on /CPU:0)")
        t0 = time.time()
        n = 3
        proc = os.path.join(self.tmp, "proc")
        dirs, images = write_subjects(proc, n, textured=False, seed=4)
        sdirs = [os.path.basename(d) for d in dirs]
        with tf.device("/CPU:0"):
            loaded = _level1()
            ckpt = save_chain_run(self.tmp, "l1", loaded.model, loaded.frozen)
            cfg = chain_config(amp=False, val_pair_chunk=1, n_classes=N_CLASSES, group_size=n,
                               deform_group_size=n)
            tr = Trainer(loaded.model, O.make_optimizer("adam", lr=1e-3, amp=False),
                         {"rigid": None, "deform": None}, np.eye(3) * 1e3, cfg, cascade=loaded.frozen)
            with _OpDeterminism():
                ref = tr._val_deform_fn(n)(tf.constant(images[..., None]))
                u_ref, ui_ref = ref["u"].numpy(), ref["u_inv"].numpy()
            del ref, tr, loaded
            self.assertGreater(float(np.abs(u_ref).max()), 1e-3)
            for mode, extra, bound in (("unchunked", ("--mode", "unchunked"), B_EXACT_FALLBACK),
                                       ("streamed chunk 1", ("--chunk-l1", "1"), B_STREAM),
                                       ("streamed chunk 3", ("--chunk-l1", "3"), B_STREAM)):
                out = os.path.join(self.tmp, "out_" + mode.replace(" ", "_"))
                t = time.time()
                with _OpDeterminism():
                    code = PR.run_predict(_run(ckpt, proc, out, "--branches", "deform", "--check-internal", "1",
                                               *extra), configure_device=False)
                self.assertEqual(code, 0, _json(os.path.join(out, "predict.json")).get("message"))
                w = {"fwd": 0.0, "inv": 0.0}
                for i, s in enumerate(sdirs):
                    fwd, inv, _ = _files(out, s)
                    w["fwd"] = max(w["fwd"], _diff(fwd / 1.0, u_ref[i]))
                    w["inv"] = max(w["inv"], _diff(inv / 1.0, ui_ref[i]))
                    del fwd, inv
                rec = _json(os.path.join(out, "predict.json"))
                tag = ("S_exact" if max(w.values()) <= B_EXACT else "S_exact FALLBACK 1e-5") \
                    if mode == "unchunked" else "S_stream"
                _say("(b) L1 files vs trainer, %s: fwd %.2e, inv %.2e vox (%s); internal check %s; %.0f s"
                     % (mode, w["fwd"], w["inv"], tag,
                        {k: round(v["coord_fwd_max_vox"], 7) for k, v in rec["internal_check"]["subjects"].items()},
                        time.time() - t))
                self.assertLessEqual(max(w.values()), bound, mode)
                self.assertTrue(rec["internal_check"]["passed"])
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        _say("(b) L1 files vs trainer on CPU: total %.0f s, peak RSS %.1f GB" % (time.time() - t0, rss_gb))


if __name__ == "__main__":
    unittest.main()
