"""T9 guard of the TTA implementation (TTA design 8 T9, review v2 NIT 1): `predict.py run` with
``--tta none`` is bit-identical to the code before any TTA change.

The fixture tests/fixtures/predict_tta_off_golden.json was written ONCE on the unchanged tree
(commit 19b8a60, recorded in the fixture) with BGR_WRITE_GOLDEN=1, and checked by a second run on
the same tree without the env (step 0 of the review rule). It holds sha256 hashes of the DECODED
outputs (for every *.nii.gz: ``np.asarray(dataobj).tobytes() + affine.tobytes()``; not the file
bytes, whose gzip header carries a timestamp) of `run` on a tiny randomised level-4 checkpoint and
a tiny level-2 cascade for the branches rigid, deform and affine+deform, plus the A matrices of
an in-process `StreamedRunner.run` (float64 bytes) for rigid and affine+deform. Every run happens
under `_OpDeterminism` (tests/test_trainer.py). Fixture 'mode' = 'hash' (stored hashes, tolerance
0) is the rule in use; see .agents/tasks/tta/test-log-FEAT-001.md.
"""
from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory limit)
from tests.predict_helpers import save_cascade_run, save_level4_run, write_subjects
from tests.runtime_helpers import free_memory, tiny_config
from tests.test_cascade import STAGE_KW, randomize
from tests.test_trainer import _OpDeterminism
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

from inference import cli as CLI
from inference import predict as PR
from inference.loader import load_model
from inference.runner import StreamedRunner
from inference.store import RamStore
from models import cascade as CS
from models.group_net import GroupRegLevelModel

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                           "predict_tta_off_golden.json")
BRANCHES = ("rigid", "deform", "affine+deform")
A_BRANCHES = ("rigid", "affine+deform")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _nifti_hash(path):
    img = nib.load(path)
    return _sha(np.asarray(img.dataobj).tobytes() + np.asarray(img.affine).tobytes())


def _golden_run(tmp):
    """{key: sha256} for every (level, branch) output file and the in-process A."""
    proc = os.path.join(tmp, "proc")
    _, images = write_subjects(proc, 3, textured=True, seed=11)
    cfg = tiny_config(**STAGE_KW)
    ck4, _ = save_level4_run(tmp, "l4", cfg, seed=5)
    m4 = GroupRegLevelModel(4, cfg).build_model(2)
    randomize(m4, 3, 0.02)
    frozen = CS.FrozenStage.from_model(m4)
    m2 = GroupRegLevelModel(2, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
    m2.copy_weights_from(m4)
    randomize(m2, 4, 0.01)
    ck2 = save_cascade_run(tmp, "l2", m2, frozen)
    hashes = {}
    for lname, ck in (("l4", ck4), ("l2", ck2)):
        for b in BRANCHES:
            out = os.path.join(tmp, "out_%s_%s" % (lname, b.replace("+", "_")))
            args = CLI.parse_args(["run", "--checkpoint", ck, "--processed-dir", proc, "--out", out,
                                   "--branches", b, "--tta", "none", "--store", "ram",
                                   "--check-internal", "0"])
            code = PR.run_predict(args, configure_device=False)
            if code != 0:
                raise AssertionError("predict run %s %s exited %d" % (lname, b, code))
            files = sorted(glob.glob(os.path.join(out, "*.nii.gz")))
            if not files:
                raise AssertionError("no NIfTI outputs in %s" % out)
            for p in files:
                hashes["%s/%s/%s" % (lname, b, os.path.basename(p))] = _nifti_hash(p)
            if b in A_BRANCHES:
                loaded = load_model(ck, b, precision="checkpoint", device="gpu")
                summary = StreamedRunner(loaded, RamStore()).run(images, b)
                hashes["%s/%s/A" % (lname, b)] = _sha(np.asarray(summary["A"], np.float64).tobytes())
                del loaded
            free_memory()
    return hashes


class TestTTAOffGolden(unittest.TestCase):
    """`run --tta none` outputs (decoded arrays + affines) and A == the pre-TTA fixture, exactly."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgr_tta_golden_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        free_memory()

    def test_matches_golden(self):
        with _OpDeterminism():
            got = _golden_run(self.tmp)
        if os.environ.get("BGR_WRITE_GOLDEN") == "1":
            sha, _ = PR.git_commit()
            os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
            with open(GOLDEN_PATH, "w") as f:
                json.dump({"commit": sha, "mode": "hash", "tolerance": 0, "hashes": got}, f,
                          indent=1, sort_keys=True)
            sys.stderr.write("\n[golden] wrote %s (%d hashes, commit %s)\n" % (GOLDEN_PATH, len(got), sha))
        with open(GOLDEN_PATH) as f:
            want = json.load(f)
        self.assertEqual(want["mode"], "hash")
        self.assertEqual(want["tolerance"], 0)
        self.assertEqual(sorted(got), sorted(want["hashes"]))
        bad = sorted(k for k in got if got[k] != want["hashes"][k])
        sys.stderr.write("\n[golden tta-off] %d hashes compared, %d differ%s (fixture commit %s)\n"
                         % (len(got), len(bad), (": " + ", ".join(bad)) if bad else "", want["commit"]))
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
