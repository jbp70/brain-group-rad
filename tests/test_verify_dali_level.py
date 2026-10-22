"""T16a (milestone 8, design section 8): tests/gpu/verify_dali_pipeline.py --level.

(i) Source scan: no literal level-4 template mask / label file name is left in the script.
(ii) `_set_level(2)` rebinds LEVEL / SHAPE / SPACING and `_template_brain(args)` loads the
level-2 template mask; `_set_level(4)` restores the defaults (tearDown).
CPU only; DALI is optional (the module imports it lazily). Skipped if the module cannot be
imported.
"""
import tests.tf_setup  # noqa: F401  (TF32 off, GPU memory setup) -- must come first

import argparse
import importlib.util
import os
import re
import shutil
import tempfile
import unittest

import numpy as np

from data_loading import params as P

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "verify_dali_pipeline.py")


def _load():
    spec = importlib.util.spec_from_file_location("verify_dali_pipeline", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    vdp = _load()
    _IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    vdp = None
    _IMPORT_ERROR = repr(e)


class TestSourceScan(unittest.TestCase):

    def test_no_level4_literals(self):
        with open(_PATH) as f:
            text = f.read()
        self.assertIsNone(re.search(r"template_mask_4", text))
        self.assertIsNone(re.search(r"labels_train18_4", text))
        self.assertIn('"template_mask%s.npy" % P.level_suffix(LEVEL)', text)
        self.assertIn("_set_level(args.level)", text)


@unittest.skipIf(vdp is None, "verify_dali_pipeline not importable: %s" % (_IMPORT_ERROR,))
class TestSetLevel(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vdp_level_")

    def tearDown(self):
        vdp._set_level(4)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_level2_template(self):
        tdir = os.path.join(self.tmp, vdp.dm.TEMPLATE_DIR)
        os.makedirs(tdir)
        mask = np.zeros(P.LEVEL_SHAPE[2], np.uint8)
        mask[20:60, 30:80, 20:70] = 1
        np.save(os.path.join(tdir, "template_mask_2.npy"), mask)
        vdp._set_level(2)
        self.assertEqual((vdp.LEVEL, vdp.SHAPE, vdp.SPACING), (2, P.LEVEL_SHAPE[2], P.LEVEL_SPACING[2]))
        brain = vdp._template_brain(argparse.Namespace(data=self.tmp))
        self.assertEqual(brain.shape, (80, 112, 96))
        self.assertEqual(brain.dtype, np.bool_)
        self.assertEqual(int(brain.sum()), int(mask.sum()))
        vdp._set_level(4)
        self.assertEqual((vdp.LEVEL, vdp.SHAPE, vdp.SPACING), (4, P.LEVEL_SHAPE[4], P.LEVEL_SPACING[4]))


if __name__ == "__main__":
    unittest.main()
