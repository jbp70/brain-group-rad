"""synth-contrast FEAT-004 (design 7.2, plan deviation D-d): scripts/synth_preview.py on a tiny
fake export (level 4, train18 + synthfine labels) writes a valid PNG grid; a missing label set
exits with a clear message."""
from tests import tf_setup  # noqa: F401  (must be first: GPU memory settings)

import contextlib
import importlib.util
import io
import os
import shutil
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402

from tests import runtime_helpers as rh  # noqa: E402

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "synth_preview.py")
_spec = importlib.util.spec_from_file_location("synth_preview", _PATH)
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)


class TestSynthPreview(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_synth_preview_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_png_grid(self):
        data = rh.write_fake_export(self.tmp, n_train=4, labels=True, fine_labels=True)
        png = os.path.join(self.tmp, "out", "l4.png")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = sp.main(["--data", data, "--level", "4", "--n", "4", "--groups", "2", "--seed", "1", "--out", png])
        self.assertEqual(rc, 0)
        self.assertIn(png, out.getvalue())
        img = mpimg.imread(png)
        self.assertEqual(img.ndim, 3)
        self.assertGreater(img.shape[0], 100)
        self.assertGreater(img.shape[1], 100)
        self.assertIn(img.shape[2], (3, 4))

    def test_missing_label_set(self):
        data = rh.write_fake_export(self.tmp, n_train=4, labels=True, fine_labels=False)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            sp.main(["--data", data, "--level", "4", "--n", "4", "--out", os.path.join(self.tmp, "x.png")])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("no 'synthfine' labels at level 4", err.getvalue())
        self.assertIn("export-add-labels", err.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "x.png")))


if __name__ == "__main__":
    unittest.main()
