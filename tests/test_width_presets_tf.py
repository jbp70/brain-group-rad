"""`models.width_presets.count_params` vs built level-4 models (milestone 9 design 3.1).

Every preset is built at level 4 with the deform decoder (N = 1; the variables do not depend on
the level or N), and the TF-free count must be within 1 % of the built trainable count, per
part and in total. default 29.63 M and f0d1 29.54 M (built) are pinned. Also the narrow and tiny
test configs and a residual / config-B variant.
"""
import sys
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.runtime_helpers import NARROW, TINY, free_memory

from models import width_presets as W
from models.config import ModelConfig
from models.group_net import GroupRegLevelModel


def _built(cfg):
    model = GroupRegLevelModel(4, cfg).build_model(1)
    c = model.count_params_by_part()
    out = {"encoder": c["encoder"], "linear_decoder": c["linear_decoder"],
           "deform_decoder": c.get("deform_decoder", 0), "total": c["trainable"]}
    del model
    free_memory()
    return out


class TestCountParams(unittest.TestCase):
    def check(self, cfg, label):
        got = W.count_params(cfg)
        want = _built(cfg)
        print("\n[presets tf] %-8s built %.4f M, count_params %.4f M" % (label, want["total"] / 1e6,
                                                                    got["total"] / 1e6), file=sys.stderr)
        for k in ("encoder", "linear_decoder", "deform_decoder", "total"):
            self.assertLessEqual(abs(got[k] - want[k]), 0.01 * max(want[k], 1), (label, k, got, want))
        return want

    def test_presets(self):
        for name in W.PRESETS:
            want = self.check(ModelConfig(deform_decoder=True, **W.preset_kwargs(name)), name)
            if name == "default":
                self.assertAlmostEqual(want["total"] / 1e6, 29.63, delta=0.005)
            if name == "f0d1":
                self.assertAlmostEqual(want["total"] / 1e6, 29.54, delta=0.005)

    def test_other_configs(self):
        self.check(ModelConfig(deform_decoder=True, **NARROW), "narrow")
        self.check(ModelConfig(deform_decoder=True, **TINY), "tiny")
        self.check(ModelConfig(**NARROW), "narrow-lin")
        self.check(ModelConfig(block="residual", group_config="B", **NARROW), "res-B")


if __name__ == "__main__":
    unittest.main()
