import json, os
R = "/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results"
runs = ["r4_cfgB_dice1_smooth05_lr3e-4_wd1e-1", "r2_cfgB_s15000_smooth10_lr3e-4_wd1e-1_n32_d16",
        "r2_cfgB_s15000_smooth0.5_lr3e-4_wd1e-1_n32_d16",
        "r1_cfgB_sm10_s2500_bnfreeze_true"]
def walk(d, pre=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from walk(v, pre + "/" + str(k) if pre else str(k))
    else:
        yield pre, d
for r in runs:
    p = os.path.join(R, r, "logs", "metrics.jsonl")
    last = None
    for line in open(p):
        if "inv_consistency_mm" in line:
            last = line
    d = json.loads(last)
    print("==", r)
    for k, v in walk(d):
        if any(t in k for t in ("inv_consistency", "centrality", "neg_jac", "step")) and "max" not in k or k.endswith("inv_consistency_max_mm"):
            print("  ", k, v)
