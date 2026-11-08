import json, os, csv
R = "/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results"
runs = ["r4_cfgB_dice1_smooth05_lr3e-4_wd1e-1", "r2_cfgB_s15000_smooth10_lr3e-4_wd1e-1_n32_d16",
        "r1_cfgB_sm10_s2500_bnfreeze_true", "r4sw_C_s16000_lr3e-4_wd1e-1_n128_d64"]
for r in runs:
    p = os.path.join(R, r, "logs", "metrics.jsonl")
    if not os.path.isfile(p):
        print(r, "no metrics"); continue
    last = None
    for line in open(p):
        if '"val_deform/inv_consistency_mm"' in line:
            last = line
    if last is None:
        print(r, "no val_deform"); continue
    d = json.loads(last)
    ks = sorted(k for k in d if k.startswith("val_deform") and any(
        t in k for t in ("inv_consistency", "centrality", "neg_jac", "template_dice", "pair_dice")))
    print("==", r, "step", d.get("step"))
    for k in ks:
        print("  %s %s" % (k, d[k]))

print()
def load(path):
    rows = list(csv.DictReader(open(path)))
    return {int(r["label"]): r for r in rows}
ours = load(R + "/predict_mindboggle/r1_cfgB_sm10_s2500_bnfreeze_true_deform/eval_template/per_label.csv")
mm = load(R + "/eval_mindboggle/multimorph_baseline/per_label.csv")
ants = load(R + "/eval_mindboggle/ants_syn/per_label.csv")
l4 = load(R + "/predict_mindboggle/r2_cfgB_smooth10_s15000_deform/eval_template/per_label.csv") if os.path.isfile(R + "/predict_mindboggle/r2_cfgB_smooth10_s15000_deform/eval_template/per_label.csv") else load(R + "/predict_mindboggle/r2_cfgB_smooth10_s15000_deform/per_label.csv")
oldA = load(R + "/predict_mindboggle/r1sw_lr1e-4_s3000_deform/eval_template/per_label.csv") if os.path.isfile(R + "/predict_mindboggle/r1sw_lr1e-4_s3000_deform/eval_template/per_label.csv") else load(R + "/predict_mindboggle/r1sw_lr1e-4_s3000_deform/per_label.csv")
print("label cort ours mm ants l2s10 oldA_L421 | ours-mm  (template_dice); transfer ours/mm; loo ours/mm")
items = []
for lab in sorted(ours):
    o, m = ours[lab], mm.get(lab)
    if m is None: continue
    f = lambda r, k="template_dice": float(r[k]) if r and r.get(k) not in (None, "", "nan") else float("nan")
    items.append((lab, o["cortical"], f(o), f(m), f(ants.get(lab)), f(l4.get(lab)), f(oldA.get(lab)),
                  f(o, "transfer_dice"), f(m, "transfer_dice"), f(o, "template_loo_dice"), f(m, "template_loo_dice")))
for it in items:
    if it[1] in ("False", "false", "0"):
        print("%5d %s %.3f %.3f %.3f %.3f %.3f | %+.3f ; tr %.3f/%.3f ; loo %.3f/%.3f" % (
            it[0], it[1][0], it[2], it[3], it[4], it[5], it[6], it[2] - it[3], it[7], it[8], it[9], it[10]))
cort = [it for it in items if it[1] not in ("False", "false", "0")]
import math
def mean(x):
    x = [v for v in x if not math.isnan(v)]
    return sum(x) / len(x)
print("cortical n=%d ours %.4f mm %.4f; loo ours %.4f mm %.4f" % (len(cort), mean([c[2] for c in cort]), mean([c[3] for c in cort]), mean([c[9] for c in cort]), mean([c[10] for c in cort])))
sub = [it for it in items if it[1] in ("False", "false", "0")]
print("subcortical n=%d ours %.4f mm %.4f; loo ours %.4f mm %.4f; transfer ours %.4f mm %.4f" % (len(sub), mean([c[2] for c in sub]), mean([c[3] for c in sub]), mean([c[9] for c in sub]), mean([c[10] for c in sub]), mean([c[7] for c in sub]), mean([c[8] for c in sub])))
