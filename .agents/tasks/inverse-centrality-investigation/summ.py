import json, os, glob, sys
R = "/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results"
runs = sorted(glob.glob(R + "/predict_mindboggle/*")) + sorted(glob.glob(R + "/eval_mindboggle/*"))
keys = ["template_dice", "template_dice_cortical", "template_dice_subcortical", "template_lncc",
        "fold_fwd_pct", "fold_inv_pct", "logjac_std_fwd", "inv_err_fwd_mm", "inv_err_inv_mm",
        "inv_err_fwd_max_mm", "inv_err_inv_max_mm", "disp_fwd_mean_mm"]
print("run | " + " | ".join(keys) + " | cen_mean cen_max mean_logjac_abs atlas_vox")
for r in runs:
    cands = [r + "/eval_template/summary.json", r + "/summary.json"]
    s = None
    for c in cands:
        if os.path.isfile(c):
            s = json.load(open(c)); break
    if s is None:
        print(os.path.basename(r), "no summary"); continue
    sm = s["subject_metrics"]
    vals = ["%.4f" % sm[k]["mean"] if k in sm and sm[k].get("n") else "-" for k in keys]
    g = s["group"]
    print(os.path.basename(r), "|", " | ".join(vals), "|", "%.3f %.3f %.4f %d" % (
        g["centrality_mean_mm"], g["centrality_max_mm"], g["mean_logjac_abs"], g["atlas_mask_voxels"]))
print()
# predict.json internal records
for r in sorted(glob.glob(R + "/predict_mindboggle/*")):
    for pj in glob.glob(r + "/fields/predict.json") + glob.glob(r + "/predict.json"):
        d = json.load(open(pj))
        print("==", os.path.basename(r), pj.replace(r, ""))
        print(" top keys:", sorted(d.keys())[:60])
        for k in ("stats", "checks", "summary", "max_abs_v", "internal_check", "records"):
            if k in d:
                txt = json.dumps(d[k])
                print(" ", k, txt[:1500])
