"""Registration time of SynthMorph (groupwise wrapper) per group, without disk I/O (paper Table 1).

Calls baselines/synthmorph/run_synthmorph_groupwise.run_align (network-space resampling and
keypoints of every image, groupwise fit, matrices, moved images, warped labels, atlas + atlas_seg)
with the input volumes read into memory before the clock starts, every NIfTI write a no-op and the
pair check off (it is a verification step, not part of the method). Detector loaded once per
alignment. Each group runs --reps times; rep 0 = cold, later reps = warm. Run with FreeSurfer's
fspython (as slurm/synthmorph_miriad.sh).

    fspython scripts/timing/time_synthmorph.py --processed-dir <dir> --label-name synthseg \
        --align rigid --groups-dir <dir> --groups g1 g2 ... --out timing.json
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from baselines.synthmorph import run_synthmorph_groupwise as SMG  # noqa: E402


def gpu_name():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                       universal_newlines=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--label-name", default="dkt31aseg")
    ap.add_argument("--align", required=True, choices=("rigid", "affine"))
    ap.add_argument("--extent", type=int, default=256)
    ap.add_argument("--num-iters", type=int, default=5)
    ap.add_argument("--unweighted", action="store_true")
    ap.add_argument("--groups-dir", required=True)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    SMG.import_freesurfer()
    tf, sf = SMG.tf, SMG.sf
    t = time.time()
    det = SMG.Detector(a.align, a.extent)
    t_load = time.time() - t

    vol_cache, arr_cache = {}, {}
    real_sf_load, real_load = sf.load_volume, SMG.load_vol

    def sf_load(path, *x, **k):
        v = vol_cache.get(path)
        return v if v is not None else real_sf_load(path, *x, **k)

    def arr_load(path):
        v = arr_cache.get(path)
        return v if v is not None else real_load(path)
    sf.load_volume = sf_load
    SMG.load_vol = arr_load
    SMG.save_nii = lambda *x, **k: None
    tmp = tempfile.mkdtemp(prefix="time_sm_", dir=os.environ.get("TMPDIR"))
    result = {"method": "synthmorph", "align": a.align, "extent": a.extent, "weighted": not a.unweighted,
              "gpu": gpu_name(), "tensorflow": tf.__version__, "model_load_s": t_load,
              "definition": "images in memory -> network-space resampling, keypoints, groupwise fit, "
                            "matrices, moved images, warped labels, atlas + atlas_seg; no NIfTI writes; "
                            "pair check off; detector loaded once",
              "groups": []}
    for g in a.groups:
        subs = SMG.find_subjects(a.processed_dir, a.label_name, os.path.join(a.groups_dir, g + ".txt"))
        vol_cache.clear()
        arr_cache.clear()
        for s in subs:
            vol_cache[s["image"]] = real_sf_load(s["image"])
            arr_cache[s["image"]] = real_load(s["image"])
            if s["labels"]:
                arr_cache[s["labels"]] = real_load(s["labels"])
        reps = []
        for r in range(a.reps):
            d = tempfile.mkdtemp(prefix="g_", dir=tmp)
            t0 = time.time()
            meta = SMG.run_align(subs, tmp, a.align, a.extent, a.num_iters, not a.unweighted, log=lambda *_: None,
                                 info={}, det=det, d=d, kp_path=os.path.join(d, "keypoints.npz"), dense=False,
                                 do_pair_check=False)
            total = time.time() - t0
            reps.append({"total_s": total, "keypoints_s": meta["keypoints_s"], "wrapper_total_s": meta["total_s"]})
            print("%s rep %d: %d images, total %.1f s (keypoints %.1f)" % (g, r, len(subs), total, meta["keypoints_s"]),
                  flush=True)
        result["groups"].append({"group": g, "n": len(subs), "reps": reps})
        with open(a.out, "w") as fh:
            json.dump(result, fh, indent=1)
    warm = [r["total_s"] for gr in result["groups"] for r in gr["reps"][1:]]
    print("done: %d groups, warm mean %.1f s -> %s" % (len(a.groups), float(np.mean(warm)) if warm else -1, a.out))


if __name__ == "__main__":
    main()
