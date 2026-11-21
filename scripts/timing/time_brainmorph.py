"""Registration time of BrainMorph per group, without disk I/O (paper Tables 1 and 2).

Calls baselines/brainmorph/run_brainmorph.run (keypoints of every image, groupwise fit, fields,
moved images, warped labels, atlas + atlas_seg) with the images and labels read into memory before
the clock starts and every NIfTI write a no-op (small .npz / .json bookkeeping goes to $TMPDIR).
Network loaded once. Each group runs --reps times; rep 0 = cold, later reps = warm.
CUDA is synchronised before the clock stops.

    python scripts/timing/time_brainmorph.py --keymorph-dir <clone> --weights <pth> \
        --processed-dir <dir> --label-name synthseg --align rigid \
        --groups-dir <dir> --groups g1 g2 ... --out timing.json
    (one group: --subjects-file <file>; Mindboggle: --align tps_0)
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
import torch  # noqa: E402

from baselines.brainmorph import run_brainmorph as RB  # noqa: E402


def gpu_name():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                       universal_newlines=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keymorph-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--label-name", default="dkt31aseg")
    ap.add_argument("--align", required=True, help="rigid | affine | tps_<lambda>")
    ap.add_argument("--num-keypoints", type=int, default=512)
    ap.add_argument("--variant", default="L", choices=("S", "M", "L"))
    ap.add_argument("--num-iters", type=int, default=5)
    ap.add_argument("--pad-to", type=int, default=256)
    ap.add_argument("--tps-chunk", type=int, default=131072)
    ap.add_argument("--groups-dir", default=None)
    ap.add_argument("--groups", nargs="+", default=None)
    ap.add_argument("--subjects-file", default=None)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if bool(a.subjects_file) == bool(a.groups_dir):
        ap.error("give either --subjects-file or --groups-dir (+ --groups)")
    groups = ([(g, os.path.join(a.groups_dir, g + ".txt")) for g in a.groups] if a.groups_dir
              else [(os.path.splitext(os.path.basename(a.subjects_file))[0], a.subjects_file)])
    RB.import_keymorph(a.keymorph_dir)
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    t = time.time()
    levels = {"S": 4, "M": 5, "L": 6}[a.variant]
    kp_fn = RB.make_kp_fn(RB.build_network(a.weights, a.num_keypoints, levels, a.device), a.device)
    t_load = time.time() - t

    cache = {}
    real_load = RB.load_vol

    def cached_load(path):
        v = cache.get(path)
        return v if v is not None else real_load(path)
    RB.load_vol = cached_load
    RB.save_nii = lambda *x, **k: None
    tmp = tempfile.mkdtemp(prefix="time_bm_", dir=os.environ.get("TMPDIR"))
    result = {"method": "brainmorph", "align": a.align, "num_keypoints": a.num_keypoints, "variant": a.variant,
              "gpu": gpu_name(), "torch": torch.__version__, "model_load_s": t_load,
              "definition": "images in memory -> keypoints, groupwise fit, fields, moved images, warped "
                            "labels, atlas + atlas_seg; no NIfTI writes; network loaded once",
              "groups": []}
    geom = None
    for g, f in groups:
        subs = RB.find_subjects(a.processed_dir, a.label_name, f)
        cache.clear()
        for s in subs:
            cache[s["image"]] = real_load(s["image"])
            if s["labels"]:
                cache[s["labels"]] = real_load(s["labels"])
        if geom is None:
            img0, aff0 = cache[subs[0]["image"]]
            geom = RB.Geometry(img0.shape[:3], aff0, a.pad_to)
        reps = []
        for r in range(a.reps):
            out = tempfile.mkdtemp(prefix="g_", dir=tmp)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            RB.run(subs, out, [a.align], geom, kp_fn, a.device, a.num_iters, a.tps_chunk,
                   log=lambda *_: None, info={}, weighted=False, save_fields="matrices")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total = time.time() - t0
            with open(os.path.join(out, "run_info.json")) as fh:
                tim = json.load(fh)["timings"]
            reps.append({"total_s": total, "keypoints_s": tim["keypoints_s"], "fit_and_warp_s": tim[a.align + "_s"]})
            print("%s rep %d: %d images, total %.1f s (keypoints %.1f)" % (g, r, len(subs), total, tim["keypoints_s"]),
                  flush=True)
        result["groups"].append({"group": g, "n": len(subs), "reps": reps})
        with open(a.out, "w") as fh:
            json.dump(result, fh, indent=1)
    warm = [r["total_s"] for gr in result["groups"] for r in gr["reps"][1:]]
    print("done: %d groups, warm mean %.1f s -> %s" % (len(groups), float(np.mean(warm)) if warm else -1, a.out))


if __name__ == "__main__":
    main()
