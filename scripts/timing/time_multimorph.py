"""Registration time of MultiMorph on one group (Mindboggle, paper Table 2), without disk I/O.

Replicates the timed part of MultiMorph's build_atlas (src/build_atlas_inference.py, run as
baselines/multimorph/run_multimorph.py): the group is read into memory with MultiMorph's own
dataset class before the clock starts, then one forward pass (forward + inverse fields), warp of
the images, atlas = mean, warp of the segmentations + atlas segmentation, and the copy of the
fields to the host (their script saves them from CPU tensors). Runs --reps times; rep 0 = cold.
Tries CUDA first; on out-of-memory it records the failure and, with --cpu-fallback, times the
same on the CPU (MultiMorph's code is written for CPU inference).

    python scripts/timing/time_multimorph.py --multimorph-dir <clone> --csv <mindboggle_data.csv> \
        --model <trained_models/model_cvpr.pt> --out timing.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("NEURITE_BACKEND", "pytorch")
os.environ.setdefault("VXM_BACKEND", "pytorch")


def gpu_name():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                       universal_newlines=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--multimorph-dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--img-header", default="img_path")
    ap.add_argument("--seg-header", default="segmentation_path")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--cpu-fallback", action="store_true")
    ap.add_argument("--fields-only", action="store_true",
                    help="time only the forward pass (fwd + inv full-resolution fields) and the copy of "
                         "the fields to the host; no image / segmentation warping, no atlas")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu"),
                    help="auto: CUDA first (then the CPU fallback); cpu: CPU only (one pass)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.multimorph_dir, "src"))
    import numpy as np
    import pandas as pd
    import torch
    import layers.group  # noqa: F401  (MultiMorph's src/layers package)
    import layers
    from build_atlas_inference import load_model, warp_segmentation
    from dataloader import SubGroupLoader3D

    csv = pd.read_csv(a.csv)
    t = time.time()
    dataset = SubGroupLoader3D(data=csv[a.img_header].tolist(), labels=None,
                               segmentations=csv[a.seg_header].tolist(), file_names=None,
                               segmentation_to_one_hot=False)
    img_size = dataset._get_img_size()
    sample = dataset[0]
    image_cpu = torch.as_tensor(sample["image"])[None]          # [1, N, 1, X, Y, Z] as the DataLoader gives
    seg_cpu = torch.as_tensor(sample["segmentation"])[None]
    t_read = time.time() - t
    n = int(image_cpu.shape[1])
    result = {"method": "multimorph", "n": n, "img_size": list(img_size), "gpu": gpu_name(),
              "torch": torch.__version__, "read_s": t_read, "fields_only": bool(a.fields_only),
              "definition": ("group in memory -> one forward pass (fwd + inv full-resolution fields) copied "
                             "to host; no warping, no atlas" if a.fields_only else
                             "group in memory -> one forward pass (fwd + inv fields), warped images, atlas, "
                             "warped segmentations + atlas_seg, fields copied to host") + "; model loaded once",
              "runs": []}

    def time_on(device, n_reps):
        t = time.time()
        model = load_model(a.model, img_size).to(device).eval()
        t_load = time.time() - t
        warp_layer = layers.group.Warp3d(img_size).to(device)
        reps = []
        for r in range(n_reps):
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with torch.no_grad():
                image = image_cpu.to(device)
                fwd, inv = model(image)
                if a.fields_only:
                    out = [x.cpu() for x in (fwd, inv)]
                    seg = warped = atlas = wseg = aseg = None
                else:
                    seg = seg_cpu.to(device)
                    warped = warp_layer(image, fwd)
                    atlas = torch.mean(warped, dim=1, keepdim=True)
                    wseg, aseg = warp_segmentation(segmentations=seg, warps=fwd, interpolation_mode="nearest")
                    out = [x.cpu() for x in (fwd, inv, warped, atlas, wseg, aseg)]
            if device.type == "cuda":
                torch.cuda.synchronize()
            total = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2.0 ** 30 if device.type == "cuda" else None
            reps.append({"total_s": total, "gpu_peak_gb": peak})
            print("multimorph %s rep %d: %d images, %.1f s" % (device, r, n, total), flush=True)
            del out, fwd, inv, warped, atlas, wseg, aseg, image, seg
        return {"device": str(device), "model_load_s": t_load, "threads": torch.get_num_threads(), "reps": reps}

    if a.device == "cpu":
        result["runs"].append(time_on(torch.device("cpu"), 1))
        with open(a.out, "w") as fh:
            json.dump(result, fh, indent=1)
        print("done -> %s" % a.out)
        return
    try:
        result["runs"].append(time_on(torch.device("cuda"), a.reps))
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        result["runs"].append({"device": "cuda", "error": "out of memory: %s" % str(e)[:300]})
        print("CUDA out of memory", flush=True)
        torch.cuda.empty_cache()
        if a.cpu_fallback:
            result["runs"].append(time_on(torch.device("cpu"), 1))   # one CPU pass (slow)
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print("done -> %s" % a.out)


if __name__ == "__main__":
    main()
