"""Registration time of our model per group, without disk I/O (paper Tables 1 and 2).

Runs the same code path as ``predict.py run`` (StreamedRunner + the run's sink: canonical
full-resolution fwd / inv fields, moved images, warped labels, atlas and atlas segmentation; with
--recenter K also the re-centering rounds), but with every NIfTI write replaced by a no-op and the
images read before the clock starts. The model is loaded once. Each group is registered --reps
times in a row; rep 0 is the cold run (includes TF graph tracing for a new group size), the later
reps are warm. The internal consistency check is off.

    python scripts/timing/time_ours.py --checkpoint <run>/ckpt/last --processed-dir <dir> \
        --label-name synthseg --grid-reference <json> --branches rigid \
        --groups-dir <dir with <group>.txt> --groups g1 g2 ... --out timing.json [--tta flip8]
    (one group: --subjects-file <file> instead of --groups-dir / --groups)

Writes one JSON with per-group times; GPU name and repo commit included.
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


def gpu_name():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                       universal_newlines=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        return None


def git_head():
    try:
        return subprocess.check_output(["git", "-C", REPO, "log", "--oneline", "-1"],
                                       universal_newlines=True).strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--label-name", default="dkt31aseg")
    ap.add_argument("--grid-reference", default=None)
    ap.add_argument("--branches", required=True, choices=("rigid", "affine", "deform", "affine+deform"))
    ap.add_argument("--tta", default="none", choices=("none", "flip8"))
    ap.add_argument("--recenter", type=int, default=0)
    ap.add_argument("--groups-dir", default=None)
    ap.add_argument("--groups", nargs="+", default=None)
    ap.add_argument("--subjects-file", default=None)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--label", default=None, help="method label stored in the JSON")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if bool(a.subjects_file) == bool(a.groups_dir):
        ap.error("give either --subjects-file or --groups-dir (+ --groups)")
    if a.groups_dir:
        groups = [(g, os.path.join(a.groups_dir, g + ".txt")) for g in a.groups]
    else:
        groups = [(os.path.splitext(os.path.basename(a.subjects_file))[0], a.subjects_file)]

    from inference import cli
    from inference import predict as PR
    tmp_out = tempfile.mkdtemp(prefix="time_ours_", dir=os.environ.get("TMPDIR"))
    argv = ["run", "--checkpoint", a.checkpoint, "--processed-dir", a.processed_dir,
            "--label-name", a.label_name, "--branches", a.branches, "--out", tmp_out, "--overwrite",
            "--tta", a.tta, "--check-internal", "0", "--subjects-file", groups[0][1]]
    if a.grid_reference:
        argv += ["--grid-reference", a.grid_reference]
    if a.recenter:
        argv += ["--recenter", str(a.recenter)]
    args = cli.parse_args(argv)
    PR.setup_device(args)
    import numpy as np
    import tensorflow as tf
    from inference import canonical as CN
    from inference import runner as RN
    from inference import store as SToR

    CN.write_nifti = lambda *x, **k: None      # no disk writes (fields, moved images, atlas)
    rec = PR._new_record(args, True)
    peek = PR._peek_config(args.checkpoint)
    plan = PR._setup_levels(args, rec, peek[0])
    tta_cfg = PR._setup_tta(args, rec, plan, peek[0])
    t = time.time()
    loaded = PR._load(args)
    t_load = time.time() - t

    # one store for the whole process, sized for the largest group (as predict.py chooses it)
    sizes = []
    for _, f in groups:
        with open(f) as fh:
            sizes.append(sum(1 for ln in fh if ln.strip()))
    tdt = "float16" if loaded.precision["trainable"] == "mixed_float16" else "float32"
    if tta_cfg is not None and RN.branch_tasks(args.branches)[1] is not None:
        est = SToR.estimate_store_gb(loaded.level, loaded.model_cfg, max(sizes), args.branches, tdt,
                                     tta_levels=tta_cfg.levels)
    else:
        est = SToR.estimate_store_gb(loaded.level, loaded.model_cfg, max(sizes), args.branches, tdt)
    store = SToR.choose_store(args.store, est, os.environ.get("TMPDIR") or tmp_out)
    runner = RN.StreamedRunner(loaded, store, chunks={"l1": args.chunk_l1, "l2": args.chunk_l2,
                                                      "l4": args.chunk_l4}, plan=plan, tta=tta_cfg)
    result = {"method": a.label or "ours", "branches": a.branches, "tta": a.tta, "recenter": a.recenter,
              "checkpoint": a.checkpoint, "gpu": gpu_name(), "repo": git_head(), "model_load_s": t_load,
              "store": {"kind": store.kind, "estimate_gb": est},
              "definition": "images in memory -> full-res fwd/inv fields, moved images, warped labels, "
                            "atlas + atlas_seg (+ re-centering); no disk writes; model loaded once",
              "groups": []}
    for g, f in groups:
        args.subjects_file = f
        subs, images, labels, shape, aff, spacing = PR._read_subjects(args, rec)
        reps = []
        for r in range(a.reps):
            rc = None
            rc_root = None
            if a.recenter:
                from inference import recenter as RC
                rc_root = tempfile.mkdtemp(prefix="rc_", dir=os.environ.get("TMPDIR") or tmp_out)
                rc = RC.Recenterer(a.recenter, len(subs), shape, spacing, rc_root)
            writer = PR._Writer(args, subs, images, labels, aff, shape, spacing, 0, recenter=rc)
            t0 = time.time()
            summary = runner.run(images, args.branches, writer.sink)
            t_model = time.time() - t0
            t1 = time.time()
            if rc is not None:
                rc.run(writer.emit_corrected, images)
            t_rc = time.time() - t1
            t2 = time.time()
            writer.atlas.write(tmp_out, aff)        # computes atlas mean + majority vote (write no-op)
            t_atlas = time.time() - t2
            total = time.time() - t0
            if rc is not None:
                rc.close()
            nonfinite = int(sum(sum(c.values()) for c in writer.nonfinite.values()))
            reps.append({"total_s": total, "model_and_sink_s": t_model, "recenter_s": t_rc,
                         "atlas_s": t_atlas, "sink_cpu_s": writer.t_write,
                         "levels_s": {k: float(v) for k, v in summary["timing_s"].items()},
                         "nonfinite": nonfinite})
            print("%s rep %d: %d images, total %.1f s (model+sink %.1f, recenter %.1f, atlas %.1f)"
                  % (g, r, len(subs), total, t_model, t_rc, t_atlas), flush=True)
        result["groups"].append({"group": g, "n": len(subs), "reps": reps})
        del images, labels
        with open(a.out, "w") as fh:
            json.dump(result, fh, indent=1)
    store.close()
    warm = [r["total_s"] for gr in result["groups"] for r in gr["reps"][1:]]
    print("done: %d groups, warm mean %.1f s -> %s" % (len(groups), float(np.mean(warm)) if warm else -1, a.out))


if __name__ == "__main__":
    main()
