"""Command-line interface: python -m preprocessing <command> [options]

Commands:
  check-tools        verify tools on PATH, FreeSurfer >= 7.3, ANTs/FSL versions, template files
  build-manifest     scan a dataset root -> input manifest CSV
  build-target-grid  estimate the LUMIR voxel -> MNI world mapping; write template_lumirgrid
  run                per-row pipeline (reorient/strip/n4/register/resample/normalize[/synthseg]/qc)
  synthseg           SynthSeg in batch mode over processed rows (one mri_synthseg call)
  qc-summary         per-dataset QC table with outlier flags, manifest_processed.csv;
                     optional on-demand qc.png (--png)
  export-npy         processed NIfTI -> per-level .npy (full, _2, _4): float32 image,
                     uint8 train18 labels; T1w only by default; chunkable (--start/--end)
  export-merge       merge the export-npy manifest parts into export_manifest.csv
  export-check       verify an export (dtypes, shapes, label range, world centers)
  cleanup            delete work/, mask.nii.gz and/or qc.png of completed rows (--dry-run first)
  n-chunks           number of SLURM array tasks for a manifest and chunk size

Storage defaults (config `outputs`): work/ is deleted when a row is done,
mask.nii.gz and qc.png are not written (image > 0 is used as the mask later).

On the Martinos cluster/glacier, run inside the tf10_nv.sif container through
the wrapper (same arguments; it sets the FreeSurfer/FSL/ANTs env and TMPDIR):
  preprocessing/run_in_container.sh check-tools --config cfg.yaml

Typical sequence (repo root as working dir, or via the wrapper from anywhere):
  python -m preprocessing check-tools --config cfg.yaml
  python -m preprocessing build-target-grid --config cfg.yaml \\
      --lumir-root /autofs/space/crater_001/datasets/public/LUMIR2025
  python -m preprocessing build-manifest --dataset lumir --root /autofs/space/crater_001/datasets/public/LUMIR2025 --out lumir.csv
  python -m preprocessing n-chunks --manifest lumir.csv --chunk-size 50
  python -m preprocessing run --config cfg.yaml --manifest lumir.csv --out-root /data/processed --rows 0 --dry-run
  python -m preprocessing run --config cfg.yaml --manifest lumir.csv --out-root /data/processed --start 0 --end 50
  python -m preprocessing synthseg --config cfg.yaml --manifest lumir.csv --out-root /data/processed --start 0 --end 50
  python -m preprocessing qc-summary --out-root /data/processed --dataset lumir [--config cfg.yaml --png flagged]
  python -m preprocessing export-npy --out-root /data/processed --export-root /data/npy_v2 --dataset lumir
  python -m preprocessing export-check --export-root /data/npy_v2
  python -m preprocessing cleanup --out-root /data/processed --dataset lumir --work --mask --qc-png --dry-run
"""
import argparse
import logging
import math
import os
import sys
from typing import List, Optional, Sequence, Tuple

from .manifest import ManifestRow, read_manifest, write_manifest

log = logging.getLogger("preprocessing")


def _rows_arg(text: str) -> List[int]:
    try:
        return [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("--rows must be comma-separated integers")


def _add_selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start", type=int, default=None, help="first row index (inclusive)")
    p.add_argument("--end", type=int, default=None, help="last row index (exclusive)")
    p.add_argument("--rows", type=_rows_arg, default=None, help="explicit row indices, e.g. 3,5,9")


def _add_work_flags(p: argparse.ArgumentParser, clean: bool) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--keep-work", action="store_true",
                   help="keep each row's work/ intermediates (debugging; default: config outputs.keep_work, "
                        "false = delete work/ once the row is fully done)")
    if clean:
        g.add_argument("--clean-work", action="store_true",
                       help="delete work/ once a row is completed even if the config sets outputs.keep_work")


def _clean_work(a) -> Optional[bool]:
    """True/False from --clean-work/--keep-work, None = config outputs.keep_work."""
    if getattr(a, "clean_work", False):
        return True
    if a.keep_work:
        return False
    return None


def select_rows(rows: List[ManifestRow], start: Optional[int], end: Optional[int],
                explicit: Optional[List[int]]) -> List[Tuple[int, ManifestRow]]:
    if explicit is not None:
        if start is not None or end is not None:
            raise SystemExit("use either --rows or --start/--end")
        bad = [i for i in explicit if i < 0 or i >= len(rows)]
        if bad:
            raise SystemExit("row indices out of range (0..%d): %s" % (len(rows) - 1, bad))
        return [(i, rows[i]) for i in explicit]
    s = 0 if start is None else max(0, start)
    e = len(rows) if end is None else min(len(rows), end)
    return [(i, rows[i]) for i in range(s, e)]


def build_parser() -> argparse.ArgumentParser:
    from .builders import BUILDERS
    from .pipeline import ALL_STEPS, DEFAULT_STEPS

    p = argparse.ArgumentParser(prog="python -m preprocessing", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--log-level", default=None, help="logging level (overrides -v)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("check-tools", help="verify external tools and template files")
    s.add_argument("--config", default=None)

    s = sub.add_parser("build-manifest", help="scan a dataset root into a manifest CSV",
                       conflict_handler="resolve", formatter_class=argparse.RawDescriptionHelpFormatter,
                       description="\n".join((m.__doc__ or "").strip() for m in BUILDERS.values()))
    s.add_argument("--dataset", required=True, choices=sorted(BUILDERS))
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--exclude", default=None, help="text file of subject IDs to drop (one per line)")
    for name, mod in BUILDERS.items():
        mod.add_arguments(s.add_argument_group(name))

    s = sub.add_parser("build-target-grid", help="LUMIR grid -> MNI world mapping and template_lumirgrid",
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=__import__("preprocessing.grid", fromlist=["x"]).__doc__)
    s.add_argument("--config", default=None)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--lumir-root", default=None, help="LUMIR root (images found with --image-glob)")
    g.add_argument("--lumir-images", nargs="+", default=None, help="explicit LUMIR image paths")
    s.add_argument("--image-glob", default="**/images*/LUMIRMRI_*_0000.nii.gz",
                   help="LUMIR images for the mean (default %(default)s: T1w channel, images* folders only; "
                        "label folders reuse the image names)")
    s.add_argument("--split", default="any", choices=("any", "train", "val", "test"))
    s.add_argument("--mni", default=None, help="MNI152 1 mm brain (default: config mni_templates.T1.image_1mm)")
    s.add_argument("--mni-mask", default=None, help="MNI brain mask (default: config mni_templates.T1.mask_1mm)")
    s.add_argument("--out-dir", default=None, help="output folder (default: config target_dir)")
    s.add_argument("--n", type=int, default=200, help="LUMIR images in the mean (default %(default)s)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--allow-permutations", action="store_true", help="search all 48 signed axis permutations")
    s.add_argument("--search-margin", type=int, default=10, help="extra offset range in voxels (default 10)")
    s.add_argument("--min-margin", type=float, default=0.01, help="required NCC margin over the runner-up")
    s.add_argument("--force-flip", choices=("no", "yes"), default=None,
                   help="use this x-flip state regardless of the margin, only after independent verification "
                        "with organizer labels that know the sides (LUMIR2025 SanityCheckLabelsTr: SLANT ids "
                        "47/48 = Right/Left hippocampus; Right-* structures must land on the subject's right "
                        "in the corrected affine); LUMIR: yes")
    s.add_argument("--no-subvoxel", action="store_true", help="skip the sub-voxel translation report")
    s.add_argument("--use-subvoxel", action="store_true", help="apply the sub-voxel shift (default: integer map)")

    s = sub.add_parser("run", help="run the per-row pipeline")
    s.add_argument("--config", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--out-root", required=True)
    _add_selection(s)
    s.add_argument("--steps", default=",".join(DEFAULT_STEPS),
                   help="comma list from %s (default %%(default)s)" % ",".join(ALL_STEPS))
    s.add_argument("--dry-run", action="store_true", help="print commands, execute nothing")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--threads", type=int, default=None)
    _add_work_flags(s, clean=True)
    s.add_argument("--fail-fast", action="store_true")

    s = sub.add_parser("synthseg", help="SynthSeg batch over processed rows")
    s.add_argument("--config", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--out-root", required=True)
    _add_selection(s)
    s.add_argument("--batch-dir", default=None)
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--threads", type=int, default=None)
    s.add_argument("--synthseg-cpu", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    _add_work_flags(s, clean=False)

    s = sub.add_parser("qc-summary", help="dataset QC summary and manifest_processed.csv",
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=__import__("preprocessing.qc", fromlist=["x"]).__doc__)
    s.add_argument("--out-root", required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--mad-threshold", type=float, default=4.0)
    s.add_argument("--png", default="none", metavar="MODE",
                   help="render qc.png on demand: none (default), flagged, all, or sample:N (needs --config)")
    s.add_argument("--png-seed", type=int, default=0, help="seed for --png sample:N (default 0)")
    s.add_argument("--config", default=None, help="config with target_dir (template for --png)")

    s = sub.add_parser("export-npy", help="export processed outputs to per-level .npy",
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=__import__("preprocessing.export", fromlist=["x"]).__doc__)
    s.add_argument("--out-root", required=True)
    s.add_argument("--export-root", required=True,
                   help="use a NEW root for this layout (e.g. $D/npy_v2), never an older export")
    s.add_argument("--dataset", action="append", default=None)
    s.add_argument("--start", type=int, default=None,
                   help="first row of manifest_processed.csv (inclusive; datasets concatenated)")
    s.add_argument("--end", type=int, default=None, help="last row of manifest_processed.csv (exclusive)")
    s.add_argument("--rows", type=_rows_arg, default=None, help="explicit manifest_processed.csv rows, e.g. 3,5,9")
    s.add_argument("--levels", type=_rows_arg, default=[1, 2, 4],
                   help="pyramid levels from 1,2,4 (default 1,2,4: full, _2, _4)")
    s.add_argument("--label-sets", default="train18",
                   help="comma list: train18 (default; all levels), synthseg (103-id remap) and provided "
                        "label names such as lumir10 (full resolution only); a missing set fails the export")
    s.add_argument("--optional-label-sets", default="", metavar="NAME[,NAME]",
                   help="comma list of evaluation-only provided label sets, added to --label-sets; a case "
                        "without one skips it and lists it in the manifest column missing_optional_labels "
                        "(never train18). Validation: --optional-label-sets lumir10")
    s.add_argument("--contrast", action="append", default=None,
                   help="contrast filter, repeatable or comma list (default T1w: drops LUMIR's non-T1 "
                        "unknown0001 val cases); 'any' exports every contrast")
    s.add_argument("--split", action="append", default=None,
                   help="split filter, repeatable or comma list (e.g. train or val; default any)")
    s.add_argument("--image-dtype", choices=("float32", "float16"), default="float32",
                   help="default float32: augmentation runs in float32 in DALI; the cast to float16 for "
                        "mixed precision happens at the model input")
    s.add_argument("--label-dtype", choices=("auto", "uint16", "uint8"), default="auto",
                   help="auto (default): uint8 if the fixed LUT fits, else uint16; uint16 without remapping")
    s.add_argument("--no-remap-labels", action="store_true",
                   help="keep original label ids (uint16; not with train18)")
    s.add_argument("--include-mask", action="store_true",
                   help="also write mask.npy, full resolution (mask.nii.gz if saved, else image > 0)")
    s.add_argument("--include-template", action="store_true",
                   help="also export the per-level template (needs --config)")
    s.add_argument("--config", default=None)
    s.add_argument("--overwrite", action="store_true", help="rewrite every file, even complete cases")

    s = sub.add_parser("export-merge", help="merge export-npy manifest parts into export_manifest.csv")
    s.add_argument("--export-root", required=True)

    s = sub.add_parser("export-check", help="verify an export: dtypes, shapes, label range, world centers")
    s.add_argument("--export-root", required=True)
    s.add_argument("--sample", type=int, default=20, help="random cases to check (default 20; 0 = all)")
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("cleanup", help="delete work/, mask.nii.gz, qc.png of completed rows",
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=__import__("preprocessing.cleanup", fromlist=["x"]).__doc__)
    s.add_argument("--out-root", required=True)
    s.add_argument("--dataset", action="append", required=True, help="dataset folder name (repeatable)")
    s.add_argument("--work", action="store_true", help="delete <subject>/work/")
    s.add_argument("--mask", action="store_true", help="delete <subject>/mask.nii.gz")
    s.add_argument("--qc-png", action="store_true", help="delete <subject>/qc.png")
    s.add_argument("--dry-run", action="store_true", help="only print what would be deleted")

    s = sub.add_parser("n-chunks", help="number of array tasks: ceil(rows / chunk-size)")
    s.add_argument("--manifest", required=True)
    s.add_argument("--chunk-size", type=int, required=True)
    return p


def _env(cfg):
    from .tools import resolve_environment
    return resolve_environment(cfg["env_setup"])


def cmd_check_tools(a) -> int:
    from .config import load_config
    from .tools import check_tools
    cfg = load_config(a.config)
    ok, lines = check_tools(cfg, _env(cfg))
    print("\n".join(lines))
    print("ALL OK" if ok else "PROBLEMS FOUND (see above)")
    return 0 if ok else 1


def cmd_build_manifest(a) -> int:
    from .builders import BUILDERS, apply_exclude
    rows = BUILDERS[a.dataset].build(os.path.abspath(a.root), a)
    rows = apply_exclude(rows, a.exclude)
    if not rows:
        log.error("no rows found under %s", a.root)
        return 1
    write_manifest(rows, a.out)
    log.info("wrote %d rows to %s", len(rows), a.out)
    return 0


def cmd_build_target_grid(a) -> int:
    from .builders.lumir import find_images, split_of
    from .config import ConfigError, expand_path, load_config
    from .grid import GridAmbiguityError, build_target_grid
    from pathlib import Path
    cfg = load_config(a.config)
    env = _env(cfg)
    if a.lumir_images:
        paths = [os.path.abspath(x) for x in a.lumir_images]
    else:
        paths = [str(x) for x in find_images(a.lumir_root, a.image_glob)
                 if a.split == "any" or split_of(Path(x)) == a.split]
    mni = a.mni or expand_path(cfg["mni_templates"]["T1"]["image_1mm"], env)
    mni_mask = a.mni_mask
    if mni_mask is None and not a.mni:  # config mask only goes with the config image
        mni_mask = expand_path(cfg["mni_templates"]["T1"].get("mask_1mm"), env)
    if mni_mask and not os.path.isfile(mni_mask):
        log.warning("MNI mask %s not found; deriving the template mask from the image", mni_mask)
        mni_mask = None
    out_dir = a.out_dir or expand_path(cfg.get("target_dir"), env)
    if not out_dir:
        raise ConfigError("--out-dir or target_dir in the config is required")
    log.info("build-target-grid: %d LUMIR images, MNI %s", len(paths), mni)
    try:
        info = build_target_grid(paths, mni, out_dir, cfg, mni_mask, a.n, a.seed, a.allow_permutations,
                                 a.search_margin, a.min_margin, a.force_flip, not a.no_subvoxel, a.use_subvoxel)
    except GridAmbiguityError as e:
        log.error("%s", e)
        return 2
    for r in info["candidates"]:
        print("candidate %-8s offset=%s ncc=%.5f symmetric_crop_pad=%s subvoxel=%s" % (
            r["name"], r["offset"], r["ncc"], r.get("symmetric_crop_pad"), r.get("subvoxel_shift")))
    print("chosen %s, NCC margin %.5f, header NCC %.5f" % (info["chosen"], info["ncc_margin"], info["header_ncc"]))
    if info["ncc_margin"] < a.min_margin:
        print("NOTE: the flip margin is below --min-margin, so left/right is NOT determined by the images. "
              "Make sure --force-flip was verified with organizer labels that know the sides (LUMIR2025 "
              "SanityCheckLabelsTr: SLANT ids 47/48 = Right/Left hippocampus; Right-* structures must land "
              "on the subject's right in the corrected affine). LUMIR: --force-flip yes.")
    print("corrected affine:\n%s" % "\n".join("  %s" % row for row in info["affine"]))
    print("Set in your config:\n  target_dir: %s" % os.path.abspath(out_dir))
    print("Verify: freeview %s %s:colormap=heat:opacity=0.4" % (info["outputs"]["lumir_mean"],
                                                               info["outputs"]["template"]))
    return 0


def _load_rows(a):
    rows = read_manifest(a.manifest)
    return select_rows(rows, a.start, a.end, a.rows)


def cmd_run(a) -> int:
    from .config import load_config
    from .grid import Target
    from .pipeline import ALL_STEPS, run_rows
    from .tools import ToolRunner, resolve_tmp_dir
    cfg = load_config(a.config, threads=a.threads)
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        raise SystemExit("unknown steps %s (choose from %s)" % (bad, ALL_STEPS))
    sel = _load_rows(a)
    if not sel:
        log.info("no rows selected; nothing to do")
        return 0
    env = _env(cfg)
    runner = ToolRunner(env, cfg["threads"], dry_run=a.dry_run, tmp_dir=resolve_tmp_dir(cfg, a.out_root, env))
    target = Target(cfg, env)
    return run_rows(sel, cfg, a.out_root, steps, runner, target, a.overwrite, _clean_work(a), a.fail_fast)


def cmd_synthseg(a) -> int:
    from .config import load_config
    from .grid import Target
    from .manifest import subject_dir_name
    from .synthseg import run_synthseg_batch
    from .tools import ToolRunner, resolve_tmp_dir
    cfg = load_config(a.config, threads=a.threads, synthseg_cpu=a.synthseg_cpu)
    sel = _load_rows(a)
    if not sel:
        log.info("no rows selected; nothing to do")
        return 0
    env = _env(cfg)
    runner = ToolRunner(env, cfg["threads"], dry_run=a.dry_run, tmp_dir=resolve_tmp_dir(cfg, a.out_root, env))
    target = Target(cfg, env)
    out_root = os.path.abspath(a.out_root)
    dirs = [os.path.join(out_root, r.dataset, subject_dir_name(r)) for _, r in sel]
    stem = os.path.splitext(os.path.basename(a.manifest))[0]
    batch = a.batch_dir or os.path.join(out_root, "_synthseg_batches", "%s_%d_%d" % (stem, sel[0][0], sel[-1][0] + 1))
    counts = run_synthseg_batch(dirs, cfg, runner, batch, target, a.overwrite,
                                clean_work=False if a.keep_work else None)
    log.info("synthseg: %s", counts)
    return 1 if counts.get("failed") else 0


def cmd_qc_summary(a) -> int:
    from .qc import parse_png_mode, render_qc_png, select_png_dirs, summarize
    try:
        png_mode, png_n = parse_png_mode(a.png)
    except ValueError as e:
        raise SystemExit(str(e))
    if png_mode != "none" and not a.config:
        raise SystemExit("--png needs --config (the template is drawn in the second row)")
    d = os.path.join(a.out_root, a.dataset)
    if not os.path.isdir(d):
        log.error("no such dataset folder: %s", d)
        return 1
    n, flagged = summarize(d, a.mad_threshold)
    print("%d subjects, %d flagged -> %s" % (n, flagged, os.path.join(d, "qc_summary.csv")))
    if png_mode != "none":
        from .config import load_config
        from .grid import Target
        cfg = load_config(a.config)
        template = Target(cfg, _env(cfg)).data()
        dirs = select_png_dirs(d, png_mode, png_n, a.png_seed)
        for sd in dirs:
            render_qc_png(sd, template)
        print("rendered %d qc.png (%s)" % (len(dirs), a.png))
    return 0


def cmd_export_npy(a) -> int:
    from .export import export_npy
    tp = None
    if a.include_template:
        if not a.config:
            raise SystemExit("--include-template needs --config")
        from .config import load_config, target_paths
        cfg = load_config(a.config)
        tp = target_paths(cfg, _env(cfg))
    contrasts = a.contrast if a.contrast is not None else ["T1w"]
    try:
        n = export_npy(a.out_root, a.export_root, a.dataset, a.image_dtype, a.label_dtype, tp, a.overwrite,
                       remap_labels=not a.no_remap_labels, include_mask=a.include_mask, levels=a.levels,
                       label_sets=[a.label_sets], contrasts=contrasts, splits=a.split,
                       start=a.start, end=a.end, rows=a.rows, optional_label_sets=[a.optional_label_sets])
    except ValueError as e:
        log.error("%s", e)
        return 1
    print("exported %d subjects" % n)
    return 0


def cmd_export_merge(a) -> int:
    from .export import merge_parts
    try:
        n = merge_parts(a.export_root)
    except ValueError as e:
        log.error("%s", e)
        return 1
    print("export_manifest.csv: %d cases" % n)
    return 0


def cmd_export_check(a) -> int:
    from .export import export_check
    n, problems, report = export_check(a.export_root, a.sample, a.seed)
    print("\n".join(report))
    for p in problems[:50]:
        print("PROBLEM " + p)
    print("checked %d cases: %s" % (n, "OK" if not problems else "%d PROBLEMS" % len(problems)))
    return 0 if not problems else 1


def cmd_cleanup(a) -> int:
    from .cleanup import cleanup
    try:
        res = cleanup(a.out_root, a.dataset, a.work, a.mask, a.qc_png, a.dry_run)
    except ValueError as e:
        log.error("%s", e)
        return 1
    return 0 if not res["refused"] else 3


def cmd_n_chunks(a) -> int:
    if a.chunk_size <= 0:
        raise SystemExit("--chunk-size must be > 0")
    n = len(read_manifest(a.manifest))
    print(int(math.ceil(n / float(a.chunk_size))))
    return 0


class _StderrHandler(logging.StreamHandler):
    """StreamHandler that always writes to the current sys.stderr."""

    @property  # type: ignore[override]
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value):
        pass


COMMANDS = {"check-tools": cmd_check_tools, "build-manifest": cmd_build_manifest,
            "build-target-grid": cmd_build_target_grid, "run": cmd_run, "synthseg": cmd_synthseg,
            "qc-summary": cmd_qc_summary, "export-npy": cmd_export_npy, "export-merge": cmd_export_merge,
            "export-check": cmd_export_check, "cleanup": cmd_cleanup, "n-chunks": cmd_n_chunks}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    a = parser.parse_args(argv)
    level = a.log_level or ("DEBUG" if a.verbose else "INFO")
    root = logging.getLogger("preprocessing")
    if not any(isinstance(x, _StderrHandler) for x in root.handlers):
        h = _StderrHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(h)
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    return int(COMMANDS[a.command](a))
