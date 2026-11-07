"""Command line of predict.py (predict-script design 7): argparse, validation, chunk defaults.

TF-free: `parse_args` (parse + `validate`) runs before TensorFlow is imported, so every input error
is an argparse error (exit 2) without a TF start-up. Exit codes (design 7): 0 ok; 2 invalid input /
config / checkpoint / no GPU with --device gpu; 3 non-finite outputs; 4 check-streaming gate
failed; 5 internal consistency check failed; 1 unexpected error.
"""
import argparse
import glob
import logging
import os
import sys

logger = logging.getLogger(__name__)

BRANCH_VALUES = ("rigid", "affine", "deform", "affine+deform")
BRANCH_NOTE = ("rigid and affine are alternatives; the deform branch was trained on affine-aligned "
               "inputs")
LEVEL_KEYS = ("l1", "l2", "l4")
RUN_CHUNK_DEFAULTS = {"l1": 2, "l2": 8, "l4": 32}
# File names `run` writes into --out (design 5.2, 8).
OUTPUT_NAMES = ("predict.json", "atlas.nii.gz", "atlas_seg.nii.gz", "tta_spread.csv")
TTA_MODES = ("none", "flip8")   # inference.tta.MODES (kept TF-free and import-free here)
OUTPUT_PATTERNS = ("*_fwd.nii.gz", "*_inv.nii.gz", "*_moved.nii.gz")
# File names `linear-eval` writes into --out (TTA design 7.3, plan D9).
LINEAR_EVAL_OUTPUTS = ("per_subject.csv", "per_pair.csv", "summary.json", "tta_spread.csv")
LINEAR_TASKS = ("rigid", "affine")

EXIT_OK, EXIT_ERROR, EXIT_INPUT, EXIT_NONFINITE, EXIT_GATE, EXIT_INTERNAL = 0, 1, 2, 3, 4, 5


def _branches(value):
    if value not in BRANCH_VALUES:
        raise argparse.ArgumentTypeError(
            "invalid value %r; supported values: %s (%s)" % (value, ", ".join(BRANCH_VALUES), BRANCH_NOTE))
    return value


def _int_ge(lo, hi=None):
    def conv(value):
        try:
            v = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("%r is not an integer" % value)
        if v < lo or (hi is not None and v > hi):
            rng = ">= %d" % lo if hi is None else "in %d-%d" % (lo, hi)
            raise argparse.ArgumentTypeError("%r must be an integer %s" % (value, rng))
        return v
    return conv


def _positive_float(value):
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("%r is not a number" % value)
    if not v > 0 or v != v or v == float("inf"):
        raise argparse.ArgumentTypeError("%r must be a finite float > 0" % value)
    return v


def _levels(value):
    from inference import levels as LV
    try:
        return LV.parse_levels(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def _common(p, default_branches, chunk_defaults, branches=True):
    p.add_argument("--checkpoint", required=True,
                   help="checkpoint prefix or checkpoint directory (e.g. <run>/ckpt/best_deform)")
    p.add_argument("--processed-dir", required=True,
                   help="processed subjects: <subject_dir>/image.nii.gz + labels_<label-name>.nii.gz")
    p.add_argument("--subjects-file", default=None, help="optional subject list (names or <subject>_T1w dirs)")
    p.add_argument("--label-name", default="dkt31aseg", help="labels_<name>.nii.gz (default %(default)s)")
    if branches:
        p.add_argument("--branches", type=_branches, default=default_branches,
                       metavar="{%s}" % ",".join(BRANCH_VALUES),
                       help="rigid | affine: linear cascade on the raw images; deform: deform cascade "
                            "on the raw images, which must already be MNI-affine-aligned (as "
                            "processed Mindboggle is; not checked); affine+deform: affine cascade, "
                            "images resampled into the affine space, deform cascade, one composed "
                            "field (%s; default %%(default)s)" % BRANCH_NOTE)
    p.add_argument("--levels", type=_levels, default=None, metavar="LEVELS",
                   help="cascade levels to run, coarse to fine, comma-separated (e.g. 4 | 4,2 | 2,1 | "
                        "1; default: all levels of the checkpoint). Levels finer than the last one "
                        "are not run (its fields are upsampled to level 1); a coarser level that is "
                        "left out is skipped as in training (identity; later stages on the "
                        "un-pre-warped inputs). affine+deform uses the same levels for both cascades")
    p.add_argument("--allow-untrained-skip", action="store_true",
                   help="run a selection that skips a level the checkpoint chain was never trained "
                        "to skip (skip probability 0 or unknown); refused (exit 2) otherwise")
    p.add_argument("--device", choices=("gpu", "cpu"), default="gpu",
                   help="gpu (default) or cpu (hides the GPUs before TF starts)")
    p.add_argument("--precision", choices=("checkpoint", "float32"), default="checkpoint",
                   help="trainable stage in the checkpoint's policy (default) or float32")
    for key in LEVEL_KEYS:
        d = chunk_defaults.get(key)
        p.add_argument("--chunk-" + key, dest="chunk_" + key, type=_int_ge(0), default=d,
                       help="images per chunk at level %s (0 = whole group; default %s)"
                            % (key[1:], d if d is not None else "min(cap, max(1, n_subjects - 1))"))
    p.add_argument("--grid-reference", default=None,
                   help="target_grid.json or a NIfTI on the training grid: the processed affine must "
                        "equal its affine within 1e-3")
    p.add_argument("--gpu-memory-limit-mb", type=_int_ge(1), default=None,
                   help="fixed TF memory limit on GPU 0 (default: memory growth)")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging, list TF devices")


def _tta_flags(p):
    """--tta / --tta-levels (TTA design 7.1, milestone 10)."""
    p.add_argument("--tta", choices=TTA_MODES, default="none",
                   help="test-time augmentation: none (default) or flip8 = the 8 group flips "
                        "(x, y, z reversals of the whole group), outputs mapped back and averaged "
                        "per level (linear: gauge-free residuals, deform: raw SVFs; milestone 10). "
                        "About 8x the model time; streamed mode only")
    p.add_argument("--tta-levels", type=_levels, default=None, metavar="LEVELS",
                   help="levels with TTA, comma-separated (e.g. 1 or 2,1; default: all levels the "
                        "plan runs); needs --tta flip8")


def _validate_tta(args, parser):
    tta = getattr(args, "tta", "none")
    if getattr(args, "tta_levels", None) is not None and tta != "flip8":
        parser.error("--tta-levels needs --tta flip8")
    if tta == "flip8" and getattr(args, "mode", "streamed") == "unchunked":
        parser.error("--tta flip8 needs --mode streamed (the unchunked runner has no TTA)")


def build_parser():
    p = argparse.ArgumentParser(prog="python predict.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="predict fields, moved images, atlas and atlas segmentation")
    _common(r, "affine+deform", RUN_CHUNK_DEFAULTS)
    r.add_argument("--out", required=True, help="output directory")
    r.add_argument("--mode", choices=("streamed", "unchunked"), default="streamed",
                   help="streamed (default; exact group streaming) or unchunked (one call per level)")
    r.add_argument("--store", choices=("auto", "ram", "disk"), default="auto",
                   help="host store of the streamed intermediates (default auto)")
    r.add_argument("--store-dir", default=None, help="disk store directory (default $TMPDIR, else <out>/_stream_tmp)")
    r.add_argument("--check-internal", type=_int_ge(0), default=2,
                   help="internal consistency check of the first K subjects (0 = off; default %(default)s)")
    r.add_argument("--compress-level", type=_int_ge(0, 9), default=1, help="gzip level 0-9 (default %(default)s)")
    _tta_flags(r)
    r.add_argument("--overwrite", action="store_true", help="allow an --out directory with earlier outputs")
    e = sub.add_parser("linear-eval",
                       help="rigid / affine errors against seeded synthetic misalignments (design 7.3)")
    _common(e, None, RUN_CHUNK_DEFAULTS, branches=False)
    e.add_argument("--task", choices=LINEAR_TASKS, required=True,
                   help="linear cascade to evaluate (the rendered misalignments follow the training "
                        "validation ranges of this task)")
    e.add_argument("--seed", type=_int_ge(0), default=0,
                   help="seed of the synthetic transforms (same seed = same rendered group in both "
                        "TTA arms; default %(default)s)")
    _tta_flags(e)
    e.add_argument("--store", choices=("auto", "ram", "disk"), default="auto",
                   help="host store of the streamed intermediates (default auto)")
    e.add_argument("--store-dir", default=None, help="disk store directory (default $TMPDIR, else <out>/_stream_tmp)")
    e.add_argument("--out", required=True,
                   help="output directory (%s)" % ", ".join(LINEAR_EVAL_OUTPUTS))
    e.add_argument("--overwrite", action="store_true", help="allow an --out directory with earlier outputs")
    c = sub.add_parser("check-streaming", help="streamed vs unchunked on the first --n-subjects subjects")
    _common(c, "deform", {})
    c.add_argument("--out", required=True, help="JSON report path")
    c.add_argument("--n-subjects", type=_int_ge(2), default=4, help="group size (default %(default)s)")
    c.add_argument("--tol-field-max-mm", type=_positive_float, default=None)
    c.add_argument("--tol-field-mean-mm", type=_positive_float, default=None)
    c.add_argument("--tol-moved", type=_positive_float, default=None)
    c.add_argument("--tol-matrix", type=_positive_float, default=None)
    c.add_argument("--tol-translation-mm", type=_positive_float, default=None)
    return p


def _creatable_dir(path):
    """True if ``path`` is a directory or could be created (nearest existing ancestor writable)."""
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            return False
        p = parent
    return os.path.isdir(p) and os.access(p, os.W_OK)


def _existing_outputs(out, names=OUTPUT_NAMES, patterns=OUTPUT_PATTERNS):
    found = [n for n in names if os.path.exists(os.path.join(out, n))]
    for pat in patterns:
        found += [os.path.basename(f) for f in glob.glob(os.path.join(out, pat))]
    return sorted(found)


def _validate_out_dir(args, parser, names, patterns):
    """--out directory checks of `run` and `linear-eval` (overwrite protection over ``names`` /
    ``patterns``), --store-dir, then the TTA flag combination."""
    out = args.out
    if os.path.exists(out) and not os.path.isdir(out):
        parser.error("--out %s exists and is not a directory" % out)
    if os.path.isdir(out) and not args.overwrite:
        found = _existing_outputs(out, names, patterns)
        if found:
            parser.error("--out %s already holds outputs (%s%s); use --overwrite"
                         % (out, ", ".join(found[:4]), ", ..." if len(found) > 4 else ""))
    if not _creatable_dir(out):
        parser.error("--out %s cannot be created" % out)
    if args.store_dir is not None and not _creatable_dir(args.store_dir):
        parser.error("--store-dir %s cannot be created" % args.store_dir)
    _validate_tta(args, parser)


def validate(args, parser):
    """Design 7 validation before TF (parser.error -> exit 2). Returns args."""
    ck = args.checkpoint
    if not (os.path.isdir(ck) or os.path.exists(ck + ".index")):
        parser.error("--checkpoint %s is neither a checkpoint directory nor a prefix with %s.index" % (ck, ck))
    if not os.path.isdir(args.processed_dir):
        parser.error("--processed-dir %s is not a directory" % args.processed_dir)
    if args.subjects_file is not None and not os.path.isfile(args.subjects_file):
        parser.error("--subjects-file %s does not exist" % args.subjects_file)
    if args.grid_reference is not None and not os.path.isfile(args.grid_reference):
        parser.error("--grid-reference %s is not an existing file" % args.grid_reference)
    if args.device == "cpu" and args.gpu_memory_limit_mb is not None:
        parser.error("--gpu-memory-limit-mb has no meaning with --device cpu")
    if args.command == "run":
        _validate_out_dir(args, parser, OUTPUT_NAMES, OUTPUT_PATTERNS)
    elif args.command == "linear-eval":
        args.branches = args.task               # no --branches flag (plan D9)
        args.mode = "streamed"
        _validate_out_dir(args, parser, LINEAR_EVAL_OUTPUTS, ())
    else:
        if os.path.isdir(args.out):
            parser.error("--out %s is a directory; check-streaming writes a JSON file" % args.out)
        if not _creatable_dir(os.path.dirname(os.path.abspath(args.out))):
            parser.error("the directory of --out %s cannot be created" % args.out)
    return args


def parse_args(argv=None):
    """Parse and validate (argparse error = SystemExit(2)); imports no TensorFlow."""
    parser = build_parser()
    args = validate(parser.parse_args(argv), parser)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    return args


def resolve_check_chunks(n_subjects, flags, caps):
    """check-streaming chunk sizes per level (design 4.6), a pure function.

    flags: {'l1', 'l2', 'l4'} -> int or None (None = default min(cap, max(1, n - 1)), so every
    level streams at least two chunks with a remainder). caps: {level key: int32 chunk cap or None}.
    An explicit 0 means the whole group (WARNING 'not chunked'). Values above the cap are capped.
    Returns {l1, l2, l4, source {key: 'default' | 'flag'}, capped {key: {requested, used}}}.
    """
    n = int(n_subjects)
    caps = caps or {}
    out = {"source": {}, "capped": {}}
    for key in LEVEL_KEYS:
        flag = (flags or {}).get(key)
        cap = caps.get(key)
        if flag is None:
            value, source = max(1, n - 1), "default"
        else:
            value, source = int(flag), "flag"
            if value == 0:
                logger.warning("check-streaming: level %s not chunked (--chunk-%s 0)", key[1:], key)
        eff = n if value == 0 else value
        if cap is not None and eff > int(cap):
            out["capped"][key] = {"requested": value, "used": int(cap)}
            if source == "flag":
                logger.warning("check-streaming: --chunk-%s %d above the int32 cap %d; capped",
                               key, value, int(cap))
            value = int(cap)
        out[key] = value
        out["source"][key] = source
    return out
