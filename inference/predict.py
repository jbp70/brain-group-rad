"""predict.py commands (predict-script design 3.3, 5, 7, 8): device setup, `run_predict`,
`run_check_streaming`, predict.json, git commit, non-finite handling and exit codes.

TensorFlow is imported only inside the functions, after `setup_device` (the first TF calls of the
process: hiding the GPUs for --device cpu, TF32 off, memory growth or a fixed limit). In-process
callers that already configured TF (the tests, via tests.tf_setup) pass ``configure_device=False``.

Exit codes (design 7): 0 ok; 2 invalid input / config / checkpoint / no GPU visible; 3 non-finite
outputs (fields still written, no atlas; wins over 5); 4 check-streaming gate exceeded; 5 internal
consistency check failed (files written); 1 unexpected error (traceback logged; GPU OOM names
--chunk-l1 / --device cpu).
"""
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback

from inference import cli as CLI

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVENTION = {"components": "xyz", "signs": "+++", "units": "mm", "swap_roles": False}
GLOBAL_TRANSFORM = "none (training augmentation only)"
OOM_HINT = ("GPU out of memory: use a smaller --chunk-l1 (or --chunk-l2 / --chunk-l4), "
            "--gpu-memory-limit-mb, or --device cpu")
# check-streaming gate G1 (design 8), per trainable-stage precision.
BOUND_SETS = {
    "float32": {"field_max_mm": 1e-3, "field_mean_mm": None, "moved": 1e-3, "matrix": 1e-5,
                "translation_mm": 1e-4},
    "mixed_float16": {"field_max_mm": 0.1, "field_mean_mm": 2e-3, "moved": 5e-3, "matrix": 1e-3,
                      "translation_mm": 0.05},
}
TOL_FLAGS = {"field_max_mm": "tol_field_max_mm", "field_mean_mm": "tol_field_mean_mm",
             "moved": "tol_moved", "matrix": "tol_matrix", "translation_mm": "tol_translation_mm"}
# predict.json keys (design 8); every key is present, also in error records.
PREDICT_KEYS = (
    "status", "message", "command", "git_commit", "git_commit_source", "configure_device",
    "checkpoint", "model_config", "knobs", "precision", "branches", "pipeline", "global_transform",
    "tta", "n_subjects", "subjects", "subject_dirs", "training_group_sizes", "mode", "device",
    "chunks", "grid", "store", "timing_s", "gpu_peak_mb", "nonfinite", "stats", "fp16_max_abs",
    "internal_check", "compress_level", "visible_devices", "convention", "outputs", "input_range",
    "warnings", "levels", "tta_record")
TTA_SPREAD_CSV = "tta_spread.csv"
_SHA_RE = re.compile(r"[0-9a-f]{40}")


class PredictError(Exception):
    """A failure with a defined exit code (design 7, 11)."""

    def __init__(self, code, message):
        super(PredictError, self).__init__(message)
        self.code = int(code)


class _WarningCollector(logging.Handler):
    """Collects WARNING+ messages of every logger during a command (predict.json ``warnings``)."""

    def __init__(self):
        super(_WarningCollector, self).__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        try:
            self.messages.append(record.getMessage())
        except Exception:  # noqa: BLE001  (never break the run on a logging problem)
            pass


# ============================================================================ git commit (8)
def _git_dir_commit(git_dir):
    """Commit from ``.git/HEAD`` (a 40-hex line, or ``ref: <ref>`` resolved through the loose
    ref file, then ``packed-refs``); None if none of these gives a commit."""
    try:
        with open(os.path.join(git_dir, "HEAD")) as f:
            head = f.read().strip()
    except OSError:
        return None
    if _SHA_RE.fullmatch(head):
        return head
    if not head.startswith("ref:"):
        return None
    ref = head[4:].strip()
    try:
        with open(os.path.join(git_dir, *ref.split("/"))) as f:
            sha = f.read().strip()
        if _SHA_RE.fullmatch(sha):
            return sha
    except OSError:
        pass
    try:
        with open(os.path.join(git_dir, "packed-refs")) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and parts[1] == ref and _SHA_RE.fullmatch(parts[0]):
                    return parts[0]
    except OSError:
        pass
    return None


def git_commit(repo_root=REPO_ROOT):
    """(sha or None, source 'git' | 'git_dir' | 'none'); never raises (git may be missing)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
                             text=True, timeout=10, check=True)
        sha = out.stdout.strip()
        if _SHA_RE.fullmatch(sha):
            return sha, "git"
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        pass
    sha = _git_dir_commit(os.path.join(repo_root, ".git"))
    if sha is not None:
        return sha, "git_dir"
    return None, "none"


# ============================================================================ device (3.3)
def setup_device(args):
    """The first TF calls of the process (design 3.3). --device cpu: hide the GPUs, then TF32 off
    without memory growth; --device gpu: TF32 off with memory growth, or a fixed logical-device
    limit (--gpu-memory-limit-mb, as main.py). Returns the visible device names."""
    import tensorflow as tf
    from models import precision as MP
    if args.device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as e:
            raise PredictError(CLI.EXIT_ERROR, "--device cpu: cannot hide the GPUs (TF already "
                                               "initialised): %s" % e)
        MP.configure_tf_precision(tf32=False, memory_growth=False)
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise PredictError(CLI.EXIT_INPUT, "no GPU visible; use --device cpu")
        limit = getattr(args, "gpu_memory_limit_mb", None)
        MP.configure_tf_precision(tf32=False, memory_growth=not limit)
        if limit:
            try:
                for gpu in gpus:
                    tf.config.set_logical_device_configuration(
                        gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=int(limit))])
            except RuntimeError as e:
                raise PredictError(CLI.EXIT_ERROR, "cannot set the GPU memory limit: %s" % e)
    return [d.name for d in tf.config.get_visible_devices()]


# ============================================================================ helpers
def _peak_rss_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:  # noqa: BLE001  (not on every platform)
        return None


def _write_json(path, rec):
    from runtime.logging import to_jsonable
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(to_jsonable(rec), f, indent=1)
    os.replace(tmp, path)


def _is_oom(e):
    tf = sys.modules.get("tensorflow")
    return tf is not None and isinstance(e, tf.errors.ResourceExhaustedError)


def _command(args):
    return ["predict.py"] + list(getattr(args, "argv", None) or sys.argv[1:])


def _gpu_reset():
    import tensorflow as tf
    if not tf.config.list_logical_devices("GPU"):
        return False
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
        return True
    except (ValueError, RuntimeError):
        return False


def _gpu_peak_mb(active):
    import tensorflow as tf
    if not active:
        return None
    try:
        return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2.0 ** 20
    except (ValueError, RuntimeError):
        return None


def _peek_config(checkpoint):
    """(level, model_cfg, split_group_conv) from config.json, without building a model."""
    from inference import loader as LD
    from models.config import ModelConfig
    from runtime import checkpoint as C
    try:
        prefix = C.resolve_checkpoint(checkpoint)
        cfg_path, config = LD._read_config(prefix)
        model_cfg = ModelConfig.from_dict(config["model_config"])
        args = config.get("args") or {}
        level = int(args["level"])
    except (LD.LoaderError, KeyError, TypeError, ValueError) as e:
        raise PredictError(CLI.EXIT_INPUT, "checkpoint %s: %s" % (checkpoint, e))
    return level, model_cfg, bool(args.get("split_group_conv", False))


def _check_unchunked(n, peek, branches):
    from inference import loader as LD
    level, model_cfg, split = peek
    try:
        return LD.check_unchunked(n, level, model_cfg, branches, split)
    except LD.LoaderError as e:
        raise PredictError(CLI.EXIT_INPUT, str(e))


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _peek_chain_configs(checkpoint, level):
    """{run level: (config.json path or None, config dict or None)} of the checkpoint chain, TF-free
    (predict-levels design 6.1): the checkpoint run; for a level-1 run also its level-2 source
    (``cascade.source_config``, else the config of the level-2 ``cascade.stages`` origin
    checkpoint; first readable wins; unreadable -> (first candidate or None, None))."""
    from inference import loader as LD
    from runtime import checkpoint as C
    try:
        cfg_path, config = LD._read_config(C.resolve_checkpoint(checkpoint))
    except (LD.LoaderError, ValueError) as e:
        raise PredictError(CLI.EXIT_INPUT, "checkpoint %s: %s" % (checkpoint, e))
    out = {int(level): (cfg_path, config)}
    if int(level) == 1:
        casc = config.get("cascade") or {}
        cands = []
        if casc.get("source_config"):
            cands.append(casc["source_config"])
        for s in casc.get("stages") or []:
            try:
                is2 = int(s.get("level", -1)) == 2
            except (TypeError, ValueError):
                is2 = False
            if is2 and s.get("origin_checkpoint"):
                p = C.source_config_path(s["origin_checkpoint"])
                if p is not None and p not in cands:
                    cands.append(p)
        found = (cands[0] if cands else None, None)
        for p in cands:
            try:
                found = (p, _read_json(p))
                break
            except (OSError, ValueError) as e:
                logger.debug("level-2 source config %s not readable: %s", p, e)
        out[2] = found
    return out


def _setup_levels(args, rec, level):
    """Level plan + training exposure of ``--levels`` (predict-levels design 3, 6), before any
    subject is read or model built. Fills rec['levels'] (also when refusing); PredictError(2) for a
    level not in the checkpoint or an untrained / unknown skip without --allow-untrained-skip."""
    from inference import levels as LV
    from inference import loader as LD
    from inference import runner as RN
    ckpt_levels = LD.frozen_levels_of(level) + (int(level),)
    selected = getattr(args, "levels", None)
    try:
        plan = LV.plan(ckpt_levels, selected)
    except ValueError as e:
        rec["levels"] = {"requested": LV.levels_text(selected), "checkpoint_levels": list(ckpt_levels),
                         "selected": None}
        raise PredictError(CLI.EXIT_INPUT, str(e))
    allow = bool(getattr(args, "allow_untrained_skip", False))
    tasks = tuple(t for t in RN.branch_tasks(args.branches) if t is not None)
    exp = LV.exposure(plan, tasks, _peek_chain_configs(args.checkpoint, level))
    d = plan.to_dict()
    d["allow_untrained_skip"] = allow
    d["exposure"] = exp
    rec["levels"] = d
    bad = LV.bad_entries(exp)
    if bad and not allow:
        raise PredictError(CLI.EXIT_INPUT, "--levels %s with --branches %s skips levels the checkpoint "
                                           "chain was never trained to skip (or its config is not "
                                           "readable): %s; use --allow-untrained-skip to run it anyway"
                           % (plan.requested, args.branches,
                              "; ".join("%s (%s)" % (LV.entry_text(e), e["status"]) for e in bad)))
    for e in bad:
        exp["forced"] = True
        logger.warning("forced (--allow-untrained-skip): %s is %s in training", LV.entry_text(e), e["status"])
    for e in exp["entries"]:
        if e["status"] == "trained" and e["joint"] < exp["low_threshold"]:
            logger.warning("level %d task %s saw this skip (levels %s skipped) in about %.2f %% of "
                           "training steps", e["level"], e["task"], e["skipped"], 100.0 * e["joint"])
    logger.info("levels: %s%s", plan, " (default: all checkpoint levels)" if plan.default else "")
    return plan


def _read_config_or_none(path):
    if not path:
        return (None, None)
    try:
        return (path, _read_json(path))
    except (OSError, ValueError) as e:
        logger.debug("config %s not readable: %s", path, e)
        return (path, None)


def _peek_flip_chain(checkpoint, level):
    """`_peek_chain_configs` plus the level-4 config of the chain (TTA design 7.4, plan D10), TF-free:
    level-2 run -> its ``cascade.source_config``; level-1 run -> the level-2 config's
    ``cascade.source_config``, else the level-1 config's ``cascade.stages`` level-4
    ``origin_checkpoint`` (`runtime.checkpoint.source_config_path`). Unreadable -> (path or None,
    None)."""
    from runtime import checkpoint as C
    level = int(level)
    chain = _peek_chain_configs(checkpoint, level)
    if level == 4:
        return chain
    own = chain[level][1] or {}
    if level == 2:
        chain[4] = _read_config_or_none((own.get("cascade") or {}).get("source_config"))
        return chain
    cands = []
    l2 = chain.get(2, (None, None))[1]
    if isinstance(l2, dict) and (l2.get("cascade") or {}).get("source_config"):
        cands.append(l2["cascade"]["source_config"])
    for s in (own.get("cascade") or {}).get("stages") or []:
        try:
            is4 = int(s.get("level", -1)) == 4
        except (TypeError, ValueError):
            is4 = False
        if is4 and s.get("origin_checkpoint"):
            p = C.source_config_path(s["origin_checkpoint"])
            if p is not None and p not in cands:
                cands.append(p)
    found = (cands[0] if cands else None, None)
    for p in cands:
        entry = _read_config_or_none(p)
        if entry[1] is not None:
            found = entry
            break
    chain[4] = found
    return chain


def _setup_tta(args, rec, plan, level):
    """TTAConfig of ``--tta`` / ``--tta-levels`` (TTA design 7.1, 7.4) or None for 'none'. Fills
    rec['tta_record'] (also when refusing: mode, levels requested, error); PredictError(2) for a
    TTA level the plan does not run. Logs the training-flip WARNINGs (warn only)."""
    from inference import levels as LV
    from inference import tta as TTA
    mode = getattr(args, "tta", None) or "none"
    if mode == "none":
        return None
    requested = getattr(args, "tta_levels", None)
    levels = tuple(int(lv) for lv in (requested if requested is not None else plan.run))
    bad = [lv for lv in levels if lv not in plan.run]
    if bad:
        msg = ("--tta-levels %s: level(s) %s are not run by the level plan (run levels %s)"
               % (LV.levels_text(levels), ", ".join(str(lv) for lv in bad),
                  LV.levels_text(plan.run)))
        rec["tta_record"] = {"mode": mode, "levels": list(levels), "error": msg}
        raise PredictError(CLI.EXIT_INPUT, msg)
    cfg = TTA.TTAConfig(mode, levels)
    probs = TTA.train_flip_probs(_peek_flip_chain(args.checkpoint, level))
    for w in TTA.flip_prob_warnings(probs, levels):
        logger.warning("%s", w)
    d = cfg.to_dict()
    d.update(linear_averaging=TTA.LINEAR_AVERAGING, svf_accumulation=TTA.SVF_ACCUMULATION,
             train_flip_prob={str(lv): probs[lv] for lv in sorted(probs, reverse=True)})
    rec["tta_record"] = d
    logger.info("TTA %s at levels %s (%d flips)", mode, LV.levels_text(levels), len(cfg.flips))
    return cfg


def _tta_csv_value(v):
    if v is None:
        return ""
    return "%.9g" % float(v)


def _write_tta_spread(path, rows, subs):
    """tta_spread.csv (TTA design 5): subject, subject_dir, level, task, `tta.SPREAD_COLUMNS`; one
    row per subject x level x task; empty where a column does not apply (plan D4)."""
    import csv
    from inference import tta as TTA
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("subject", "subject_dir", "level", "task") + tuple(TTA.SPREAD_COLUMNS))
        for r in rows:
            s = subs[int(r["image"])]
            w.writerow([s["subject"], s["dir"], int(r["level"]), r["task"]]
                       + [_tta_csv_value(r.get(c)) for c in TTA.SPREAD_COLUMNS])
    os.replace(tmp, path)


def _tta_finish(rec, summary, args, subs):
    """After a TTA run: tta_spread.csv, rec['tta_record'] spread / gauge / nonfinite_flips /
    timing_s, INFO per level and task. Returns the CSV name (or None without TTA rows)."""
    from inference import tta as TTA
    tta = summary.get("tta")
    if not tta or rec.get("tta_record") is None:
        return None
    rows = tta["rows"]
    spread = TTA.summarize_spread(rows)
    for key, (deg, mm) in tta["gauge"].items():
        spread.setdefault(key, {}).update(gauge_deg=deg, gauge_mm=mm)
    rec["tta_record"].update(spread=spread, nonfinite_flips=tta["nonfinite_flips"],
                             timing_s=tta["timing_s"])
    for key in sorted(spread):
        s = spread[key]
        rms, shift = s.get("spread_rms_mm") or {}, s.get("shift_mm") or {}
        logger.info("TTA %s: spread_rms_mm mean %s max %s, shift_mm mean %s%s", key, rms.get("mean"),
                    rms.get("max"), shift.get("mean"),
                    "" if "gauge_deg" not in s else ", gauge %s deg / %s mm" % (s["gauge_deg"], s["gauge_mm"]))
    if not rows:
        return None
    _write_tta_spread(os.path.join(args.out, TTA_SPREAD_CSV), rows, subs)
    return TTA_SPREAD_CSV


def _read_subjects(args, rec, n_first=None):
    """Subjects (find_subjects order, first ``n_first``), grid check on the first header, then
    images / labels (load_grid_data), finiteness and intensity range. Fills rec grid / subjects."""
    import nibabel as nib
    import numpy as np
    from evaluation import evaluate as EV
    from inference import canonical as CN
    try:
        subs = EV.find_subjects(args.processed_dir, args.label_name, args.subjects_file)
    except (OSError, ValueError) as e:
        raise PredictError(CLI.EXIT_INPUT, "subjects: %s" % e)
    if n_first is not None:
        if len(subs) < n_first:
            raise PredictError(CLI.EXIT_INPUT, "--n-subjects %d but only %d subjects in %s"
                               % (n_first, len(subs), args.processed_dir))
        subs = subs[:n_first]
    rec["n_subjects"] = len(subs)
    rec["subjects"] = [s["subject"] for s in subs]
    rec["subject_dirs"] = [s["dir"] for s in subs]
    try:
        ref = nib.load(subs[0]["image"])
        aff0, shape0 = ref.affine, tuple(int(s) for s in ref.shape[:3])
        rec["grid"] = CN.check_grid(aff0, shape0, EV.spacing_of(aff0), args.grid_reference)
        images, labels, masks, shape, aff = EV.load_grid_data(subs)
    except ValueError as e:                     # GridError is a ValueError
        raise PredictError(CLI.EXIT_INPUT, str(e))
    del masks
    spacing = EV.spacing_of(aff)
    for k, s in enumerate(subs):
        if not np.all(np.isfinite(images[k])):
            raise PredictError(CLI.EXIT_INPUT, "input image of %s (%s) has non-finite values"
                               % (s["subject"], s["image"]))
    lo, hi = float(images.min()), float(images.max())
    rec["input_range"] = [lo, hi]
    if hi > 1.5 or lo < -1e-3:
        logger.warning("input intensities in [%.4g, %.4g]; the model was trained on [0, 1] images", lo, hi)
    return subs, images, labels, shape, aff, spacing


def _load(args):
    from inference import loader as LD
    try:
        return LD.load_model(args.checkpoint, args.branches, precision=args.precision,
                             device=args.device)
    except LD.LoaderError as e:
        raise PredictError(CLI.EXIT_INPUT, str(e))


def _checkpoint_record(loaded):
    cfg = loaded.config or {}
    casc = cfg.get("cascade") or {}
    frozen = []
    if loaded.level == 2:
        frozen.append({"level": 4, "checkpoint_key": "frozen_stage",
                       "origin_checkpoint": casc.get("source_checkpoint"),
                       "origin_step": casc.get("source_step"), "fingerprint_matched": True})
    elif loaded.level == 1:
        for s in casc.get("stages") or []:
            frozen.append({"level": s.get("level"), "checkpoint_key": s.get("checkpoint_key"),
                           "origin_checkpoint": s.get("origin_checkpoint"),
                           "origin_step": s.get("origin_step"), "fingerprint_matched": True})
    run_dir = loaded.run_dir
    return {"prefix": loaded.prefix, "step": loaded.step, "run_dir": run_dir,
            "config": None if run_dir is None else os.path.join(run_dir, "config.json"),
            "level": loaded.level, "frozen_stages": frozen}


def _chain_text(levels, plan=None):
    """'L4 -> L2 -> L1' (default plan: today's text) or the per-level kinds of a non-default plan,
    e.g. 'L4 skipped (identity) -> L2 -> L1' or 'L4 -> L2; L1 not run (truncated after level 2)'."""
    if plan is None or plan.default:
        return " -> ".join("L%d" % lv for lv in levels)
    parts = []
    for lv in plan.ckpt_levels:
        kind = plan.kind(lv)
        if kind == "run":
            parts.append("L%d" % lv)
        elif kind == "skipped":
            parts.append("L%d skipped (identity)" % lv)
    text = " -> ".join(parts)
    if plan.truncated:
        text += "; %s not run (truncated after level %d)" % (
            ", ".join("L%d" % lv for lv in plan.truncated), plan.last)
    return text


def _pipeline(branches, levels, plan=None):
    chain = _chain_text(levels, plan)
    return {"rigid": "rigid cascade (%s) on the raw images; fwd / inv = linear field of A" % chain,
            "affine": "affine cascade (%s) on the raw images; fwd / inv = linear field of A" % chain,
            "deform": "deform cascade (%s) on the raw images (inputs MNI-affine-aligned); fwd = u, "
                      "inv = u_inv" % chain,
            "affine+deform": "affine cascade (%s); images resampled once into the affine central space "
                             "(I o A^-1, trilinear, zero fill); deform cascade (%s) on them; fwd = "
                             "A^-1 o (id + u), inv its inverse" % (chain, chain)}[branches]


def _slice(t, k):
    return None if t is None else t[k:k + 1]


def _nonfinite(x):
    import numpy as np
    return int(x.size - np.count_nonzero(np.isfinite(x)))


def _finite_max_abs(x):
    import numpy as np
    a = np.abs(x[np.isfinite(x)])
    return float(a.max()) if a.size else None


def _new_record(args, configure_device):
    rec = {k: None for k in PREDICT_KEYS}
    rec.update(status="error", command=_command(args), configure_device=bool(configure_device),
               branches=args.branches, global_transform=GLOBAL_TRANSFORM, mode=getattr(args, "mode", None),
               tta=getattr(args, "tta", None), device=args.device, convention=dict(CONVENTION),
               compress_level=getattr(args, "compress_level", None), timing_s={}, warnings=[])
    return rec


def _finish(rec, collector, t0, path, kind):
    rec["timing_s"]["total"] = time.time() - t0
    rec["warnings"] = list(collector.messages)
    if isinstance(rec.get("store"), dict):
        rec["store"]["peak_rss_gb"] = _peak_rss_gb()
    try:
        _write_json(path, rec)
        logger.info("%s written (status %s)", path, rec["status"])
    except Exception as e:  # noqa: BLE001  (best effort; the exit code still reports the run)
        logger.error("cannot write %s %s: %s", kind, path, e)


def _command_wrapper(body, args, configure_device, rec, path, kind):
    t0 = time.time()
    collector = _WarningCollector()
    root = logging.getLogger()
    root.addHandler(collector)
    code = CLI.EXIT_ERROR
    try:
        sha, source = git_commit()
        rec["git_commit"], rec["git_commit_source"] = sha, source
        if sha is None:
            logger.warning("git commit unknown")
        code = body(args, configure_device, rec)
    except PredictError as e:
        code = e.code
        rec["status"], rec["message"] = "error", str(e)
        logger.error("%s", e)
    except Exception as e:  # noqa: BLE001  (exit 1 with the traceback logged)
        code = CLI.EXIT_ERROR
        rec["status"] = "error"
        if isinstance(e, ValueError) and type(e).__name__ in ("LoaderError", "GridError"):
            code = CLI.EXIT_INPUT                   # raised outside the explicit mappings
            rec["message"] = str(e)
            logger.error("%s", e)
        elif _is_oom(e):
            rec["message"] = "%s (%s)" % (OOM_HINT, str(e).splitlines()[0] if str(e) else type(e).__name__)
            logger.error("%s\n%s", OOM_HINT, e)
        else:
            rec["message"] = "%s: %s" % (type(e).__name__, e)
            logger.error("unexpected error:\n%s", traceback.format_exc())
    finally:
        _finish(rec, collector, t0, path, kind)
        root.removeHandler(collector)
    return code


# ============================================================================ run
class _Writer(object):
    """Per-chunk sink of `run`: canonical fields on the device, internal check of the first K
    subjects, moved image / warped labels (evaluator warps), NIfTI files, atlas sums."""

    def __init__(self, args, subs, images, labels, aff, shape, spacing, k_check):
        from inference import canonical as CN
        self.CN = CN
        self.args = args
        self.subs, self.images, self.labels = subs, images, labels
        self.aff, self.shape, self.spacing = aff, tuple(shape), tuple(spacing)
        self.k_check = int(k_check)
        self.atlas = CN.AtlasAccumulator(len(subs), shape, labels.dtype)
        self.nonfinite = {}
        self.checks = {}
        self.outputs = []
        self.t_write = 0.0
        self.t_check = 0.0
        self.max_fwd = None
        self.max_inv = None

    def _note_max(self, attr, value):
        if value is not None:
            cur = getattr(self, attr)
            setattr(self, attr, value if cur is None else max(cur, value))

    def sink(self, i0, i1, res):
        CN, br = self.CN, self.args.branches
        for k in range(i1 - i0):
            i = i0 + k
            A, U1, U1i = _slice(res["A"], k), _slice(res["U1"], k), _slice(res["U1_inv"], k)
            fwd_t, inv_t = CN.to_canonical(br, A, U1, U1i, self.shape, self.spacing)
            fwd, inv = fwd_t.numpy()[0], inv_t.numpy()[0]
            del fwd_t, inv_t
            t = time.time()
            moved, wl = CN.moved_and_labels(self.images[i], self.labels[i], fwd)
            name = self.subs[i]["dir"]
            counts = {"fwd": _nonfinite(fwd), "inv": _nonfinite(inv), "moved": _nonfinite(moved)}
            self.nonfinite[name] = counts
            fwd_mm, inv_mm = CN.to_mm(fwd, self.spacing), CN.to_mm(inv, self.spacing)
            self._note_max("max_fwd", _finite_max_abs(fwd_mm))
            self._note_max("max_inv", _finite_max_abs(inv_mm))
            for kind, arr in (("fwd", fwd_mm), ("inv", inv_mm), ("moved", moved)):
                fname = "%s_%s.nii.gz" % (name, kind)
                CN.write_nifti(os.path.join(self.args.out, fname), arr, self.aff)
                self.outputs.append(fname)
            del fwd_mm, inv_mm
            self.atlas.add(i, moved, wl)
            self.t_write += time.time() - t
            if sum(counts.values()):
                logger.error("subject %s: non-finite values fwd %d, inv %d, moved %d", name,
                             counts["fwd"], counts["inv"], counts["moved"])
            if i < self.k_check:
                t = time.time()
                rec = CN.internal_check(self.images[i], moved, fwd, inv, br, A, U1, U1i, self.spacing)
                self.checks[name] = rec
                self.t_check += time.time() - t
                logger.info("internal check %s: coord fwd %.3g / inv %.3g vox, moved %.3g%s, inverse "
                            "consistency %.3g mm, interior %.1f %% -> %s", name,
                            rec["coord_fwd_max_vox"], rec["coord_inv_max_vox"],
                            rec["moved_vs_internal_max_abs"], "" if rec["moved_gated"] else " (not gated)",
                            rec["inv_consistency_max_mm"], 100.0 * rec["interior_fraction"],
                            "pass" if rec["passed"] else "FAIL")
            logger.info("subject %d / %d %s written", i + 1, len(self.subs), name)


def _run_predict(args, configure_device, rec):
    timing = rec["timing_s"]
    if configure_device:
        setup_device(args)
    import tensorflow as tf
    from inference import canonical as CN
    from inference import runner as RN
    from inference import store as SToR
    visible = tf.config.get_visible_devices()
    rec["visible_devices"] = [d.name for d in visible]
    logger.debug("visible devices: %s", visible)
    peek = _peek_config(args.checkpoint)
    plan = _setup_levels(args, rec, peek[0])
    tta_cfg = _setup_tta(args, rec, plan, peek[0])
    t = time.time()
    subs, images, labels, shape, aff, spacing = _read_subjects(args, rec)
    timing["read"] = time.time() - t
    n = len(subs)
    if args.mode == "unchunked":
        _check_unchunked(n, peek, args.branches)
    k_check = min(int(args.check_internal), n)
    if k_check < int(args.check_internal):
        logger.info("--check-internal %d clipped to the %d subjects", args.check_internal, n)
    t = time.time()
    loaded = _load(args)
    timing["load"] = time.time() - t
    rec.update(checkpoint=_checkpoint_record(loaded), model_config=loaded.model_cfg.to_dict(),
               knobs=loaded.knobs, precision=loaded.precision,
               training_group_sizes=loaded.training_group_sizes,
               pipeline=_pipeline(args.branches, loaded.levels, plan))
    if tta_cfg is not None:
        rec["pipeline"] += " with 8-flip TTA at levels %s" % ",".join(str(lv) for lv in tta_cfg.levels)
    store = None
    if args.mode == "streamed":
        tdt = "float16" if loaded.precision["trainable"] == "mixed_float16" else "float32"
        if tta_cfg is not None and RN.branch_tasks(args.branches)[1] is not None:
            est = SToR.estimate_store_gb(loaded.level, loaded.model_cfg, n, args.branches, tdt,
                                         tta_levels=tta_cfg.levels)
        else:
            est = SToR.estimate_store_gb(loaded.level, loaded.model_cfg, n, args.branches, tdt)
        store_dir = args.store_dir or os.environ.get("TMPDIR") or os.path.join(args.out, "_stream_tmp")
        store = SToR.choose_store(args.store, est, store_dir)
        rec["store"] = {"kind": store.kind, "dir": getattr(store, "directory", None),
                        "estimate_gb": est, "choice": store.choice}
        runner = RN.StreamedRunner(loaded, store, chunks={"l1": args.chunk_l1, "l2": args.chunk_l2,
                                                          "l4": args.chunk_l4}, plan=plan, tta=tta_cfg)
    else:
        rec["store"] = {"kind": "none (unchunked)", "dir": None, "estimate_gb": None}
        runner = RN.UnchunkedRunner(loaded, plan=plan)
    os.makedirs(args.out, exist_ok=True)
    writer = _Writer(args, subs, images, labels, aff, shape, spacing, k_check)
    logger.info("run: %d subjects, branches %s, mode %s, level-%d checkpoint step %s, trainable %s",
                n, args.branches, args.mode, loaded.level, loaded.step, loaded.precision["trainable"])
    t = time.time()
    try:
        with CN.compress_level(args.compress_level):
            summary = runner.run(images, args.branches, writer.sink)
    finally:
        if store is not None:
            rec["store"]["peak_store_gb"] = store.peak_nbytes() / 1e9
            store.close()
    timing["model_and_write"] = time.time() - t
    timing["write"] = writer.t_write
    timing["internal_check"] = writer.t_check
    for key, sec in sorted(summary["timing_s"].items()):
        timing[key] = sec
        peak = (summary["gpu_peak_mb"] or {}).get(key)
        logger.info("%s: %.1f s%s", key, sec, "" if peak is None else ", TF peak %.0f MiB" % peak)
    rec["chunks"] = {"l1": summary["chunk_sizes"].get("l1"), "l2": summary["chunk_sizes"].get("l2"),
                     "l4": summary["chunk_sizes"].get("l4"), "capped": summary["capped"],
                     "bounds": summary["chunks"]}
    rec["gpu_peak_mb"] = summary["gpu_peak_mb"]
    rec["fp16_max_abs"] = summary["fp16_max_abs"]
    total_nf = int(sum(sum(c.values()) for c in writer.nonfinite.values()))
    rec["nonfinite"] = {"per_subject": writer.nonfinite, "total": total_nf}
    rec["stats"] = {"max_abs_fwd_mm": writer.max_fwd, "max_abs_inv_mm": writer.max_inv,
                    "max_abs_v": summary["stats"]["max_abs_v"]}
    passed = None
    if k_check:
        passed = all(r["passed"] for r in writer.checks.values())
    rec["internal_check"] = {"k": k_check, "gate": {"coord_vox": CN.GATE_COORD_VOX, "moved": CN.GATE_MOVED},
                             "subjects": writer.checks, "passed": passed}
    outputs = list(writer.outputs)
    tta_csv = _tta_finish(rec, summary, args, subs)
    if tta_csv is not None:
        outputs.append(tta_csv)
    if total_nf == 0:
        t = time.time()
        with CN.compress_level(args.compress_level):
            writer.atlas.write(args.out, aff)
        timing["atlas"] = time.time() - t
        outputs += ["atlas.nii.gz", "atlas_seg.nii.gz"]
    rec["outputs"] = outputs + ["predict.json"]
    if total_nf:
        rec["status"] = "nonfinite"
        rec["message"] = "%d non-finite output values; atlas files not written" % total_nf
        bad_flips = {k: v for k, v in ((summary.get("tta") or {}).get("nonfinite_flips") or {}).items() if v}
        if bad_flips:
            rec["message"] += "; TTA flips with non-finite values (level/task: flip indices): %s" % bad_flips
        logger.error("%s: %s", rec["message"], {k: v for k, v in writer.nonfinite.items() if sum(v.values())})
        return CLI.EXIT_NONFINITE
    if passed is False:
        bad = {k: v for k, v in writer.checks.items() if not v["passed"]}
        rec["status"] = "internal_mismatch"
        rec["message"] = "internal consistency check failed for %s" % sorted(bad)
        logger.error("internal consistency check failed (gates coord %g vox, moved %g): %s",
                     CN.GATE_COORD_VOX, CN.GATE_MOVED, bad)
        return CLI.EXIT_INTERNAL
    rec["status"] = "ok"
    return CLI.EXIT_OK


def run_predict(args, configure_device=True):
    """``predict.py run`` -> exit code; predict.json written last (also on failure)."""
    rec = _new_record(args, configure_device)
    return _command_wrapper(_run_predict, args, configure_device, rec,
                            os.path.join(args.out, "predict.json"), "predict.json")


# ============================================================================ check-streaming
def _collect_side(runner, images, branches, shape, spacing):
    """Canonical fwd / inv (mm) and moved images of every subject from one runner."""
    from evaluation import fields as F
    from inference import canonical as CN
    n = images.shape[0]
    out = {"fwd": [None] * n, "inv": [None] * n, "moved": [None] * n}

    def sink(i0, i1, res):
        for k in range(i1 - i0):
            i = i0 + k
            fwd_t, inv_t = CN.to_canonical(branches, _slice(res["A"], k), _slice(res["U1"], k),
                                           _slice(res["U1_inv"], k), shape, spacing)
            fwd, inv = fwd_t.numpy()[0], inv_t.numpy()[0]
            out["moved"][i] = F.warp(images[i], fwd, order=1)
            out["fwd"][i], out["inv"][i] = CN.to_mm(fwd, spacing), CN.to_mm(inv, spacing)
    active = _gpu_reset()
    t = time.time()
    summary = runner.run(images, branches, sink)
    return out, summary, _gpu_peak_mb(active), time.time() - t


def _differences(un, st, sum_un, sum_st):
    import numpy as np
    d = {}
    for key in ("fwd", "inv"):
        mx, tot, cnt = 0.0, 0.0, 0
        for a, b in zip(st[key], un[key]):
            e = np.abs(a.astype(np.float64) - b.astype(np.float64))
            mx, tot, cnt = max(mx, float(e.max())), tot + float(e.sum()), cnt + e.size
        d[key + "_max_mm"], d[key + "_mean_mm"] = mx, tot / max(cnt, 1)
    d["moved_max"] = max(float(np.max(np.abs(a - b))) for a, b in zip(st["moved"], un["moved"]))
    if sum_un["A"] is not None:
        da = np.abs(np.asarray(sum_st["A"], np.float64) - np.asarray(sum_un["A"], np.float64))
        d["dA_lin"], d["dA_t_mm"] = float(da[:, :3, :3].max()), float(da[:, :3, 3].max())
    return d


def _gate(d, bounds):
    """Exceeded bounds as 'quantity value > bound' strings (non-finite values exceed)."""
    import math
    checks = [("fwd_max_mm", "field_max_mm"), ("inv_max_mm", "field_max_mm"),
              ("fwd_mean_mm", "field_mean_mm"), ("inv_mean_mm", "field_mean_mm"),
              ("moved_max", "moved"), ("dA_lin", "matrix"), ("dA_t_mm", "translation_mm")]
    bad = []
    for q, b in checks:
        if q not in d or bounds.get(b) is None:
            continue
        v = d[q]
        if not (math.isfinite(v) and v <= bounds[b]):
            bad.append("%s %.4g > %g" % (q, v, bounds[b]))
    return bad


def _run_check(args, configure_device, rec):
    timing = rec["timing_s"]
    if configure_device:
        setup_device(args)
    import tensorflow as tf
    from inference import runner as RN
    from inference import store as SToR
    from inference import streaming as ST
    visible = tf.config.get_visible_devices()
    rec["visible_devices"] = [d.name for d in visible]
    logger.debug("visible devices: %s", visible)
    n = int(args.n_subjects)
    peek = _peek_config(args.checkpoint)
    plan = _setup_levels(args, rec, peek[0])
    _check_unchunked(n, peek, args.branches)   # config only, no model
    t = time.time()
    subs, images, labels, shape, aff, spacing = _read_subjects(args, rec, n_first=n)
    del labels
    timing["read"] = time.time() - t
    t = time.time()
    loaded = _load(args)
    timing["load"] = time.time() - t
    rec["checkpoint"] = {"prefix": loaded.prefix, "step": loaded.step}
    lin, dfm = RN.branch_tasks(args.branches)
    tasks = tuple(x for x in (lin, dfm) if x is not None)
    models = loaded.stage_models()
    keys = ["l%d" % lv for lv in plan.run]
    caps = {"l%d" % lv: ST.chunk_cap(lv, models[lv].config, tasks) for lv in plan.run}
    flags = {k: getattr(args, "chunk_" + k) for k in keys}
    res = CLI.resolve_check_chunks(n, flags, caps)
    chunks = {k: res[k] for k in keys}
    rec["chunks"] = dict(chunks, source={k: res["source"][k] for k in keys},
                         capped={k: v for k, v in res["capped"].items() if k in keys})
    bound_set = "mixed_float16" if loaded.precision["trainable"] == "mixed_float16" else "float32"
    bounds = dict(BOUND_SETS[bound_set])
    for q, flag in TOL_FLAGS.items():
        if getattr(args, flag, None) is not None:
            bounds[q] = float(getattr(args, flag))
    rec["precision"] = {"bound_set": bound_set, "trainable": loaded.precision["trainable"],
                        "record": loaded.precision}
    rec["bounds"] = bounds
    logger.info("check-streaming: %d subjects, branches %s, chunks %s, bound set %s",
                n, args.branches, rec["chunks"], bound_set)
    un, sum_un, peak_un, t_un = _collect_side(RN.UnchunkedRunner(loaded, plan=plan), images,
                                              args.branches, shape, spacing)
    tdt = "float16" if bound_set == "mixed_float16" else "float32"
    est = SToR.estimate_store_gb(loaded.level, loaded.model_cfg, n, args.branches, tdt)
    store_dir = os.environ.get("TMPDIR") or os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                                         "_stream_tmp")
    store = SToR.choose_store("auto", est, store_dir)
    try:
        st, sum_st, peak_st, t_st = _collect_side(RN.StreamedRunner(loaded, store, chunks=chunks,
                                                                    plan=plan),
                                                  images, args.branches, shape, spacing)
    finally:
        store.close()
    diff = _differences(un, st, sum_un, sum_st)
    exceeded = _gate(diff, bounds)
    rec["differences"] = {args.branches: diff}
    rec["exceeded"] = exceeded
    rec["chunk_bounds"] = {"streamed": sum_st["chunks"], "unchunked": sum_un["chunks"]}
    rec["gpu_peak_mb"] = {"unchunked": peak_un, "streamed": peak_st}
    timing.update(unchunked=t_un, streamed=t_st)
    logger.info("check-streaming differences %s: %s", args.branches,
                ", ".join("%s %.3g" % kv for kv in sorted(diff.items())))
    if exceeded:
        rec["status"] = "fail"
        rec["message"] = "gate exceeded: %s" % "; ".join(exceeded)
        logger.error("check-streaming gate (%s bounds) exceeded: %s", bound_set, "; ".join(exceeded))
        return CLI.EXIT_GATE
    rec["status"] = "pass"
    return CLI.EXIT_OK


CHECK_KEYS = ("status", "message", "command", "git_commit", "git_commit_source", "precision",
              "bounds", "chunks", "chunk_bounds", "n_subjects", "subjects", "subject_dirs",
              "branches", "differences", "exceeded", "gpu_peak_mb", "timing_s", "grid",
              "checkpoint", "device", "visible_devices", "input_range", "warnings", "levels")


def run_check_streaming(args, configure_device=True):
    """``predict.py check-streaming`` -> exit code (4 = gate exceeded); JSON report at --out."""
    rec = {k: None for k in CHECK_KEYS}
    rec.update(status="error", command=_command(args), branches=args.branches, device=args.device,
               timing_s={}, warnings=[])
    return _command_wrapper(_run_check, args, configure_device, rec, args.out, "check-streaming JSON")
