"""Post-training synthetic-contrast evaluation (synth-contrast design section 8, H2; plan D-e).

One process evaluates one or two runs (real-only and mixed) of the same level on the SAME
synthetic validation sets; nothing is written into a run directory (output only to ``out_dir``).

Steps (`run_eval`):
1. `load_run_config`: <run>/config.json (path rule of `inference.loader._read_config`:
   `C.source_config_path` of <run>/ckpt/last) -> argparse.Namespace of the saved ``args`` on top
   of the parser defaults (fields missing from older configs, e.g. every ``synth_*`` of a
   real-only run, get the defaults -> synthesis off), every ``*_parsed`` value normalised
   recursively (JSON turned int dict keys into strings: ``cascade_skip_stage_parsed {task: {"4":
   p}}`` -> ``{task: {4: p}}``). `train_config_for_eval` = `A.train_config_from_args` (val PNGs
   stay off: ``val_png_dir`` None).
2. `build_trainer`: the model (saved ``model_config``) and the frozen stages exactly like the
   resume branch of `main.main` (`GroupRegLevelModel(...).build_model`, `FrozenStage` /
   `FrozenChain.from_config`), a Trainer with placeholder sources (``next()`` raises), restore of
   <run>/ckpt/last through `C.RunCheckpoint` (never saved), frozen-stage fingerprints checked,
   BN freeze re-applied when the saved ``bn_recalibration.freeze`` is true.
3. `real_val_sets`: the validation sets of `main._build_dali_data` without DALI.
4. `sanity_gate`: `trainer.validate()` on the real sets vs the run's logged validation at the
   checkpoint step (logs/metrics.jsonl): every mm metric within 1e-2 mm, every Dice within 1e-3.
5. `synthetic_val_sets`: per preset, sets built on the CPU through the ``group_images_fn`` hooks
   of `fd.ValidationSet.build` / `fd.DeformValidationSet.build` (same groups and transforms as
   the real sets; images synthesised in template space from the val ``synthfine`` labels with one
   contrast per group, `synth.preset_params` seeded by (preset, task, group)); named
   ``synth_<preset>_<real set name>``, so `Trainer._identity_cache` / ``_frozen_cache`` (keyed by
   set name) never reuse another preset's or the real sets' baselines.
6. Both trainers stay resident; each preset's sets are built ONCE per distinct validation spec
   and assigned to every run of that spec (``trainer.val_sets = sets``) before ``validate()``,
   then freed. ``eval_synth.json`` (all metrics per run and preset, shared-set flag, sha256 per
   set, sanity gates, spec per run, ru_maxrss after each preset) and ``eval_synth.csv`` (paired
   table: one value column per run, diff = second - first).

Exit codes of `run_eval`: 0 OK, 5 failed sanity gate (JSON written with the gate results, no
preset evaluated). Configuration problems raise `EvalError` (the CLI maps it to exit 2).
"""
import argparse
import contextlib
import csv
import hashlib
import json
import os
import sys
import time

from runtime import args as A

EXIT_GATE_FAILED = 5
GATE_TOL_MM = 1e-2
GATE_TOL_DICE = 1e-3
SYNTH_PREFIX = "synth_"
JSON_NAME = "eval_synth.json"
CSV_NAME = "eval_synth.csv"
# Seed component of the synthetic group contrasts per real set task (the _g sets share their
# task's contrasts, so G and non-G sets see the same images).
TASK_SEED = {"rigid": 1, "affine": 2, "deform": 3}


class EvalError(ValueError):
    """A run / config / data problem the user can fix (CLI exit 2)."""


# ============================================================================ 1. config
def _int_keys(v):
    """Recursively convert decimal-string dict keys back to int (JSON round trip of *_parsed)."""
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if isinstance(k, str) and k.lstrip("-").isdigit():
                k = int(k)
            out[k] = _int_keys(x)
        return out
    if isinstance(v, list):
        return [_int_keys(x) for x in v]
    return v


def _parser_defaults():
    """Namespace dict of the training parser's defaults (type-converted)."""
    return dict(vars(A.build_parser().parse_args([])))


def config_path_of(run_dir):
    """<run>/config.json: the rule of `inference.loader._read_config` (`C.source_config_path` of
    a prefix <run>/ckpt/last/<name> is three directories up), computed without TF."""
    path = os.path.join(os.path.abspath(run_dir), "config.json")
    if not os.path.exists(path):
        raise EvalError("no config.json in run directory %s" % run_dir)
    return path


def load_run_config(run_dir):
    """(argparse.Namespace of the saved args on top of the parser defaults, raw config dict).

    Every ``*_parsed`` value has its decimal-string dict keys converted back to int."""
    path = config_path_of(run_dir)
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        raise EvalError("cannot read %s: %s" % (path, e))
    saved = raw.get("args")
    if not isinstance(saved, dict) or saved.get("level") is None:
        raise EvalError("%s: no args / args.level" % path)
    d = _parser_defaults()
    for k, v in saved.items():
        d[k] = _int_keys(v) if k.endswith("_parsed") else v
    return argparse.Namespace(**d), raw


def train_config_for_eval(ns):
    """`A.train_config_from_args(ns)` with the validation PNGs off (``val_png_dir`` None)."""
    cfg = A.train_config_from_args(ns)
    cfg.val_png_dir = None
    return cfg


# ============================================================================ 2. model + trainer
class _NoData(object):
    """Placeholder training source: evaluation never draws a training batch."""

    def __init__(self, task):
        self.task = task

    def __iter__(self):
        return self

    def __next__(self):
        raise RuntimeError("eval_synth: no training data for task %r (evaluation only)" % self.task)


def _policy(ns):
    return "mixed_float16" if bool(getattr(ns, "amp", False)) else "float32"


@contextlib.contextmanager
def run_scope(entry):
    """Global Keras policy of a run (``mixed_float16`` for AMP runs, as main.py sets it) while its
    model is built and while it validates."""
    from models.precision import policy_scope
    with policy_scope(_policy(entry["ns"])):
        yield


def build_trainer(run_dir, ns, raw_cfg, data_root):
    """Restore <run>/ckpt/last read-only -> dict(trainer, model, frozen, step, prefix, ns, raw).

    Model + frozen stages as in the resume branch of `main.main`; the checkpoint is restored
    through `C.RunCheckpoint` (no save is ever called); BN freeze re-applied from the saved
    ``bn_recalibration`` record."""
    import main as M
    from data_loading import data_module as dm
    from data_loading import postprocess
    from models.config import ModelConfig
    from models.group_net import GroupRegLevelModel
    from models.precision import policy_scope
    from runtime import checkpoint as C
    from runtime import optimizers as O
    from runtime.run import Trainer

    run_dir = os.path.abspath(run_dir)
    last_dir = os.path.join(run_dir, "ckpt", "last")
    try:
        prefix = C.resolve_checkpoint(last_dir)
    except ValueError as e:
        raise EvalError("run %s: %s" % (run_dir, e))
    model_cfg = ModelConfig.from_dict(raw_cfg["model_config"])
    level = int(ns.level)
    cascade = raw_cfg.get("cascade")
    if level in (2, 1) and not cascade:
        raise EvalError("run %s: a level-%d run needs the config.json cascade record" % (run_dir, level))
    with policy_scope(_policy(ns)):
        model = GroupRegLevelModel(level, model_cfg, gradient_checkpointing=ns.gradient_checkpointing,
                                   lean_sampler=ns.lean_sampler, split_group_conv=ns.split_group_conv,
                                   fp16_act_clip=ns.fp16_act_clip,
                                   fp16_act_clip_scope=ns.fp16_act_clip_scope,
                                   deform_decoder_dtype=ns.deform_decoder_dtype,
                                   bn_epsilon_override=ns.bn_epsilon_override
                                   ).build_model(M._build_n(ns))
        frozen = None
        if cascade:
            from models.cascade import FrozenChain, FrozenStage
            grid = (cascade or {}).get("frozen_integration_grid", "native")   # fine-grid-integration D4
            frozen = (FrozenChain.from_config(model_cfg, integration_grid=grid) if level == 1
                      else FrozenStage.from_config(model_cfg, integration_grid=grid))
        optimizer = O.make_optimizer(ns.optimizer, lr=ns.lr, weight_decay=ns.weight_decay,
                                     momentum=ns.momentum, dampening=ns.dampening,
                                     nesterov=ns.nesterov, decoupled=ns.decoupled,
                                     clipnorm=ns.clipnorm, exclude_wd=ns.exclude_from_weight_decay,
                                     amp=ns.amp, jit_compile=ns.xla)
        cfg = train_config_for_eval(ns)
        S = postprocess.brain_S_matrix(level, os.path.join(data_root, dm.TEMPLATE_DIR))
        sources = {t: _NoData(t) for t in ns.task_ratio_parsed}
        trainer = Trainer(model, optimizer, sources, S, cfg, logger=None, val_sets={},
                          cascade=frozen)
    extra = trainer.checkpoint_variables()
    if frozen is not None:
        extra = dict(extra, **frozen.checkpoint_extra())
    extra_best = {name: os.path.join(run_dir, "ckpt", "best_%s" % name) for name in trainer.extra_best} or None
    ckpt = C.RunCheckpoint(last_dir, os.path.join(run_dir, "ckpt", "best"), model, optimizer,
                           trainer.step, extra=extra, extra_best=extra_best)
    try:
        ckpt.checkpoint.restore(prefix).assert_existing_objects_matched()
    except (AssertionError, ValueError) as e:
        raise EvalError("run %s: restore of %s failed: %s" % (run_dir, prefix, e))
    if frozen is not None:
        try:
            if cascade.get("stages"):
                M._check_cascade_fingerprints(cascade, frozen, config_path_of(run_dir))
            else:
                M._check_cascade_fingerprint(cascade.get("fingerprint"), frozen.fingerprint,
                                             config_path_of(run_dir))
        except RuntimeError as e:
            raise EvalError(str(e))
    if ((raw_cfg.get("bn_recalibration") or {}).get("freeze")) is True:
        model.set_bn_stats_frozen(True)
        trainer.rebuild_steps()
    return {"trainer": trainer, "model": model, "frozen": frozen, "step": int(trainer.step.numpy()),
            "prefix": prefix, "ns": ns, "raw": raw_cfg, "run_dir": run_dir}


# ============================================================================ 3. validation spec
def val_spec(ns, data_root):
    """The settings that define a run's validation sets (runs with equal specs share sets)."""
    lin = A.linear_tasks(ns)
    deform = A.DEFORM_TASK in ns.task_ratio_parsed
    ranges = A.ranges_from_args(ns)
    return {"data": os.path.abspath(data_root), "level": int(ns.level), "linear_tasks": sorted(lin),
            "deform": bool(deform), "group_size": int(ns.group_size),
            "deform_group_size": int(ns.deform_group_size), "val_group_size": ns.val_group_size,
            "val_groups": 1 if ns.dry_run else ns.val_groups, "val_seed": int(ns.val_seed),
            "target_iters": int(ns.target_iters),
            "ranges": {t: ranges[t] for t in sorted(lin)},
            "global_transform": ({"rotation_deg": float(ns.global_rot_max_deg),
                                  "translation_mm": float(ns.global_trans_max_mm)}
                                 if ns.global_transform else None)}


def spec_key(spec):
    return json.dumps(spec, sort_keys=True)


def load_val_data(spec, fine=True):
    """Val images / labels of a spec: dict(lin_images, deform_images, labels18, fine)."""
    from runtime import fixed_data as fd
    root, level = spec["data"], spec["level"]
    out = {}
    if spec["linear_tasks"]:
        out["lin_images"], files = fd.load_level_images(root, level, "val")
        out["lin_files"] = files
    if spec["deform"] or fine:
        imgs, labs, files, _ = fd.load_level_images_and_labels(root, level, "val")
        out["deform_images"], out["labels18"], out["deform_files"] = imgs, labs, files
    if fine:
        try:
            _, fl, ffiles, _ = fd.load_level_images_and_labels(root, level, "val", label_set="synthfine")
        except ValueError as e:
            raise EvalError("synthetic evaluation needs the val synthfine labels (export-add-labels): %s" % e)
        if list(ffiles) != list(out["deform_files"]):
            raise EvalError("synthfine and train18 val selections differ in %s" % root)
        if "lin_files" in out and list(out["lin_files"]) != list(ffiles):
            raise EvalError("val image and label selections differ in %s" % root)
        out["fine"] = fl
    return out


def _nv(spec, n_val, deform=False):
    if spec["val_group_size"]:
        return int(spec["val_group_size"])
    return min(spec["deform_group_size"] if deform else spec["group_size"], int(n_val))


def real_val_sets(spec, data):
    """The validation sets of `main._build_dali_data` (same calls and seeds), without DALI."""
    from runtime import fixed_data as fd
    level, n_groups = spec["level"], spec["val_groups"]
    out = {}
    if spec["linear_tasks"]:
        imgs = data["lin_images"]
        out.update(fd.build_validation_sets(imgs, spec["linear_tasks"], level, _nv(spec, imgs.shape[0]),
                                            n_groups, spec["ranges"], seed=spec["val_seed"],
                                            target_iters=spec["target_iters"],
                                            global_transform=spec["global_transform"]))
    if spec["deform"]:
        imgs = data["deform_images"]
        out[A.DEFORM_TASK] = fd.DeformValidationSet.build(imgs, data["labels18"], level,
                                                          _nv(spec, imgs.shape[0], True), n_groups,
                                                          seed=spec["val_seed"])
    return out


# ============================================================================ 4. sanity gate
def logged_validation(run_dir, step):
    """{key: value} of the run's logged validation (``val*`` keys) at ``step`` from
    logs/metrics.jsonl (later records at the same step override earlier ones); {} if none."""
    path = os.path.join(run_dir, "logs", "metrics.jsonl")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("step") != step or not isinstance(rec.get("data"), dict):
                continue
            for k, v in rec["data"].items():
                if k.startswith("val") and isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[k] = float(v)
    return out


def _metric_kind(key):
    name = key.split("/")[-1]
    if "dice" in name:
        return "dice"
    if name.endswith("_mm"):
        return "mm"
    return None


def compare_validation(current, logged):
    """{max_mm_diff, max_dice_diff, n_compared, worst, pass} over the keys of both dicts."""
    mm = dice = 0.0
    worst = {"mm": None, "dice": None}
    n = 0
    for k in sorted(set(current) & set(logged)):
        kind = _metric_kind(k)
        if kind is None:
            continue
        d = abs(float(current[k]) - float(logged[k]))
        if d != d:                                # NaN on one side only
            d = float("inf")
        n += 1
        if kind == "mm" and d >= mm:
            mm, worst["mm"] = d, k
        elif kind == "dice" and d >= dice:
            dice, worst["dice"] = d, k
    ok = n > 0 and mm <= GATE_TOL_MM and dice <= GATE_TOL_DICE
    return {"max_mm_diff": mm, "max_dice_diff": dice, "n_compared": n, "worst": worst,
            "tol_mm": GATE_TOL_MM, "tol_dice": GATE_TOL_DICE, "pass": bool(ok)}


def sanity_gate(entry, val_sets):
    """`trainer.validate()` on the real sets vs the logged validation at the checkpoint step."""
    trainer = entry["trainer"]
    trainer.val_sets = dict(val_sets)
    with run_scope(entry):
        cur = trainer.validate()
    logged = logged_validation(entry["run_dir"], entry["step"])
    res = compare_validation(cur, logged)
    res["step"] = entry["step"]
    if not logged:
        res["reason"] = "no logged validation at step %d in logs/metrics.jsonl" % entry["step"]
    elif res["n_compared"] == 0:
        res["reason"] = "no common mm / Dice metric"
    return res


# ============================================================================ 5. synthetic sets
def _sha256_set(vs):
    """sha256 over every group's images (+ labels, T) in group order."""
    import numpy as np
    h = hashlib.sha256()
    for grp in vs.groups:
        for k in ("images", "labels", "T"):
            if k in grp:
                h.update(np.ascontiguousarray(grp[k]).tobytes())
    return h.hexdigest()


def synthetic_val_sets(preset, spec, data, synth_cfg):
    """{synth_<preset>_<real set name>: set} with the real sets' groups and transforms and
    images synthesised from the val synthfine labels (CPU, one contrast per group, seeded by
    (preset, task, group)) -> (sets, {name: sha256})."""
    import numpy as np
    import tensorflow as tf
    from data_loading import synth as SY
    from runtime import fixed_data as fd
    level, n_groups = spec["level"], spec["val_groups"]
    fine = data["fine"]

    def hook_for(task):
        memo = {}

        def fn(g, idx):
            key = int(g)
            if key not in memo:
                seed = TASK_SEED[task] * 100000 + key
                prm = SY.preset_params(preset, len(idx), seed, synth_cfg, level)
                memo[key] = SY.synthesize_group(fine[np.asarray(idx)], prm, synth_cfg, level).numpy()
            return memo[key]
        return fn

    sets = {}
    with tf.device("/cpu:0"):
        for t in spec["linear_tasks"]:
            imgs = data["lin_images"]
            built = fd.build_validation_sets(imgs, [t], level, _nv(spec, imgs.shape[0]), n_groups,
                                             spec["ranges"], seed=spec["val_seed"],
                                             target_iters=spec["target_iters"],
                                             global_transform=spec["global_transform"],
                                             group_images_fn=hook_for(t))
            for name, vs in built.items():
                sets[SYNTH_PREFIX + preset + "_" + name] = vs
        if spec["deform"]:
            imgs = data["deform_images"]
            sets[SYNTH_PREFIX + preset + "_" + A.DEFORM_TASK] = fd.DeformValidationSet.build(
                imgs, data["labels18"], level, _nv(spec, imgs.shape[0], True), n_groups,
                seed=spec["val_seed"], group_images_fn=hook_for(A.DEFORM_TASK))
    return sets, {k: _sha256_set(v) for k, v in sorted(sets.items())}


# ============================================================================ 6. driver
def _maxrss_kb():
    try:
        import resource
    except ImportError:                     # not on Unix
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _labels(run_dirs):
    out, seen = [], {}
    for r in run_dirs:
        base = os.path.basename(os.path.normpath(r)) or "run"
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else "%s#%d" % (base, seen[base]))
    return out


def _finite_or_str(v):
    import math
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return v if math.isfinite(v) else str(v)


def write_csv(path, report):
    """Paired table: preset, metric, one column per run, diff = last - first (2 runs)."""
    labels = [r["label"] for r in report["runs"]]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["preset", "metric"] + labels + (["diff"] if len(labels) == 2 else []))
        for preset in report["presets"]:
            res = report["results"].get(preset)
            if not res:
                continue
            per = [res["runs"][lb]["metrics"] for lb in labels]
            keys = sorted(set.intersection(*[set(m) for m in per]))
            for k in keys:
                vals = [per[i][k] for i in range(len(labels))]
                row = [preset, k] + vals
                if len(labels) == 2:
                    try:
                        row.append(float(vals[1]) - float(vals[0]))
                    except (TypeError, ValueError):
                        row.append("")
                w.writerow(row)


def _write_json(path, report):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def run_eval(runs, level, out_dir, presets, data_root=None, sanity_gate_on=True, quiet=False):
    """Evaluate ``runs`` (1 or 2 run dirs) at ``level`` on the synthetic ``presets`` -> exit code
    (0 OK, 5 failed sanity gate). ``data_root`` None = each run's saved ``args.data``."""
    import gc

    from data_loading import params as P
    from data_loading import synth as SY

    runs = [os.path.abspath(r) for r in runs]
    if not 1 <= len(runs) <= 2:
        raise EvalError("--runs takes 1 or 2 run directories, got %d" % len(runs))
    unknown = [p for p in presets if p not in SY.EVAL_PRESETS]
    if unknown:
        raise EvalError("unknown preset(s) %s (known: %s)" % (unknown, ", ".join(SY.PRESET_NAMES)))
    os.makedirs(out_dir, exist_ok=True)
    labels = _labels(runs)
    entries = []
    for r, lb in zip(runs, labels):
        ns, raw = load_run_config(r)
        if int(ns.level) != int(level):
            raise EvalError("run %s is level %s, --level is %d" % (r, ns.level, int(level)))
        root = data_root or ns.data
        if not root:
            raise EvalError("run %s has no saved args.data; pass --data" % r)
        e = build_trainer(r, ns, raw, root)
        e["label"] = lb
        e["spec"] = val_spec(ns, root)
        entries.append(e)
        if not quiet:
            print("eval_synth: loaded %s (level %d, step %d) from %s" % (lb, int(level), e["step"], e["prefix"]),
                  flush=True)
    specs = {}
    for e in entries:
        specs.setdefault(spec_key(e["spec"]), []).append(e)
    report = {"level": int(level), "presets": list(presets), "sanity_gate_on": bool(sanity_gate_on),
              "out_dir": os.path.abspath(out_dir),
              "runs": [{"label": e["label"], "run_dir": e["run_dir"], "checkpoint": e["prefix"],
                        "step": e["step"], "spec": e["spec"], "spec_index": list(specs).index(spec_key(e["spec"])),
                        "synth": e["raw"].get("synth")} for e in entries],
              "n_specs": len(specs), "sanity_gate": {}, "results": {}}
    json_path = os.path.join(out_dir, JSON_NAME)
    data = {}
    for key, group in specs.items():
        data[key] = load_val_data(group[0]["spec"], fine=True)
    if sanity_gate_on:
        failed = False
        for key, group in specs.items():
            real = real_val_sets(group[0]["spec"], data[key])
            for e in group:
                res = sanity_gate(e, real)
                report["sanity_gate"][e["label"]] = res
                failed = failed or not res["pass"]
                if not quiet:
                    print("eval_synth: sanity gate %s: max mm diff %.3g, max Dice diff %.3g over %d metrics -> %s"
                          % (e["label"], res["max_mm_diff"], res["max_dice_diff"], res["n_compared"],
                             "PASS" if res["pass"] else "FAIL"), flush=True)
            for e in group:
                e["trainer"].val_sets = {}
            del real
            gc.collect()
        if failed:
            report["exit_code"] = EXIT_GATE_FAILED
            _write_json(json_path, report)
            return EXIT_GATE_FAILED
    synth_cfg = P.SynthConfig()                 # training ranges (prob ignored by the presets)
    for preset in presets:
        t0 = time.perf_counter()
        res = {"runs": {}, "shared_sets": {}, "sha256": {}}
        for key, group in specs.items():
            sets, sha = synthetic_val_sets(preset, group[0]["spec"], data[key], synth_cfg)
            res["sha256"][str(list(specs).index(key))] = sha
            for e in group:
                e["trainer"].val_sets = sets           # the same set objects for every run
                with run_scope(e):
                    val = e["trainer"].validate()
                res["runs"][e["label"]] = {"metrics": {k: _finite_or_str(v) for k, v in val.items()},
                                           "sha256": sha}
                res["shared_sets"][e["label"]] = len(group) > 1
                e["trainer"].val_sets = {}
            del sets
            gc.collect()
        res["elapsed_s"] = time.perf_counter() - t0
        res["ru_maxrss_kb"] = _maxrss_kb()
        report["results"][preset] = res
        _write_json(json_path, report)
        if not quiet:
            print("eval_synth: preset %s done in %.1f s (ru_maxrss %s kB)" % (preset, res["elapsed_s"],
                                                                             res["ru_maxrss_kb"]), flush=True)
    report["exit_code"] = 0
    _write_json(json_path, report)
    write_csv(os.path.join(out_dir, CSV_NAME), report)
    return 0


def is_inside(path, root):
    """True iff ``path`` equals ``root`` or lies below it (real paths)."""
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


if __name__ == "__main__":                      # pragma: no cover
    sys.exit("use eval_synth.py")
