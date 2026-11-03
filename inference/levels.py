"""Pyramid-level selection of predict.py (predict-levels design v2, sections 2-7). TF-free.

``--levels`` picks a strictly descending, non-empty subset of the checkpoint's cascade levels
(`plan`). Levels finer than the last selected level are not run (truncation: the last run level's
fields are upsampled to level 1 with the cascade's own `diffeo.upsample_displacement`, matrices
pass unchanged). A coarser level that is not selected is skipped exactly as the trainer skips a
stage (runtime/run.py stage mode, `models.cascade.skip_fields` / `skip_linear` /
`FrozenChain.deform_skip` / `linear_skip`): it contributes the identity and the later stages still
run on the un-pre-warped inputs.

`run_skip_probs` reads the skip probabilities a run was trained with from its config.json, and
`exposure` decides per (task, run level) whether the selection's skips were seen in training
(design 6). ``joint`` = P(all levels in K skipped) = product of the per-level probabilities
(independent draws, `runtime.run.cascade_stage_skip_draw`); other frozen levels may also have been
skipped in those steps (``exact_pattern`` = the probability of exactly this pattern).
"""
import logging

logger = logging.getLogger(__name__)

LEVEL_VALUES = (4, 2, 1)
TASKS = ("rigid", "affine", "deform")
LOW_EXPOSURE = 0.05
KIND_TEXT = {"run": "run",
             "skipped": "skipped (identity; later stages on un-pre-warped inputs)"}


def parse_levels(text):
    """'4,2' -> (4, 2). ValueError unless every item is one of 4, 2, 1 (spaces allowed), no item is
    empty, the order is strictly descending (no duplicates) and at least one level is given."""
    if text is None or not str(text).strip():
        raise ValueError("--levels needs at least one level (e.g. 4 or 4,2 or 2,1)")
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            raise ValueError("--levels %r has an empty item" % text)
        try:
            lv = int(item)
        except ValueError:
            raise ValueError("--levels %r: %r is not one of 4, 2, 1" % (text, item))
        if lv not in LEVEL_VALUES:
            raise ValueError("--levels %r: %r is not one of 4, 2, 1" % (text, item))
        out.append(lv)
    for a, b in zip(out, out[1:]):
        if b >= a:
            raise ValueError("--levels %r must be strictly descending (coarse to fine), e.g. 4,2,1"
                             % text)
    return tuple(out)


def levels_text(levels):
    return None if levels is None else ",".join(str(int(lv)) for lv in levels)


class LevelPlan(object):
    """Which checkpoint levels run, are skipped or are truncated (design 4.1)."""

    def __init__(self, ckpt_levels, selected, default, requested=None):
        self.ckpt_levels = tuple(int(lv) for lv in ckpt_levels)
        self.selected = tuple(int(lv) for lv in selected)
        self.default = bool(default)
        self.requested = requested
        self.last = self.selected[-1]
        self.run = self.selected
        self.truncated = tuple(lv for lv in self.ckpt_levels if lv < self.last)
        self.skipped = tuple(lv for lv in self.ckpt_levels if lv > self.last and lv not in self.selected)

    def skipped_before(self, level):
        """Skipped levels coarser than ``level``."""
        return tuple(lv for lv in self.skipped if lv > int(level))

    def kind(self, level):
        level = int(level)
        if level in self.selected:
            return "run"
        if level in self.skipped:
            return "skipped"
        if level in self.truncated:
            return "truncated"
        raise ValueError("level %d is not a level of this checkpoint %s" % (level, list(self.ckpt_levels)))

    def describe(self, level):
        k = self.kind(level)
        if k == "truncated":
            return "not run (truncated; fields upsampled from level %d)" % self.last
        return KIND_TEXT[k]

    def to_dict(self):
        return {"requested": self.requested, "checkpoint_levels": list(self.ckpt_levels),
                "selected": list(self.selected), "default": self.default, "last": self.last,
                "skipped": list(self.skipped), "truncated": list(self.truncated),
                "per_level": {str(lv): self.describe(lv) for lv in self.ckpt_levels},
                "upsampled_from_level": self.last}

    def __repr__(self):
        return "LevelPlan(%s)" % ", ".join("L%d %s" % (lv, self.kind(lv)) for lv in self.ckpt_levels)


def plan(ckpt_levels, selected, normalise=True):
    """LevelPlan of ``selected`` (tuple, None = all) on ``ckpt_levels`` (coarse to fine).

    ``default`` is True iff ``normalise`` and the selection equals all checkpoint levels (then the
    runners take the unchanged default path); ``normalise=False`` is for tests (a full plan through
    the general path). ValueError for a level that is not one of the checkpoint's levels."""
    ckpt = tuple(int(lv) for lv in ckpt_levels)
    requested = levels_text(selected)
    sel = ckpt if selected is None else tuple(int(lv) for lv in selected)
    if not sel:
        raise ValueError("no level selected")
    bad = [lv for lv in sel if lv not in ckpt]
    if bad:
        raise ValueError("--levels %s: the checkpoint is a level-%d run with levels %s"
                         % (requested, ckpt[-1], list(ckpt)))
    if list(sel) != sorted(sel, reverse=True) or len(set(sel)) != len(sel):
        raise ValueError("--levels %s must be strictly descending" % requested)
    return LevelPlan(ckpt, sel, default=bool(normalise) and sel == ckpt, requested=requested)


# ============================================================================ exposure (6)
def _get(d, level):
    """d[level] with int or string keys; None if absent."""
    if not isinstance(d, dict):
        return None
    for key in (level, str(level)):
        if key in d:
            return d[key]
    return None


def run_skip_probs(config):
    """Skip probabilities a run was trained with (design 6.2), from config.json ``args``.

    Returns {'level': run level, 'probs': {task: {'4': p, '2': p}} over the run's frozen levels,
    'source', 'skip_mode' ('legacy' | 'stage' | 'off'), 'record' (cascade.skip_probs or
    cascade.skip_prob, reference only), 'args_vs_record_mismatch' (bool)}."""
    args = config.get("args") or {}
    level = int(args["level"])
    frozen = [lv for lv in LEVEL_VALUES if lv > level]
    casc = config.get("cascade") or {}
    legacy = float(args.get("cascade_skip_prob") or 0.0)
    probs = {t: {str(lv): 0.0 for lv in frozen} for t in TASKS}
    if legacy > 0:
        mode, source = "legacy", "args.cascade_skip_prob (legacy: skips the last frozen level)"
        skip_lv = 4 if level == 2 else 2
        for t in TASKS:
            if str(skip_lv) in probs[t]:
                probs[t][str(skip_lv)] = legacy
        record = casc.get("skip_prob")
        mismatch = record is not None and float(record) != legacy
    else:
        parsed = args.get("cascade_skip_stage_parsed")
        if parsed is None:
            source = "absent (run predates per-stage skip)"
        else:
            source = "args.cascade_skip_stage_parsed"
        for t in TASKS:
            d = (parsed or {}).get(t) or {}
            for lv in frozen:
                p = _get(d, lv)
                probs[t][str(lv)] = 0.0 if p is None else float(p)
        on = any(p > 0 for d in probs.values() for p in d.values())
        mode = "stage" if on else "off"
        record = casc.get("skip_probs")
        mismatch = False
        if mode == "stage" and isinstance(record, dict):
            for t, d in record.items():
                for lv in frozen:
                    rp = _get(d, lv)
                    if rp is not None and t in probs and float(rp) != probs[t][str(lv)]:
                        mismatch = True
    return {"level": level, "probs": probs, "source": source, "skip_mode": mode,
            "record": record, "args_vs_record_mismatch": bool(mismatch)}


def exposure(level_plan, tasks, run_configs, low_threshold=LOW_EXPOSURE):
    """Training exposure of the plan's skips (design 6.3).

    tasks: the branch's model tasks. run_configs: {run level: (config path or None, config dict or
    None)} of the checkpoint chain (`predict._peek_chain_configs`). Returns {'low_threshold',
    'runs' {level: run record}, 'entries' [per (task, run level) with skipped coarser levels],
    'min_joint' {task: min joint over known entries}}; ``forced`` is set by the caller. Logs the
    args vs record WARNING for runs that an entry uses."""
    runs, parsed = {}, {}
    for lv, (path, config) in sorted(run_configs.items(), reverse=True):
        rec = {"config": path}
        if config is None:
            rec.update(source="unknown (config.json not readable)", probs=None, skip_mode=None)
        else:
            try:
                info = run_skip_probs(config)
            except (KeyError, TypeError, ValueError) as e:
                info = None
                rec.update(source="unknown (bad config.json: %s)" % e, probs=None, skip_mode=None)
            if info is not None:
                parsed[lv] = info
                rec.update(source=info["source"], probs=info["probs"], skip_mode=info["skip_mode"],
                           record=info["record"],
                           args_vs_record_mismatch=info["args_vs_record_mismatch"])
        runs[str(lv)] = rec
    entries, warned = [], set()
    for s in level_plan.run:
        K = level_plan.skipped_before(s)
        if not K:
            continue
        info = parsed.get(s)
        path = runs.get(str(s), {}).get("config")
        if info is not None and info["args_vs_record_mismatch"] and s not in warned:
            warned.add(s)
            logger.warning("run %s: config skip flags (args) differ from cascade.skip_probs (changed on "
                           "resume?); exposure uses args", path)
        for t in tasks:
            e = {"task": t, "level": int(s), "skipped": list(K), "run_config": path}
            if info is None:
                e.update(p=None, joint=None, exact_pattern=None, status="unknown")
            else:
                p = {str(lv): float(info["probs"][t].get(str(lv), 0.0)) for lv in K}
                joint = 1.0
                for v in p.values():
                    joint *= v
                exact = joint
                for m, pm in info["probs"][t].items():
                    if int(m) not in K and level_plan.kind(int(m)) == "run":
                        exact *= 1.0 - float(pm)
                e.update(p=p, joint=joint, exact_pattern=exact,
                         status="trained" if joint > 0 else "untrained")
            entries.append(e)
    min_joint = {}
    for e in entries:
        if e["joint"] is not None:
            cur = min_joint.get(e["task"])
            min_joint[e["task"]] = e["joint"] if cur is None else min(cur, e["joint"])
    return {"forced": False, "low_threshold": float(low_threshold), "runs": runs, "entries": entries,
            "min_joint": min_joint}


def entry_text(e):
    return ("level %d, task %s, skipped levels %s, p %s, run config %s"
            % (e["level"], e["task"], e["skipped"], e["p"], e["run_config"]))


def bad_entries(exp):
    """Entries with status 'untrained' or 'unknown'."""
    return [e for e in exp["entries"] if e["status"] in ("untrained", "unknown")]
