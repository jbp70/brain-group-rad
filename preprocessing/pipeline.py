"""Per-row pipeline: native image -> skull-stripped, [0,1]-normalized image on the
LUMIR target grid (MNI152 world), with mask and labels.

Steps (ALL_STEPS, in order; validation always runs):
  reorient   flirt backend only: canonical reorientation (nibabel, world-preserving) or fslreorient2std
  strip      provided mask | derive_mask (already stripped) | mri_synthstrip
  n4         N4BiasFieldCorrection inside the mask (skipped if bias_corrected)
  register   9-DOF FLIRT (default) or ANTs Rigid+Affine/Similarity to MNI152 1 mm (skipped if in_template_space)
  resample   ONE interpolation native -> target grid (anti-alias smoothing for sub-mm inputs first);
             flirt matrices are composed in world space, ANTs uses template_lumirgrid as reference
  normalize  in-mask 99.5th percentile -> [0,1] (numpy), write image/labels (and mask.nii.gz only with
             outputs.save_mask) with the target affine
  synthseg   optional per-row SynthSeg (the batch `synthseg` command is the main path)
  qc         qc.json (+ qc.png only with outputs.save_qc_png)

Storage: once a row is "completed" with every requested step done, work/ is
deleted (outputs.keep_work: false; `--keep-work` keeps it, `--clean-work`
forces deletion even if the config keeps it). Partial/failed rows keep work/
so they can resume.

LUMIR rows (in_template_space): voxel arrays are never resampled when the
shape matches and the header is the recorded raw LUMIR header (or already the
corrected affine); outputs get the corrected affine from target_grid.json.

Every step is skipped when its outputs exist AND meta.json records it as
done (unless --overwrite); a step is recorded "running" before it starts, so
outputs left by a killed job are redone. Step records include the exact
argv of every tool call. Dry-run
prints ``CMD ...`` / ``PY ...`` lines and writes nothing.

    python -m preprocessing run --config cfg.yaml --manifest oasis.csv --out-root /data/processed --start 0 --end 50
    python -m preprocessing run --config cfg.yaml --manifest oasis.csv --out-root /data/processed --rows 0 --dry-run
"""
import json
import logging
import os
import platform
import shutil
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

from . import __version__, nifti
from .commands import (ants_apply_cmd, ants_registration_cmd, flirt_apply_cmd, flirt_register_cmd,
                       fslreorient2std_cmd, n4_cmd, synthstrip_cmd)
from .config import mni_template
from .grid import Target
from .intensity import antialias_sigma_vox, antialias_smooth, normalize_intensity
from .labels import resample_labels_nearest, to_label_dtype
from .manifest import ManifestRow, subject_dir_name
from .masks import derive_mask
from .meta import load_meta, now, outputs_exist, record_step, write_meta_atomic
from .tools import ToolError, ToolRunner

log = logging.getLogger(__name__)

ALL_STEPS = ["reorient", "strip", "n4", "register", "resample", "normalize", "synthseg", "qc"]
DEFAULT_STEPS = [s for s in ALL_STEPS if s != "synthseg"]
WORK_STEPS = ("reorient", "strip", "n4", "resample")


class PipelineError(RuntimeError):
    pass


def _public_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


class RowContext:
    """State and paths for one manifest row."""

    def __init__(self, row: ManifestRow, cfg: Dict[str, Any], out_root: str, runner: ToolRunner,
                 target: Target, overwrite: bool, echo: Callable[[str], None]):
        self.row = row
        self.cfg = cfg
        self.runner = runner
        self.target = target
        self.overwrite = overwrite
        self.echo = echo
        self.dry = runner.dry_run
        self.D = os.path.join(os.path.abspath(out_root), row.dataset, subject_dir_name(row))
        self.work = os.path.join(self.D, "work")
        self.logs = os.path.join(self.D, "logs")
        self.meta_path = os.path.join(self.D, "meta.json")
        self.meta = load_meta(self.meta_path)
        self.backend = cfg["registration"]["backend"]
        self.commands: Dict[str, List[List[str]]] = {}  # step -> argv lists run this time (meta.json)

    # ---- paths
    def w(self, name: str) -> str:
        return os.path.join(self.work, name)

    def o(self, name: str) -> str:
        return os.path.join(self.D, name)

    def log_for(self, step: str) -> str:
        return os.path.join(self.logs, "%s.log" % step)

    @property
    def reorient_applies(self) -> bool:
        return (self.backend == "flirt" and self.cfg["registration"]["reorient"] != "none"
                and not self.row.in_template_space)

    @property
    def src_image(self) -> str:
        return self.w("reoriented.nii.gz") if self.reorient_applies else self.row.image_path

    @property
    def reg_input(self) -> str:
        return self.w("stripped.nii.gz") if self.row.bias_corrected else self.w("n4.nii.gz")

    def applies(self, step: str) -> bool:
        if step == "reorient":
            return self.reorient_applies
        if step == "n4":
            return not self.row.bias_corrected
        if step == "register":
            return not self.row.in_template_space
        return True

    # ---- outputs per step
    def resampled_labels(self) -> Dict[str, str]:
        return {n: self.w("labels_%s_resampled.nii.gz" % n) for n in self.row.label_paths}

    def outputs(self, step: str) -> List[str]:
        if step == "reorient":
            return [self.w("reoriented.nii.gz")]
        if step == "strip":
            return [self.w("stripped.nii.gz"), self.w("native_mask.nii.gz")]
        if step == "n4":
            return [self.w("n4.nii.gz")]
        if step == "register":
            outs = [self.o("affine_to_template.mat")]
            if self.backend == "flirt":
                outs.append(self.o("affine_to_template_world.txt"))
            return outs
        if step == "resample":
            if self.row.in_template_space and self.meta.get("grid_state") != "mismatch":
                return []
            return [self.w("image_resampled.nii.gz"), self.w("mask_resampled.nii.gz")] + \
                list(self.resampled_labels().values())
        if step == "normalize":
            return [self.o("image.nii.gz")] + ([self.o("mask.nii.gz")] if self.save_mask else []) + \
                [self.o("labels_%s.nii.gz" % n) for n in self.row.label_paths]
        if step == "synthseg":
            return [self.o("labels_synthseg.nii.gz")]
        if step == "qc":
            return [self.o("qc.json")] + ([self.o("qc.png")] if self.cfg["outputs"]["save_qc_png"] else [])
        raise ValueError(step)

    @property
    def save_mask(self) -> bool:
        return bool(self.cfg["outputs"]["save_mask"])

    def recorded_done(self, step: str) -> bool:
        """meta.json says the step finished (it is marked 'running' before it starts)."""
        return self.meta.get("steps", {}).get(step, {}).get("status") == "done"

    def is_done(self, step: str) -> bool:
        """Outputs exist AND the step was recorded as done.

        Requiring the record means a file left half-written by a killed job
        (e.g. SLURM time limit) is redone on the next run instead of trusted.
        """
        if not self.applies(step):
            return True
        outs = self.outputs(step)
        if not outs:
            return True
        if not self.recorded_done(step):
            return False
        if outputs_exist(outs):
            return True
        if self.meta.get("work_cleaned") and step in WORK_STEPS:
            return self.recorded_done("normalize") and outputs_exist(self.outputs("normalize"))
        return False

    def need(self, *paths: str) -> None:
        if self.dry:
            return
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            hint = "run the earlier steps first"
            if self.meta.get("work_cleaned") and any(os.path.dirname(p) == self.work for p in missing):
                hint = ("work/ was deleted after the row completed; rerun all steps with --overwrite "
                        "(add --keep-work to keep intermediates)")
            raise PipelineError("missing inputs %s (%s)" % (missing, hint))

    def py(self, step: str, inputs: Sequence[str], outputs: Sequence[str]) -> None:
        if self.dry:
            self.echo("PY %s: %s -> %s" % (step, ", ".join(inputs), ", ".join(outputs)))

    def run_tool(self, argv: List[str], step: str) -> None:
        self.commands.setdefault(step, []).append([str(a) for a in argv])
        self.runner.run(argv, self.log_for(step))


# ------------------------------------------------------------------ steps
def step_validate(c: RowContext) -> None:
    r = c.row
    if "synthseg" in r.label_paths:
        raise PipelineError("label name 'synthseg' is reserved")
    if not os.path.isfile(r.image_path):
        raise PipelineError("image not found: %s" % r.image_path)
    img = nifti.load(r.image_path)
    shape = nifti.spatial_shape(img)
    info: Dict[str, Any] = {"shape": list(shape), "spacing": [float(x) for x in nifti.voxel_sizes(img.affine)],
                            "affine": img.affine.tolist(), "qform_sform_consistent": nifti.qform_sform_consistent(img)}
    if not info["qform_sform_consistent"]:
        log.warning("%s: qform and sform disagree; using nibabel's choice (sform)", r.image_path)
    if r.mask_path and not os.path.isfile(r.mask_path):
        raise PipelineError("mask not found: %s" % r.mask_path)
    for name, p in r.label_paths.items():
        if not os.path.isfile(p):
            raise PipelineError("label %s not found: %s" % (name, p))
    if r.in_template_space:
        state = c.target.state(shape, img.affine)
        c.meta["grid_state"] = state
        c.meta["grid_mismatch"] = state == "mismatch"
        c.meta["original_affine"] = img.affine.tolist()
        c.meta["corrected_affine"] = c.target.affine.tolist()
        if state == "mismatch":
            log.warning("%s: not on the LUMIR grid (shape %s); will resample in world space with its own "
                        "header -- only valid if that header is world-consistent", r.image_path, shape)
        if r.dataset == "lumir" and tuple(shape) != tuple(c.cfg["lumir_expected_shape"]):
            log.warning("%s: shape %s != lumir_expected_shape", r.image_path, shape)
    c.meta["input"] = info


def step_reorient(c: RowContext) -> None:
    out = c.w("reoriented.nii.gz")
    how = c.cfg["registration"]["reorient"]
    if how == "fslreorient2std":
        c.run_tool(fslreorient2std_cmd(c.runner.tool(c.cfg, "fslreorient2std"), c.row.image_path, out), "reorient")
        return
    c.py("reorient", [c.row.image_path], [out])
    if c.dry:
        return
    img = nib.as_closest_canonical(nifti.load(c.row.image_path))
    data = np.asanyarray(img.dataobj).reshape(nifti.spatial_shape(img))
    nifti.save_like(data.astype(np.float32), img.affine, out, np.float32, xform_code=1)


def _on_grid(data: np.ndarray, data_img, ref_img) -> np.ndarray:
    """Return data on ref_img's grid (exact nearest lookup, for masks/labels)."""
    if nifti.img_grid_matches(data_img, ref_img):
        return data
    log.warning("resampling %s onto the image grid (nearest, world space)", data_img.get_filename())
    return resample_labels_nearest(data, data_img.affine, nifti.spatial_shape(ref_img), ref_img.affine)


def step_strip(c: RowContext) -> None:
    r = c.row
    src = c.src_image
    stripped, mask_out = c.w("stripped.nii.gz"), c.w("native_mask.nii.gz")
    if not r.mask_path and not r.skull_stripped:
        c.need(src)
        argv = synthstrip_cmd(c.runner.tool(c.cfg, "mri_synthstrip"), src, stripped, mask_out,
                              c.cfg["synthstrip"]["extra_args"])
        c.run_tool(argv, "strip")
        c.meta["mask_source"] = "synthstrip"
        return
    how = "provided" if r.mask_path else "derived"
    c.py("strip(%s mask)" % how, [src] + ([r.mask_path] if r.mask_path else []), [stripped, mask_out])
    c.meta["mask_source"] = how
    if c.dry:
        return
    c.need(src)
    data, img = nifti.load_data(src)
    if r.mask_path:
        mdata, mimg = nifti.load_data(r.mask_path)
        mask = _on_grid(mdata, mimg, img) > 0
    else:
        extra = None
        if r.label_paths:
            ldata, limg = nifti.load_data(next(iter(r.label_paths.values())))
            extra = _on_grid(ldata, limg, img)
        mc = c.cfg["mask"]
        mask = derive_mask(data, mc["threshold"], mc["fill_holes"], mc["largest_component"], extra=extra)
    if not mask.any():
        raise PipelineError("empty brain mask")
    nifti.save_like(data * mask, img.affine, stripped, np.float32, xform_code=1)
    nifti.save_like(mask.astype(np.uint8), img.affine, mask_out, np.uint8, xform_code=1)


def step_n4(c: RowContext) -> None:
    c.need(c.w("stripped.nii.gz"), c.w("native_mask.nii.gz"))
    argv = n4_cmd(c.runner.tool(c.cfg, "N4BiasFieldCorrection"), c.w("stripped.nii.gz"),
                  c.w("native_mask.nii.gz"), c.w("n4.nii.gz"), c.cfg["n4"])
    c.run_tool(argv, "n4")


def _reg_template(c: RowContext) -> Dict[str, Optional[str]]:
    t = mni_template(c.cfg, c.row.contrast, c.runner.env)
    if not c.dry and not os.path.isfile(t["image_1mm"]):
        raise PipelineError("MNI template not found: %s" % t["image_1mm"])
    return t


def step_register(c: RowContext) -> None:
    moving = c.reg_input
    c.need(moving)
    t = _reg_template(c)
    mat = c.o("affine_to_template.mat")
    if c.backend == "ants":
        ac = c.cfg["registration"]["ants"]
        masks_used = bool(ac.get("use_masks") and t.get("mask_1mm"))
        if ac.get("use_masks") and not t.get("mask_1mm"):
            log.warning("%s: registration.ants.use_masks is true but the MNI template has no mask_1mm; "
                        "running antsRegistration WITHOUT -x masks", c.D)
        argv = ants_registration_cmd(c.runner.tool(c.cfg, "antsRegistration"), t["image_1mm"], moving,
                                     c.w("reg_"), ac, fixed_mask=t.get("mask_1mm"),
                                     moving_mask=c.w("native_mask.nii.gz"))
        if not c.dry:
            os.makedirs(c.work, exist_ok=True)
        c.run_tool(argv, "register")
        c.py("register(copy)", [c.w("reg_0GenericAffine.mat")], [mat])
        if not c.dry:
            shutil.copyfile(c.w("reg_0GenericAffine.mat"), mat)
        c.meta["registration"] = {"backend": "ants", "transform": ac["transform"], "fixed": t["image_1mm"],
                                  "matrix_format": "itk", "masks_used": masks_used}
        return
    fc = c.cfg["registration"]["flirt"]
    ref = t["image_1mm"]
    if fc["estimate_at_2mm"] and t.get("image_2mm") and (c.dry or os.path.isfile(t["image_2mm"])):
        ref = t["image_2mm"]
    omat = c.w("flirt_reg.mat")
    if not c.dry:
        os.makedirs(c.work, exist_ok=True)
    c.run_tool(flirt_register_cmd(c.runner.tool(c.cfg, "flirt"), moving, ref, omat, fc), "register")
    world_txt = c.o("affine_to_template_world.txt")
    c.py("register(compose world matrix)", [omat, moving, ref], [world_txt, mat])
    if c.dry:
        return
    m = nifti.read_matrix_txt(omat)
    mov_img, ref_img = nifti.load(moving), nifti.load(ref)
    world = nifti.flirt_to_world(m, nifti.spatial_shape(mov_img), mov_img.affine, nifti.spatial_shape(ref_img),
                                 ref_img.affine, mov_img.header.get_zooms()[:3], ref_img.header.get_zooms()[:3])
    to_target = nifti.world_to_flirt(world, nifti.spatial_shape(mov_img), mov_img.affine, c.target.shape,
                                     c.target.affine, mov_img.header.get_zooms()[:3],
                                     c.target.img.header.get_zooms()[:3])
    nifti.write_matrix_txt(world, world_txt)
    nifti.write_matrix_txt(to_target, mat)
    c.meta["registration"] = {"backend": "flirt", "dof": fc["dof"], "reference": ref,
                              "matrix_format": "fsl (registration input -> template_lumirgrid)",
                              "world_matrix_ras": world.tolist()}


def _flirt_apply(c: RowContext, moving: str, out: str, interp: str, name: str, is_image: bool) -> None:
    mat = c.w("flirt_apply_%s.mat" % name)
    c.py("resample(compose flirt matrix %s)" % name, [c.o("affine_to_template_world.txt"), moving], [mat])
    if not c.dry:
        world = nifti.read_matrix_txt(c.o("affine_to_template_world.txt"))
        img = nifti.load(moving)
        m = nifti.world_to_flirt(world, nifti.spatial_shape(img), img.affine, c.target.shape, c.target.affine,
                                 img.header.get_zooms()[:3], c.target.img.header.get_zooms()[:3])
        nifti.write_matrix_txt(m, mat)
    c.run_tool(flirt_apply_cmd(c.runner.tool(c.cfg, "flirt"), moving, c.target.template_path, mat, out, interp,
                               float_output=is_image), "resample")


def step_resample(c: RowContext) -> None:
    r = c.row
    img_out, mask_out = c.w("image_resampled.nii.gz"), c.w("mask_resampled.nii.gz")
    if r.in_template_space:
        if c.meta.get("grid_state") != "mismatch":
            return
        c.py("resample(world, own header)", [c.w("stripped.nii.gz"), c.w("native_mask.nii.gz")],
             [img_out, mask_out] + list(c.resampled_labels().values()))
        if c.dry:
            return
        for src, out, order in ((c.w("stripped.nii.gz"), img_out, 1), (c.w("native_mask.nii.gz"), mask_out, 0)):
            data, img = nifti.load_data(src)
            res = nifti.resample_world(data, img.affine, c.target.shape, c.target.affine, order=order)
            nifti.save_like(res, c.target.affine, out, np.float32, xform_code=4)
        for name, out in c.resampled_labels().items():
            data, img = nifti.load_data(r.label_paths[name], np.float64)
            res = resample_labels_nearest(data, img.affine, c.target.shape, c.target.affine)
            nifti.save_like(res, c.target.affine, out, np.float32, xform_code=4)
        return
    moving = c.reg_input
    c.need(moving, c.w("native_mask.nii.gz"), c.o("affine_to_template.mat"))
    rc = c.cfg["resample"]
    image_src = moving
    if rc["antialias"]:
        if os.path.isfile(moving):
            spacing = [float(x) for x in nifti.voxel_sizes(nifti.load(moving).affine)]
        else:  # dry-run before the input exists: use the raw image spacing
            spacing = list(c.meta["input"]["spacing"])
        sig = antialias_sigma_vox(spacing, rc["target_spacing_mm"], rc["min_spacing_ratio"])
        if np.any(sig > 0):
            image_src = c.w("smoothed.nii.gz")
            c.meta["antialias"] = {"sigma_vox": [float(s) for s in sig], "spacing": list(spacing),
                                   "target_spacing_mm": rc["target_spacing_mm"]}
            c.py("resample(antialias sigma_vox=%s)" % [round(float(s), 4) for s in sig], [moving], [image_src])
            if not c.dry:
                data, img = nifti.load_data(moving)
                sm, _, _ = antialias_smooth(data, spacing, rc["target_spacing_mm"], rc["min_spacing_ratio"])
                nifti.save_like(sm, img.affine, image_src, np.float32, xform_code=1)
        else:
            c.meta["antialias"] = None
    jobs = [(image_src, img_out, "image", True), (c.w("native_mask.nii.gz"), mask_out, "mask", False)]
    for name, out in c.resampled_labels().items():
        jobs.append((r.label_paths[name], out, "labels_" + name, False))
    if c.backend == "flirt":
        interp_img = c.cfg["registration"]["flirt"]["image_interp"]
        for src, out, name, is_img in jobs:
            _flirt_apply(c, src, out, interp_img if is_img else "nearestneighbour", name, is_img)
    else:
        interp_img = c.cfg["registration"]["ants"]["image_interp"]
        tool = c.runner.tool(c.cfg, "antsApplyTransforms")
        for src, out, name, is_img in jobs:
            c.run_tool(ants_apply_cmd(tool, src, c.target.template_path, out,
                                      interp_img if is_img else "GenericLabel", c.o("affine_to_template.mat")),
                       "resample")


def _sources(c: RowContext) -> Tuple[str, str, Dict[str, str]]:
    r = c.row
    if r.in_template_space and c.meta.get("grid_state") != "mismatch":
        return c.w("stripped.nii.gz"), c.w("native_mask.nii.gz"), dict(r.label_paths)
    return c.w("image_resampled.nii.gz"), c.w("mask_resampled.nii.gz"), c.resampled_labels()


def _to_target(c: RowContext, data: np.ndarray, img, what: str, labels: bool) -> np.ndarray:
    """Data on the target grid: stamped (no resampling) if on grid, else error/nearest fallback."""
    state = c.target.state(data.shape, img.affine)
    if state in ("target", "header"):
        return data
    if labels:
        log.warning("%s: %s grid differs from target; nearest-resampling", c.D, what)
        c.meta.setdefault("label_nearest_fallback", []).append(what)
        return resample_labels_nearest(data, img.affine, c.target.shape, c.target.affine)
    raise PipelineError("%s is not on the target grid (shape %s); tool output mismatch" % (what, data.shape))


def step_normalize(c: RowContext) -> None:
    img_src, mask_src, labels = _sources(c)
    outs = c.outputs("normalize")
    c.py("normalize", [img_src, mask_src] + list(labels.values()), outs)
    if c.dry:
        return
    c.need(img_src, mask_src, *labels.values())
    data, img = nifti.load_data(img_src)
    mdata, mimg = nifti.load_data(mask_src)
    data = _to_target(c, data, img, "image", False)
    mask = _to_target(c, mdata, mimg, "mask", False) > 0.5
    nc = c.cfg["normalization"]
    y, stats = normalize_intensity(data, mask, nc["upper_percentile"], nc["lower_percentile"])
    c.meta["normalization"] = stats
    aff = c.target.affine
    nifti.save_like(y, aff, c.o("image.nii.gz"), np.float32, xform_code=4)
    # The mask is used above; saving it is optional (later steps use image > 0 when absent).
    c.meta["mask_saved"] = c.save_mask
    if c.save_mask:
        nifti.save_like(mask.astype(np.uint8), aff, c.o("mask.nii.gz"), np.uint8, xform_code=4)
    elif os.path.isfile(c.o("mask.nii.gz")):
        os.remove(c.o("mask.nii.gz"))  # stale mask of an earlier run would be preferred by QC/export
    for name, p in labels.items():
        ldata, limg = nifti.load_data(p, np.float64)
        lab = to_label_dtype(_to_target(c, ldata, limg, "labels_" + name, True), "uint16")
        nifti.save_like(lab, aff, c.o("labels_%s.nii.gz" % name), np.uint16, xform_code=4)


def step_synthseg(c: RowContext) -> None:
    from .synthseg import run_synthseg_batch
    # work/ is cleaned by process_row once every step of the row is done.
    counts = run_synthseg_batch([c.D], c.cfg, c.runner, c.w("synthseg_batch"), c.target, overwrite=True,
                                echo=c.echo, clean_work=False)
    if counts.get("failed"):
        raise PipelineError("SynthSeg postprocess failed")
    if not c.dry:
        c.meta.update({k: v for k, v in load_meta(c.meta_path).items() if k in ("steps", "synthseg_resampled")})


def step_qc(c: RowContext) -> None:
    c.py("qc", [c.o("image.nii.gz")], c.outputs("qc"))
    if c.dry:
        return
    c.need(c.o("image.nii.gz"))
    from .synthseg import refresh_qc
    refresh_qc(c.D, c.target, c.cfg, c.meta)


STEP_FUNCS = {"reorient": step_reorient, "strip": step_strip, "n4": step_n4, "register": step_register,
              "resample": step_resample, "normalize": step_normalize, "synthseg": step_synthseg, "qc": step_qc}


# ------------------------------------------------------------------ row driver
def process_row(row: ManifestRow, cfg: Dict[str, Any], out_root: str, steps: Sequence[str], runner: ToolRunner,
                target: Target, overwrite: bool = False, clean_work: Optional[bool] = None,
                tool_versions: Optional[Dict[str, Any]] = None, echo: Optional[Callable[[str], None]] = None,
                row_index: Optional[int] = None) -> str:
    """Process one row; returns 'completed', 'partial', 'skipped', 'dry-run' or 'failed'.

    clean_work (default: not outputs.keep_work): delete work/ when the row
    ends "completed" with every requested step done (incl. synthseg when it
    is in ``steps``). Nothing later needs work/: resume checks trust the
    normalize outputs (meta work_cleaned) and the SynthSeg batch reads image.nii.gz.
    """
    echo = echo or print
    if clean_work is None:
        clean_work = not cfg["outputs"]["keep_work"]
    c = RowContext(row, cfg, out_root, runner, target, overwrite, echo)
    if c.dry:
        echo("# row %s: %s -> %s" % (row_index, row.image_path, c.D))
    handler = None
    current: Optional[str] = None  # step marked "running" in meta.json, if any
    try:
        step_validate(c)
        todo = [s for s in ALL_STEPS if s in steps and c.applies(s)]
        if (not overwrite and not c.dry and c.meta.get("status") == "completed"
                and all(c.is_done(s) for s in todo)):
            log.info("%s: complete, skipping", c.D)
            return "skipped"
        if not c.dry:
            os.makedirs(c.logs, exist_ok=True)
            os.makedirs(c.work, exist_ok=True)
            handler = logging.FileHandler(os.path.join(c.logs, "pipeline.log"))
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logging.getLogger("preprocessing").addHandler(handler)
            c.meta.update({
                "row": row.to_dict(), "row_index": row_index, "status": "running", "error": None,
                "config_snapshot": _public_cfg(cfg), "config_path": cfg.get("_config_path"),
                "tool_versions": tool_versions or {}, "package_version": __version__,
                "python": sys.version.split()[0], "platform": platform.platform(),
                "numpy": np.__version__, "nibabel": nib.__version__,
                "target": {"grid": target.grid.get("outputs", {}).get("grid"), "template": target.template_path,
                           "template_mask": target.mask_path, "template_sha256": target.sha256(),
                           "affine": target.affine.tolist()},
                "registration_backend": c.backend,
            })
            write_meta_atomic(c.meta, c.meta_path)
        for s in todo:
            if not overwrite and c.is_done(s):
                log.info("%s: step %s done, skipping", c.D, s)
                if c.dry:
                    echo("# skip %s (outputs exist)" % s)
                continue
            started = now()
            if not c.dry and s != "synthseg":
                # Mark as running first: if the job dies mid-write, the step is not trusted on rerun.
                record_step(c.meta, s, "running", outputs=c.outputs(s), started=started)
                write_meta_atomic(c.meta, c.meta_path)
                current = s
            STEP_FUNCS[s](c)
            if not c.dry and s in WORK_STEPS:
                c.meta.pop("work_cleaned", None)  # work/ holds fresh intermediates again
            if not c.dry and s != "synthseg":
                record_step(c.meta, s, "done", outputs=c.outputs(s), commands=c.commands.get(s),
                            params=_step_params(c, s), started=started,
                            log=c.log_for(s) if os.path.isfile(c.log_for(s)) else None)
                write_meta_atomic(c.meta, c.meta_path)
                current = None
        if c.dry:
            return "dry-run"
        # "completed" only when every default step is done; a --steps subset leaves "partial".
        full = all(c.is_done(s) for s in DEFAULT_STEPS)
        c.meta["status"] = "completed" if full else "partial"
        if clean_work and full and all(c.is_done(s) for s in todo):
            from .synthseg import clean_work_dir
            clean_work_dir(c.D, c.meta)
        write_meta_atomic(c.meta, c.meta_path)
        return c.meta["status"]
    except Exception as e:  # row failure policy: log, mark failed, continue
        log.error("%s: FAILED: %s", c.D, e)
        log.debug(traceback.format_exc())
        if c.dry:
            echo("# FAILED: %s" % e)
            return "failed"
        c.meta["status"] = "failed"
        c.meta["error"] = "%s: %s" % (type(e).__name__, e)
        if current is not None:
            rec = c.meta.get("steps", {}).get(current)
            if rec is not None:
                rec.update({"status": "failed", "finished": now(), "commands": c.commands.get(current, []),
                            "log": c.log_for(current) if os.path.isfile(c.log_for(current)) else None})
        if isinstance(e, ToolError) and e.log_path:
            c.meta["error_log"] = e.log_path
        os.makedirs(c.logs, exist_ok=True)
        with open(os.path.join(c.logs, "pipeline.log"), "a", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        write_meta_atomic(c.meta, c.meta_path)
        return "failed"
    finally:
        if handler is not None:
            logging.getLogger("preprocessing").removeHandler(handler)
            handler.close()


def _step_params(c: RowContext, s: str) -> Dict[str, Any]:
    if s == "normalize":
        return {"normalization": c.meta.get("normalization")}
    if s == "resample":
        return {"antialias": c.meta.get("antialias"), "grid_state": c.meta.get("grid_state")}
    if s == "register":
        return dict(c.meta.get("registration") or {})
    if s == "strip":
        return {"mask_source": c.meta.get("mask_source")}
    return {}


def run_rows(rows: List[Tuple[int, ManifestRow]], cfg: Dict[str, Any], out_root: str, steps: Sequence[str],
             runner: ToolRunner, target: Target, overwrite: bool = False, clean_work: Optional[bool] = None,
             fail_fast: bool = False, echo: Optional[Callable[[str], None]] = None) -> int:
    """Process rows; returns 0 if none failed, else 1."""
    from .tools import get_tool_versions
    versions = {} if runner.dry_run else get_tool_versions(cfg, runner.env)
    counts: Dict[str, int] = {}
    for idx, row in rows:
        st = process_row(row, cfg, out_root, steps, runner, target, overwrite, clean_work, versions, echo, idx)
        counts[st] = counts.get(st, 0) + 1
        if st == "failed" and fail_fast:
            break
    log.info("run finished: %s", json.dumps(counts, sort_keys=True))
    return 1 if counts.get("failed") else 0
