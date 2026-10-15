"""Pure argv builders for every external tool (no execution here).

Each function returns a list of strings for subprocess.run (never a shell
string). ``tool`` is the resolved executable path.

Example:
    argv = n4_cmd("/opt/ants/bin/N4BiasFieldCorrection", "in.nii.gz", "mask.nii.gz", "out.nii.gz", cfg["n4"])
"""
from typing import Any, Dict, List, Optional


def synthstrip_cmd(tool: str, image: str, out: str, mask: str, extra_args: List[str]) -> List[str]:
    return [tool, "-i", image, "-o", out, "-m", mask] + [str(a) for a in extra_args]


def n4_cmd(tool: str, image: str, mask: str, out: str, n4: Dict[str, Any]) -> List[str]:
    return [tool, "-d", "3", "-i", image, "-x", mask, "-o", out, "-s", str(n4["shrink_factor"]),
            "-b", str(n4["bspline_fitting"]), "-c", str(n4["convergence"]), "-v", "0"]


def fslreorient2std_cmd(tool: str, image: str, out: str) -> List[str]:
    return [tool, image, out]


def flirt_register_cmd(tool: str, moving: str, ref: str, omat: str, flirt: Dict[str, Any]) -> List[str]:
    r = str(int(flirt["search_range"]))
    argv = [tool, "-in", moving, "-ref", ref, "-omat", omat, "-dof", str(flirt["dof"]),
            "-cost", str(flirt["cost"])]
    for ax in ("x", "y", "z"):
        argv += ["-searchr" + ax, "-" + r, r]
    return argv + [str(a) for a in flirt["extra_args"]]


def flirt_apply_cmd(tool: str, moving: str, ref: str, init: str, out: str, interp: str,
                    float_output: bool = False) -> List[str]:
    argv = [tool, "-in", moving, "-ref", ref, "-applyxfm", "-init", init, "-out", out, "-interp", interp]
    if float_output:
        argv += ["-datatype", "float"]
    return argv


def ants_registration_cmd(tool: str, fixed: str, moving: str, out_prefix: str, ants: Dict[str, Any],
                          fixed_mask: Optional[str] = None, moving_mask: Optional[str] = None) -> List[str]:
    metric = "MI[%s,%s,1,%d,%s,%s]" % (fixed, moving, int(ants["metric_bins"]),
                                       ants["sampling_strategy"], ants["sampling_percentage"])
    stage = ["--convergence", str(ants["convergence"]), "--shrink-factors", str(ants["shrink_factors"]),
             "--smoothing-sigmas", str(ants["smoothing_sigmas"])]
    step = ants["gradient_step"]
    argv = [tool, "--dimensionality", "3", "--float", "1", "--collapse-output-transforms", "1",
            "--output", out_prefix, "--interpolation", "Linear",
            "--winsorize-image-intensities", str(ants["winsorize"]), "--use-histogram-matching", "0",
            "--initial-moving-transform", "[%s,%s,1]" % (fixed, moving),
            "--transform", "Rigid[%s]" % step, "--metric", metric] + stage + [
            "--transform", "%s[%s]" % (ants["transform"], step), "--metric", metric] + stage
    if ants.get("random_seed") is not None:
        argv += ["--random-seed", str(ants["random_seed"])]
    if ants.get("use_masks") and fixed_mask and moving_mask:
        argv += ["-x", "[%s,%s]" % (fixed_mask, moving_mask)]
    return argv + ["--verbose", "0"]


def ants_apply_cmd(tool: str, moving: str, reference: str, out: str, interp: str,
                   transform: Optional[str]) -> List[str]:
    argv = [tool, "-d", "3", "-i", moving, "-r", reference, "-o", out, "-n", interp, "--float", "1", "-v", "0"]
    if transform:
        argv += ["-t", transform]
    return argv


def synthseg_cmd(tool: str, inp: str, out: str, synthseg: Dict[str, Any], vol: Optional[str] = None,
                 qc: Optional[str] = None) -> List[str]:
    argv = [tool, "--i", inp, "--o", out]
    if synthseg["parc"]:
        argv.append("--parc")
    if synthseg["robust"]:
        argv.append("--robust")
    if synthseg["vol"] and vol:
        argv += ["--vol", vol]
    if synthseg["qc"] and qc:
        argv += ["--qc", qc]
    argv += ["--threads", str(synthseg["threads"])]
    if synthseg["cpu"]:
        argv.append("--cpu")
    return argv + [str(a) for a in synthseg["extra_args"]]
