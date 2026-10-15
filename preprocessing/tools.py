"""External tool environment, execution and version checks.

* resolve_environment(env_setup): runs the config's env_setup lines once in
  ``bash -c`` and captures the resulting environment (only those
  user-authored lines go through a shell; data paths never do).
* ToolRunner.run(argv, log_path): subprocess with an argument list,
  stdout+stderr appended to the per-step log, nonzero exit -> ToolError.
  dry_run prints ``CMD <shell-quoted argv>`` and executes nothing.
* check_tools(cfg): which tools are on PATH, FreeSurfer/ANTs/FSL versions.
* resolve_tmp_dir(cfg, out_root, env): TMPDIR handed to the tools (never an
  unbound host TMPDIR inside the container).

Example:
    env = resolve_environment(cfg["env_setup"])
    runner = ToolRunner(env, threads=8)
    runner.run(["N4BiasFieldCorrection", "--version"], "/tmp/n4.log")
"""
import datetime
import logging
import os
import re
import shlex
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .config import ConfigError, mni_template

log = logging.getLogger(__name__)
_ENV_MARK = "\0__BGR_ENV_BEGIN__\0"


class ToolError(RuntimeError):
    """External tool failure (missing executable or nonzero exit)."""

    def __init__(self, msg: str, log_path: Optional[str] = None):
        super().__init__(msg)
        self.log_path = log_path


def resolve_environment(env_setup: List[str], base_env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Environment after running env_setup lines (empty list -> copy of base env)."""
    base = dict(os.environ if base_env is None else base_env)
    if not env_setup:
        return base
    script = "\n".join(env_setup) + "\nprintf '%s'\nenv -0\n" % _ENV_MARK.replace("\0", "\\0")
    proc = subprocess.run(["bash", "-c", script], env=base, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise ToolError("env_setup failed (exit %d): %s" % (proc.returncode,
                                                            proc.stderr.decode(errors="replace").strip()))
    out = proc.stdout.decode(errors="replace")
    if _ENV_MARK not in out:
        raise ToolError("env_setup: could not capture environment")
    env: Dict[str, str] = {}
    for item in out.split(_ENV_MARK, 1)[1].split("\0"):
        if "=" in item:
            k, v = item.split("=", 1)
            env[k] = v
    for k in ("PWD", "SHLVL", "_", "OLDPWD"):
        env.pop(k, None)
    return env


class ToolRunner:
    """Runs argv lists with a fixed environment and thread settings."""

    def __init__(self, env: Mapping[str, str], threads: int = 1, dry_run: bool = False,
                 echo: Optional[Callable[[str], None]] = None, tmp_dir: Optional[str] = None):
        self.env = dict(env)
        self.threads = int(threads)
        self.dry_run = dry_run
        self.echo = echo or print
        self.tmp_dir = tmp_dir  # exported as TMPDIR/TMP/TEMP to every tool (see resolve_tmp_dir)

    def which(self, name: str) -> Optional[str]:
        if os.path.isabs(name):
            return name if os.access(name, os.X_OK) else None
        return shutil.which(name, path=self.env.get("PATH", ""))

    def tool(self, cfg: Dict[str, Any], key: str) -> str:
        """Resolved path of cfg['tools'][key] (dry-run falls back to the bare name)."""
        name = cfg["tools"][key]
        path = self.which(name)
        if path is None:
            if self.dry_run:
                log.warning("tool %s (%s) not found on PATH; printing bare name", key, name)
                return name
            raise ToolError("tool %s not found: %r (check tools/env_setup in the config)" % (key, name))
        return path

    def tool_env(self) -> Dict[str, str]:
        env = dict(self.env)
        env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(self.threads)
        env["OMP_NUM_THREADS"] = str(self.threads)
        env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
        if self.tmp_dir:
            for k in ("TMPDIR", "TMP", "TEMP"):
                env[k] = self.tmp_dir
        return env

    def run(self, argv: List[str], log_path: str) -> int:
        argv = [str(a) for a in argv]
        if self.dry_run:
            self.echo("CMD " + shlex.join(argv))
            return 0
        if self.tmp_dir:
            os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("[%s] $ %s\n" % (datetime.datetime.now().isoformat(timespec="seconds"), shlex.join(argv)))
            f.flush()
            proc = subprocess.run(argv, env=self.tool_env(), stdout=f, stderr=subprocess.STDOUT, check=False)
            f.write("[exit %d]\n" % proc.returncode)
        if proc.returncode != 0:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = "".join(f.readlines()[-15:])
            raise ToolError("%s exited with %d (log: %s)\n%s" % (os.path.basename(argv[0]), proc.returncode,
                                                                log_path, tail), log_path)
        return 0


def resolve_tmp_dir(cfg: Dict[str, Any], out_root: str, env: Mapping[str, str]) -> str:
    """Temp folder for the external tools.

    cfg['tmp_dir'] if set; else $TMPDIR from ``env`` when it is an existing
    writable folder (on SLURM nodes /scratch/$SLURM_JOBID, bound into the
    container); else <out_root>/_tmp. A host TMPDIR that is not bound into the
    container (e.g. glacier's /local_mount/...) is therefore never used.
    The '_' prefix keeps export-npy from treating it as a dataset folder.
    """
    from .config import expand_path
    if cfg.get("tmp_dir"):
        return os.path.abspath(expand_path(cfg["tmp_dir"], env))  # type: ignore[arg-type]
    t = env.get("TMPDIR")
    if t and os.path.isdir(t) and os.access(t, os.W_OK | os.X_OK):
        return t
    return os.path.join(os.path.abspath(out_root), "_tmp")


_FS_VER = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_FS_DEV = re.compile(r"-dev-(\d{8})")
_ANTS_VER = re.compile(r"ANTs Version:\s*v?(\d+)\.(\d+)(?:\.(\d+))?")


def parse_freesurfer_version(text: str) -> Optional[Tuple[int, int, int]]:
    m = _FS_VER.search(text or "")
    return tuple(int(x) for x in m.groups()) if m else None  # type: ignore[return-value]


def parse_ants_version(text: str) -> Optional[Tuple[int, int]]:
    m = _ANTS_VER.search(text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def get_tool_versions(cfg: Dict[str, Any], env: Mapping[str, str]) -> Dict[str, Optional[str]]:
    """Version strings: FreeSurfer build stamp, ANTs, FSL (None when unavailable)."""
    out: Dict[str, Optional[str]] = {"freesurfer": None, "ants": None, "fsl": None}
    fs_home = env.get("FREESURFER_HOME")
    if fs_home:
        out["freesurfer"] = _read(os.path.join(fs_home, "build-stamp.txt"))
    fsl = env.get("FSLDIR")
    if fsl:
        out["fsl"] = _read(os.path.join(fsl, "etc", "fslversion"))
    ants = shutil.which(cfg["tools"]["antsRegistration"], path=env.get("PATH", "")) \
        if not os.path.isabs(cfg["tools"]["antsRegistration"]) else cfg["tools"]["antsRegistration"]
    if ants and os.path.exists(ants):
        try:
            p = subprocess.run([ants, "--version"], env=dict(env), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=60, check=False)
            out["ants"] = p.stdout.decode(errors="replace").strip() or None
        except (OSError, subprocess.SubprocessError):
            pass
    return out


def required_tools(cfg: Dict[str, Any]) -> List[str]:
    """Tool keys needed by the configured pipeline."""
    keys = ["mri_synthstrip", "N4BiasFieldCorrection", "mri_synthseg"]
    reg = cfg["registration"]
    if reg["backend"] == "flirt":
        keys.append("flirt")
        if reg["reorient"] == "fslreorient2std":
            keys.append("fslreorient2std")
    else:
        keys += ["antsRegistration", "antsApplyTransforms"]
    return keys


def check_tools(cfg: Dict[str, Any], env: Mapping[str, str]) -> Tuple[bool, List[str]]:
    """Return (ok, report lines)."""
    ok = True
    lines = []
    runner = ToolRunner(env)
    for key in required_tools(cfg):
        path = runner.which(cfg["tools"][key])
        lines.append("%-24s %s" % (key, path or "MISSING (%s)" % cfg["tools"][key]))
        ok = ok and path is not None
    vers = get_tool_versions(cfg, env)
    fs = parse_freesurfer_version(vers["freesurfer"] or "")
    if not env.get("FREESURFER_HOME"):
        lines.append("FreeSurfer               FREESURFER_HOME not set")
        ok = False
    elif fs is None and _FS_DEV.search(vers["freesurfer"] or ""):
        # Development build (e.g. Martinos /usr/local/freesurfer/dev): newer than any 7.x release.
        lines.append("FreeSurfer               dev build %s OK" % _FS_DEV.search(vers["freesurfer"] or "").group(1))
    elif fs is None:
        lines.append("FreeSurfer               version unknown (no build-stamp.txt)")
        ok = False
    else:
        good = fs >= (7, 3, 0)
        lines.append("FreeSurfer               %d.%d.%d %s" % (fs + ("OK" if good else "TOO OLD (need >= 7.3)",)))
        ok = ok and good
    if cfg["registration"]["backend"] == "ants" or vers["ants"]:
        av = parse_ants_version(vers["ants"] or "")
        if av is None:
            lines.append("ANTs                     version unknown")
            ok = ok and cfg["registration"]["backend"] != "ants"
        else:
            lines.append("ANTs                     %d.%d%s" % (av + (
                "" if av >= (2, 3) else " (old: --random-seed needs >= 2.3)",)))
    if cfg["registration"]["backend"] == "flirt":
        lines.append("FSL                      %s" % (vers["fsl"] or "version unknown (FSLDIR/etc/fslversion)"))
        if not env.get("FSLDIR"):
            lines.append("FSL                      FSLDIR not set")
    for contrast in sorted(set(cfg["contrast_template"]) | {"T1w"}):
        try:
            t = mni_template(cfg, contrast, env)
        except ConfigError as e:
            lines.append("MNI template %-11s %s" % (contrast, e))
            if contrast == "T1w":
                ok = False
            continue
        for k, v in t.items():
            if v is None:
                continue
            exists = os.path.isfile(v)
            lines.append("MNI %-6s %-13s %s %s" % (contrast, k, v, "OK" if exists else "MISSING"))
            if not exists and contrast == "T1w" and k != "image_2mm":
                ok = False
    from .config import target_paths
    try:
        tp = target_paths(cfg, env)
        for k in ("grid", "template", "template_mask"):
            exists = os.path.isfile(tp[k])
            lines.append("target %-17s %s %s" % (k, tp[k], "OK" if exists else "MISSING (run build-target-grid)"))
    except ConfigError as e:
        lines.append("target                   %s" % e)
    return ok, lines
