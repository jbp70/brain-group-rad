"""Container wrapper (run_in_container.sh) argv handling and tool temp-dir placement."""
import json
import os
import shutil
import stat
import subprocess
import sys

import pytest

from preprocessing.cli import main
from preprocessing.config import load_config
from preprocessing.tools import ToolRunner, resolve_tmp_dir

from . import datasets as ds
from .conftest import write_config
from .fake_tools import read_calls

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PKG)
WRAPPER = os.path.join(PKG, "run_in_container.sh")

# Fake `singularity`: records its argv, then runs everything after the image
# path on the host (like `singularity exec` would inside the container).
FAKE_SINGULARITY = r'''#!%(py)s
import json, os, sys
argv = sys.argv[1:]
with open(os.environ["FAKE_SING_LOG"], "a") as f:
    f.write(json.dumps(argv) + "\n")
assert argv[0] == "exec", argv
i = 1
while argv[i].startswith("-"):
    i += 2 if argv[i] in ("-B", "--bind") else 1
rest = argv[i + 1:]
os.execvp(rest[0], rest)
'''


def _exe(path, text):
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def wrapper_env(tmp_path):
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    _exe(str(bin_dir / "singularity"), FAKE_SINGULARITY % {"py": sys.executable})
    sif = tmp_path / "img.sif"
    sif.write_text("not a real image")
    (tmp_path / "bind_b").mkdir()
    env = {
        "PATH": "%s:/usr/bin:/bin" % bin_dir,
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "PYTHONDONTWRITEBYTECODE": "1",
        "FAKE_SING_LOG": str(tmp_path / "sing.jsonl"),
        "BGR_SIF": str(sif),
        "BGR_BINDS": "%s %s %s:/mnt/b" % (tmp_path / "missing", tmp_path / "bind_b", tmp_path / "bind_b"),
        "BGR_REPO_DIR": REPO,
        "BGR_PYTHON": sys.executable,
        "BGR_TMPDIR": str(tmp_path / "tmp dir"),
        "BGR_FREESURFER_HOME": "/fs home",
        "BGR_FSLDIR": "/fsl",
        "BGR_ANTS_BIN": "/ants/bin",
    }
    return tmp_path, bin_dir, env


def _call(tmp_path, env, args, cwd):
    p = subprocess.run(["bash", WRAPPER] + args, env=env, cwd=cwd, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, universal_newlines=True)
    log = tmp_path / "sing.jsonl"
    calls = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    if log.exists():
        log.unlink()
    return p, calls


def test_wrapper_passes_args_verbatim(wrapper_env):
    tmp_path, _, env = wrapper_env
    env["USE_GPU"] = "1"
    work = tmp_path / "work dir"
    work.mkdir()
    rows = [ds.lumir_row(str(tmp_path / "in"), subject="LUMIRMRI_%04d" % i) for i in range(3)]
    # A manifest name full of shell metacharacters: must reach python unchanged and never run.
    nasty = "m 1;$(touch PWNED)`touch PWNED2`'\"&.csv"
    ds.write_rows(str(work / nasty), rows)
    args = ["n-chunks", "--manifest", nasty, "--chunk-size", "2"]
    p, calls = _call(tmp_path, env, args, str(work))
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "2"
    assert "skipping bind %s" % (tmp_path / "missing") in p.stderr
    assert not list(tmp_path.rglob("PWNED*"))
    assert len(calls) == 1
    argv = calls[0]
    inner = argv[argv.index("-c") + 1]
    assert argv == ["exec", "--nv", "-B", str(tmp_path / "bind_b"), "-B", "%s:/mnt/b" % (tmp_path / "bind_b"),
                    env["BGR_SIF"], "env",
                    "FREESURFER_HOME=/fs home", "FSLDIR=/fsl", "FSLOUTPUTTYPE=NIFTI_GZ", "BGR_ANTS_BIN=/ants/bin",
                    "BGR_REPO_DIR=" + REPO, "BGR_WORKDIR=" + os.path.realpath(str(work)),
                    "BGR_PYTHON=" + sys.executable, "TMPDIR=" + str(tmp_path / "tmp dir"),
                    "bash", "-c", inner, "run_in_container"] + args
    # The bash -c script is fixed text: no user value is interpolated into it.
    for value in (nasty, str(work), REPO, "/fs home"):
        assert value not in inner
    assert inner.rstrip().endswith('exec "$BGR_PYTHON" -m preprocessing "$@"')
    assert (tmp_path / "tmp dir").is_dir()


def test_wrapper_gpu_flag(wrapper_env):
    tmp_path, bin_dir, env = wrapper_env
    args = ["n-chunks", "--help"]
    _exe(str(bin_dir / "nvidia-smi"), "#!/bin/bash\necho 'GPU 0: Fake A100 (UUID: GPU-0)'\n")
    for use_gpu, expect in ((None, True), ("0", False), ("1", True)):
        e = dict(env)
        if use_gpu is not None:
            e["USE_GPU"] = use_gpu
        p, calls = _call(tmp_path, e, args, str(tmp_path))
        assert p.returncode == 0, p.stderr
        assert ("--nv" in calls[0][:calls[0].index(env["BGR_SIF"])]) is expect, (use_gpu, calls[0])
    _exe(str(bin_dir / "nvidia-smi"), "#!/bin/bash\necho 'NVIDIA-SMI has failed' >&2\nexit 9\n")
    p, calls = _call(tmp_path, env, args, str(tmp_path))
    assert p.returncode == 0 and "--nv" not in calls[0]
    os.remove(str(bin_dir / "nvidia-smi"))
    if shutil.which("nvidia-smi", path="/usr/bin:/bin") is None:  # no nvidia-smi at all -> CPU
        p, calls = _call(tmp_path, env, args, str(tmp_path))
        assert p.returncode == 0 and "--nv" not in calls[0]


def test_wrapper_errors(wrapper_env):
    tmp_path, _, env = wrapper_env
    p, calls = _call(tmp_path, env, [], str(tmp_path))
    assert p.returncode == 2 and calls == [] and "usage" in p.stderr
    e = dict(env, BGR_SIF=str(tmp_path / "nope.sif"))
    p, calls = _call(tmp_path, e, ["check-tools"], str(tmp_path))
    assert p.returncode == 2 and calls == [] and "image not found" in p.stderr


def test_wrapper_defaults_are_martinos_values():
    text = open(WRAPPER).read()
    for s in ("/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif",
              "/scratch /autofs /cluster /space /vast /usr/local/freesurfer",
              "/usr/local/freesurfer/dev",
              "/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/fsl.64bit/5.0.7",
              "/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/ANTS/2.4.4/bin",
              '"/scratch/${SLURM_JOB_ID}"'):
        assert s in text, s
    assert "eval" not in text


# ------------------------------------------------------------------ temp files
def test_resolve_tmp_dir(tmp_path):
    cfg = load_config(None)
    out = str(tmp_path / "out")
    assert resolve_tmp_dir(cfg, out, {"TMPDIR": "/local_mount/space/glacier/unbound"}) == os.path.join(out, "_tmp")
    assert resolve_tmp_dir(cfg, out, {}) == os.path.join(out, "_tmp")
    assert resolve_tmp_dir(cfg, out, {"TMPDIR": str(tmp_path)}) == str(tmp_path)
    cfg["tmp_dir"] = str(tmp_path / "cfg_tmp")
    assert resolve_tmp_dir(cfg, out, {"TMPDIR": str(tmp_path)}) == str(tmp_path / "cfg_tmp")


def test_runner_exports_tmp_dir(tmp_path, fake_tools, monkeypatch):
    monkeypatch.setenv("TMPDIR", "/nonexistent/unbound_tmp")
    work_tmp = str(tmp_path / "work_tmp")
    r = ToolRunner(dict(os.environ), tmp_dir=work_tmp)
    assert all(r.tool_env()[k] == work_tmp for k in ("TMPDIR", "TMP", "TEMP"))
    r.run([r.which("antsRegistration"), "--version"], str(tmp_path / "v.log"))
    calls = read_calls(fake_tools.log)
    assert [(c["tmp_dir"], c["TMPDIR"]) for c in calls] == [(work_tmp, work_tmp)]
    assert ToolRunner(dict(os.environ)).tool_env()["TMPDIR"] == "/nonexistent/unbound_tmp"  # unchanged without tmp_dir


def _tool_tmp_dirs(log):
    with open(log) as f:
        return {(c["tmp_dir"], c["TMPDIR"]) for c in map(json.loads, f) if c.get("argv") != ["--version"]}


def test_tools_temp_files_go_to_work_dir_not_tmpdir(tmp_path, fake_tools, target_dir, mni_files, monkeypatch):
    # Host TMPDIR not bound into the container (glacier case): must not be used.
    monkeypatch.setenv("TMPDIR", "/nonexistent/local_mount/scratch")
    row = ds.neurite_row(str(tmp_path / "in"))
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    cfg = write_config(tmp_path / "c.yaml", target_dir, mni_files)
    assert main(["run", "--config", cfg, "--manifest", man, "--out-root", out]) == 0
    default_tmp = os.path.join(out, "_tmp")
    assert _tool_tmp_dirs(fake_tools.log) == {(default_tmp, default_tmp)}
    # Configured tmp_dir wins (here for the batch SynthSeg command).
    open(fake_tools.log, "w").close()
    cfg_tmp = str(tmp_path / "configured tmp")
    cfg2 = write_config(tmp_path / "c2.yaml", target_dir, mni_files, tmp_dir=cfg_tmp)
    assert main(["synthseg", "--config", cfg2, "--manifest", man, "--out-root", out]) == 0
    assert _tool_tmp_dirs(fake_tools.log) == {(cfg_tmp, cfg_tmp)}
    assert not os.path.exists("/nonexistent/local_mount/scratch")
