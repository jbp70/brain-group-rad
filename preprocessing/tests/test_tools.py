import os

import pytest

from preprocessing.config import load_config
from preprocessing.tools import (ToolError, ToolRunner, check_tools, get_tool_versions, parse_ants_version,
                                 parse_freesurfer_version, required_tools, resolve_environment)


def test_resolve_environment_runs_setup_lines():
    env = resolve_environment(["echo noise-on-stdout", "export BGR_FOO='bar baz'"], base_env=dict(os.environ))
    assert env["BGR_FOO"] == "bar baz"
    assert resolve_environment([], base_env={"A": "1"}) == {"A": "1"}
    with pytest.raises(ToolError):
        resolve_environment(["exit 4"])


def test_runner_logs_errors_threads_and_dry_run(tmp_path, fake_tools, monkeypatch, capsys):
    env = dict(os.environ)
    r = ToolRunner(env, threads=3)
    log = str(tmp_path / "logs" / "x.log")
    monkeypatch.setenv("FAKE_FAIL", "N4BiasFieldCorrection")
    r = ToolRunner(dict(os.environ), threads=3)
    with pytest.raises(ToolError) as ei:
        r.run([r.which("N4BiasFieldCorrection"), "-i", "/nope"], log)
    assert ei.value.log_path == log and "exited with 3" in str(ei.value)
    text = open(log).read()
    assert "fake failure" in text and "$ " in text
    assert r.tool_env()["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] == "3"
    assert r.tool_env()["OMP_NUM_THREADS"] == "3"
    n_calls = len(fake_tools.calls())
    dry = ToolRunner(dict(os.environ), dry_run=True, echo=print)
    dry.run(["flirt", "-in", "/a b.nii.gz"], str(tmp_path / "never.log"))
    assert capsys.readouterr().out.strip() == "CMD flirt -in '/a b.nii.gz'"
    assert len(fake_tools.calls()) == n_calls and not os.path.exists(tmp_path / "never.log")


def test_tool_resolution(fake_tools):
    cfg = load_config(None)
    r = ToolRunner(dict(os.environ))
    assert r.tool(cfg, "flirt") == os.path.join(fake_tools.bin, "flirt")
    cfg["tools"]["flirt"] = "definitely_missing_tool"
    with pytest.raises(ToolError):
        r.tool(cfg, "flirt")
    assert ToolRunner(dict(os.environ), dry_run=True).tool(cfg, "flirt") == "definitely_missing_tool"


def test_check_tools_accepts_freesurfer_dev_build(fake_tools, config_file):
    # Martinos: FREESURFER_HOME=/usr/local/freesurfer/dev has no x.y.z in its build stamp.
    fs = os.path.join(os.path.dirname(fake_tools.bin), "fs_dev")
    os.makedirs(fs)
    with open(os.path.join(fs, "build-stamp.txt"), "w") as f:
        f.write("freesurfer-linux-centos7_x86_64-dev-20260312\n")
    env = dict(os.environ, FREESURFER_HOME=fs)
    ok, lines = check_tools(load_config(config_file), env)
    assert ok, lines
    assert any("dev build 20260312 OK" in x for x in lines)
    # Tool lines report the resolved absolute path.
    assert any(x.split()[:2] == ["flirt", os.path.join(fake_tools.bin, "flirt")] for x in lines)


def test_version_parsing():
    assert parse_freesurfer_version("freesurfer-linux-ubuntu22_x86_64-7.4.1-20230614-7eb8460") == (7, 4, 1)
    assert parse_ants_version("ANTs Version: 2.5.0.post7\nCompiled") == (2, 5)
    assert parse_ants_version("ANTs Version: v2.3.5") == (2, 3)
    assert parse_ants_version("garbage") is None


def test_check_tools_with_fakes_and_missing(fake_tools, config_file):
    cfg = load_config(config_file)
    ok, lines = check_tools(cfg, dict(os.environ))
    assert ok, lines
    text = "\n".join(lines)
    assert "7.4.1 OK" in text and "6.0.7.9" in text
    v = get_tool_versions(cfg, dict(os.environ))
    assert "2.5.0" in v["ants"]
    env = dict(os.environ)
    env["PATH"] = "/nonexistent"
    ok, lines = check_tools(cfg, env)
    assert not ok and any("MISSING" in x for x in lines)
    env = dict(os.environ)
    env["FREESURFER_HOME"] = str(fake_tools.bin)  # no build-stamp
    assert not check_tools(cfg, env)[0]
    assert "antsRegistration" not in required_tools(cfg)
    cfg["registration"]["backend"] = "ants"
    assert "antsApplyTransforms" in required_tools(cfg) and "flirt" not in required_tools(cfg)
