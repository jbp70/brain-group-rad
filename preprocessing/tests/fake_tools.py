"""Install fake FreeSurfer/ANTs/FSL executables (see fake_impl.py) into a bin dir."""
import json
import os
import stat
import sys
from typing import Dict, List

from .fake_impl import TOOLS

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def install_fake_tools(bin_dir: str, python: str = sys.executable) -> List[str]:
    os.makedirs(bin_dir, exist_ok=True)
    paths = []
    for name in TOOLS:
        p = os.path.join(bin_dir, name)
        with open(p, "w") as f:
            f.write("#!%s\nimport sys\nsys.path.insert(0, %r)\nimport fake_impl\nfake_impl.main(%r)\n"
                    % (python, TESTS_DIR, name))
        os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        paths.append(p)
    return paths


def read_calls(log_path: str) -> List[Dict]:
    if not os.path.isfile(log_path):
        return []
    with open(log_path) as f:
        return [json.loads(line) for line in f if line.strip()]
