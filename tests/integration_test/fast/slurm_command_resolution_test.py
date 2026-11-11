# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preserve multicall dispatch through real executable command symlinks."""

import os
import subprocess
import sys

import pytest

from nvflare.app_opt.job_launcher.slurm.config import resolve_slurm_parent_executables

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Slurm command dispatch uses POSIX executable symlinks")


@pytest.mark.parametrize("configured", [False, True])
def test_parent_executable_preserves_multicall_command_name(tmp_path, monkeypatch, configured):
    target = tmp_path / "site-wrapper"
    target.write_text(f"#!{sys.executable}\nimport os,sys\nprint(os.path.basename(sys.argv[0]))\n")
    target.chmod(0o700)
    commands = ("sbatch", "squeue", "sacct", "scancel")
    for command in commands:
        (tmp_path / command).symlink_to(target)
    monkeypatch.setenv("PATH", str(tmp_path))
    values = {name: str(tmp_path / name) if configured else None for name in commands}
    resolved = resolve_slurm_parent_executables(values)
    for name in commands:
        result = subprocess.run([resolved[name]], check=True, capture_output=True, text=True, timeout=30)
        assert result.stdout.strip() == name
