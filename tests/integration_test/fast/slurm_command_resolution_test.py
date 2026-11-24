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
@pytest.mark.parametrize("directory_alias", [False, True])
def test_parent_executable_preserves_multicall_command_name(tmp_path, monkeypatch, configured, directory_alias):
    command_dir = tmp_path / "commands"
    command_dir.mkdir()
    search_dir = command_dir
    if directory_alias:
        search_dir = tmp_path / "alias"
        search_dir.symlink_to(command_dir, target_is_directory=True)
    target = tmp_path / "site-wrapper"
    target.write_text(f"#!{sys.executable}\nimport os,sys\nprint(os.path.basename(sys.argv[0]))\n")
    target.chmod(0o700)
    commands = ("sbatch", "squeue", "sacct", "scancel")
    for command in commands:
        (command_dir / command).symlink_to(target)
    monkeypatch.setenv("PATH", str(search_dir))
    values = {name: str(search_dir / name) if configured else None for name in commands}
    resolved = resolve_slurm_parent_executables(values)
    for name in commands:
        assert resolved[name] == str(command_dir.resolve() / name)
        result = subprocess.run([resolved[name]], check=True, capture_output=True, text=True, timeout=30)
        assert result.stdout.strip() == name
