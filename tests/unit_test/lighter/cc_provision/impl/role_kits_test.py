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

"""Assemble isolated role kits from a throwaway Git fixture, not the user's index."""

import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "examples/devops/coco"


def test_assembled_roles_vendor_importable_helpers_without_nvflare(tmp_path):
    pytest.importorskip("tomllib", reason="role-kit assembly requires Python 3.11+")
    kits = runpy.run_path(str(SOURCE / "role_kits.py"))
    repo = tmp_path / "fixture-repo"
    source = repo / "examples/devops/coco"
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "examples/devops/coco"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    for name in set(names) | set(kits["PACKAGE_SOURCES"].values()):
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    # This commit is exclusively inside pytest's temporary fixture. Never stage
    # or commit the user's checkout to bypass the assembler's clean-tree check.
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "Fixture",
        ],
        check=True,
        capture_output=True,
    )
    output = tmp_path / "assembled"
    kits["assemble"](source, output)
    subprocess.run([sys.executable, str(output / "validate-package.py"), "--assembled"], check=True)
    # Isolated Python excludes checkout/PYTHONPATH; copied helpers cannot
    # accidentally pass by importing the locally installed editable NVFlare.
    checker = (
        "import runpy,sys; from pathlib import Path; root=Path(sys.argv[1]); "
        "api=runpy.run_path(str(root/'lib/workload-security-context.py')); "
        "assert api['normalize_resources']({'limits': {'nvidia.com/pgpu': 1}})['requests']['nvidia.com/pgpu']=='1'; "
        "assert runpy.run_path(str(root/'lib/trustee_claims.py'))['TRUST_VECTOR']['hardware']==2"
    )
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    for role in kits["ROLES"]:
        subprocess.run(
            [sys.executable, "-I", "-S", "-c", checker, str(output / role)], cwd=tmp_path, env=env, check=True
        )
    subprocess.run(
        [sys.executable, "-I", "-S", str(output / "admin/lib/workload-launch-profile.py"), "--help"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [sys.executable, "-I", "-S", str(output / "coco/lib/kata-runtime-profile.py"), "--help"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )
