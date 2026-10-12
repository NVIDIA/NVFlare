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

import ast
import os
import subprocess
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[4] / "examples" / "docker"


@pytest.mark.parametrize(
    "target_arch,debian_arch,pinned",
    [("arm64", "amd64", True), ("amd64", "arm64", False), ("", "arm64", True), ("", "amd64", False)],
)
def test_job_image_architecture_dependencies(tmp_path, target_arch, debian_arch, pinned):
    # Execute the actual dependency-selection shell, recording pip invocations
    # instead of downloading wheels. Conflicting architectures verify precedence.
    dockerfile = (EXAMPLE / "Dockerfile.nvflare-job").read_text().replace("\\\n", "")
    install_command = next(line[4:] for line in dockerfile.splitlines() if line.startswith("RUN pip install"))
    pip_log = tmp_path / "pip.log"
    pip = tmp_path / "pip"
    pip.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$PIP_LOG"\n')
    pip.chmod(0o755)
    dpkg = tmp_path / "dpkg"
    dpkg.write_text('#!/bin/sh\nprintf "%s\\n" "$DEBIAN_ARCH"\n')
    dpkg.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", PIP_LOG=str(pip_log))
    env.update(TARGETARCH=target_arch, DEBIAN_ARCH=debian_arch)
    subprocess.run(["bash", "-ec", install_command], env=env, check=True)
    calls = pip_log.read_text().splitlines()
    assert calls[0] == "install -U pip"
    assert calls[1].startswith("install -e .[PT,TRACKING]")
    assert ("torch==2.5.1 torchvision==0.20.1" in calls[1]) == pinned
    if not pinned:
        assert calls[1] == "install -e .[PT,TRACKING]"


@pytest.mark.parametrize("override", [None, "", "/workspace/cifar10"])
def test_cifar10_cache_override(monkeypatch, override):
    if override is None:
        monkeypatch.delenv("NVFL_CIFAR10_ROOT", raising=False)
    else:
        monkeypatch.setenv("NVFL_CIFAR10_ROOT", override)
    # Evaluate only the assignment so host torch/torchvision are unnecessary.
    tree = ast.parse((EXAMPLE / "jobs/hello-pt-docker/app/custom/client.py").read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "DATASET_PATH" for t in node.targets)
    )
    namespace = {"os": os}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "client.py", "exec"), namespace)
    assert namespace["DATASET_PATH"] == (override or "/var/tmp/nvflare/data")
