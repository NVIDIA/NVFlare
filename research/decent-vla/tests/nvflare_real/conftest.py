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

"""Helpers for the tests against the real NVFlare package (no fakes).

Each test exports a job with the decent-vla CLI and runs the exported folder
in the FLARE simulator, both in subprocesses, so nothing here shares a
process with the fake nvflare of tests/nvflare/. Skipped when nvflare is not
installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("nvflare")

NAME = "mock_job"


def mock_config(**fl) -> dict:
    """A 2-client, 2-round mock run exported for deployment."""
    return {
        "name": NAME,
        "seed": 42,
        "device": "cpu",
        "backend": {"name": "mock", "optim": {"name": "adamw", "lr": 1.0e-3, "weight_decay": 0.0}},
        "data": {"name": "mock", "batch_size": 8, "num_clients": 2, "partitioning": "iid"},
        "fl": {
            "strategy": {"name": "fedavg"},
            "num_rounds": 2,
            "local_steps": 2,
            "mode": "nvflare",
            "output_dir": f"outputs/{NAME}",
            **fl,
            "nvflare": {"env": "export", **fl.get("nvflare", {})},
        },
    }


def export(tmp_path: Path, cfg: dict, extra_files: dict | None = None) -> Path:
    """Write the run config (and `extra_files`) to tmp_path/src, export the job
    from there with the CLI, and return the job folder (the one with meta.json)."""
    src = tmp_path / "src"
    src.mkdir()
    for name, text in (extra_files or {}).items():
        (src / name).write_text(text)
    cfg["fl"]["nvflare"]["job_dir"] = str(tmp_path / "jobs")
    (src / "run.yaml").write_text(yaml.safe_dump(cfg))
    subprocess.run(
        [sys.executable, "-m", "decent_vla.cli.main", "train", "--config", "run.yaml"],
        cwd=src, check=True, capture_output=True, timeout=300,
    )
    return tmp_path / "jobs" / NAME


def use_fault_client(job: Path) -> None:
    shutil.copy(Path(__file__).with_name("fault_client.py"), job / "app" / "custom" / "client_script.py")


def simulate(job: Path, workspace: Path, fault: str = "", timeout: int = 300) -> Path:
    """Run the exported job in the FLARE simulator (sites c0, c1); return the workspace."""
    nvflare = shutil.which("nvflare", path=str(Path(sys.executable).parent)) or "nvflare"
    env = {**os.environ, "DVLA_TEST_FAULT": fault}
    subprocess.run(
        [nvflare, "simulator", "-w", str(workspace), "-n", "2", "-t", "2", "-c", "c0,c1", str(job)],
        cwd=workspace.parent, env=env, capture_output=True, timeout=timeout,
    )
    return workspace


def server_outputs(workspace: Path) -> Path:
    """Where a deployed job's server writes: under its own run folder."""
    return workspace / "server" / "simulate_job" / "outputs" / NAME


def server_log(workspace: Path) -> str:
    return (workspace / "server" / "log.txt").read_text()
