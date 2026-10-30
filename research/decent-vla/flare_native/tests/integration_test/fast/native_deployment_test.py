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


"""Opt-in tests of real FLARE jobs. RUN_NATIVE_SIMULATOR_TESTS=1 enables them."""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch
import yaml

from tests.process_test_utils import run_process

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NATIVE_SIMULATOR_TESTS") != "1", reason="real FLARE simulator")


@pytest.mark.parametrize("algorithm,external", [("fedavg", True), ("fedadam", False)])
def test_exported_native_job_is_portable_and_preserves_warm_clients(tmp_path, algorithm, external):
    source = tmp_path / "source"
    source.mkdir()
    raw = {
        "name": "portable",
        "device": "cpu",
        "backend": {"name": "mock"},
        "data": {"num_clients": 2},
        "fl": {
            "num_rounds": 2,
            "local_steps": 2,
            "strategy": {"name": algorithm, "server_lr": 0.1},
            "persist_client_state": True,
            "save_every_n_rounds": 1,
            "output_dir": "outputs/portable",
            "nvflare": {"launch_external_process": external},
        },
    }
    (source / "run.yaml").write_text(yaml.safe_dump(raw))
    exported = run_process(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[3] / "job.py"),
            "--config",
            "run.yaml",
            "--job-dir",
            str(tmp_path / "jobs"),
        ],
        cwd=source,
        timeout=120,
    )
    assert exported.returncode == 0, exported.stdout + exported.stderr
    moved = tmp_path / "moved"
    shutil.move(str(tmp_path / "jobs" / "portable"), moved)
    (source / "run.yaml").unlink()
    binary = Path(sys.executable).parent / "nvflare"
    ws = tmp_path / "ws"
    run = run_process(
        [str(binary), "simulator", str(moved), "-w", str(ws), "-c", "c0,c1", "-t", "2"],
        cwd=tmp_path,
        timeout=180,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    out = ws / "server/simulate_job/outputs/portable"
    assert (out / "checkpoints/server_round_final.pt").exists(), run.stdout + run.stderr
    assert len(json.loads((out / "training_curve.json").read_text())) == 2
    state = torch.load(
        ws / "c1/simulate_job/outputs/portable/client_states/c1_round_0001.pt", map_location="cpu", weights_only=False
    )
    assert state["global_step"] == 4
    assert {int(v["step"]) for v in state["optimizer"]["state"].values()} == {4}
    if algorithm == "fedadam":
        checkpoint = torch.load(out / "checkpoints/server_round_final.pt", map_location="cpu", weights_only=False)
        assert {int(v["step"]) for v in checkpoint["optimizer"]["state"].values()} == {2}
