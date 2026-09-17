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

import json
from types import SimpleNamespace

import torch

from tests.integration_test.src.validators.swarm_tensor_offload_validator import (
    OFFLOAD_OBSERVATIONS_FILE,
    SwarmTensorOffloadValidator,
)


def _make_results(tmp_path, leave_root=False):
    job_id = "job-1"
    workspace_root = tmp_path / "server" / "workspace"
    workspace_root.mkdir(parents=True)
    clients = []

    for site_name, expected_round in (("site-1", 0), ("site-2", 1)):
        client_root = tmp_path / site_name
        run_dir = client_root / job_id
        run_dir.mkdir(parents=True)
        checkpoint_path = run_dir / "app_client" / "FL_global_model.pt"
        checkpoint_path.parent.mkdir()
        torch.save({"model": {"weight": torch.full((4,), 3.0)}}, checkpoint_path)

        root_dir = tmp_path / f"nvflare_tensor_offload_{site_name}"
        file_path = root_dir / "nvflare_tensors_1" / "chunk_0.safetensors"
        if leave_root and site_name == "site-1":
            file_path.parent.mkdir(parents=True)
            file_path.write_bytes(b"tensor")

        observations = [
            {
                "site": site_name,
                "round": expected_round,
                "contributor": contributor,
                "file_paths": [str(file_path)],
                "root_dirs": [str(root_dir)],
                "total_bytes": 6,
            }
            for contributor in ("site-1", "site-2")
        ]
        marker_path = run_dir / OFFLOAD_OBSERVATIONS_FILE
        marker_path.write_text(
            "".join(json.dumps(observation) + "\n" for observation in observations),
            encoding="utf-8",
        )
        clients.append(SimpleNamespace(name=site_name, root_dir=str(client_root)))

    return {"job_id": job_id, "workspace_root": str(workspace_root)}, clients


def test_validator_accepts_correct_aggregate_and_cleaned_offload_roots(tmp_path):
    job_result, clients = _make_results(tmp_path)

    assert SwarmTensorOffloadValidator().validate_results(job_result, clients)


def test_validator_rejects_offload_root_left_after_end_run(tmp_path):
    job_result, clients = _make_results(tmp_path, leave_root=True)

    assert not SwarmTensorOffloadValidator().validate_results(job_result, clients)
