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

import importlib.util
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


def _load_module(relative_path: str, module_name: str):
    module_path = Path(__file__).resolve().parents[1] / "nvflare_smoke" / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_smoke_client_module():
    return _load_module("client.py", "adaptive_hetero_nvflare_smoke_client")


def _load_smoke_job_module():
    return _load_module("job.py", "adaptive_hetero_nvflare_smoke_job")


def test_evaluate_rejects_empty_validation_loader():
    smoke_client = _load_smoke_client_module()
    model = torch.nn.Linear(smoke_client.NUM_FEATURES, smoke_client.NUM_CLASSES)
    empty_dataset = TensorDataset(
        torch.empty((0, smoke_client.NUM_FEATURES), dtype=torch.float32),
        torch.empty((0,), dtype=torch.long),
    )
    loader = DataLoader(empty_dataset, batch_size=4)

    with pytest.raises(ValueError, match="validation data loader must contain at least one example"):
        smoke_client._evaluate(model, loader)


def test_smoke_job_reads_persisted_adaptive_blend(tmp_path):
    smoke_job = _load_smoke_job_module()
    checkpoint_path = (
        tmp_path
        / "server"
        / "simulate_job"
        / "app_server"
        / smoke_job.DefaultCheckpointFileName.GLOBAL_MODEL
    )
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(
        {
            "model": {},
            "meta_props": {smoke_job.AdaptiveMetaKey.BLEND_FACTOR: 0.125},
        },
        checkpoint_path,
    )

    blend, resolved_path = smoke_job._final_checkpoint_blend(str(tmp_path))

    assert blend == pytest.approx(0.125)
    assert resolved_path == checkpoint_path


def test_smoke_job_rejects_missing_persisted_blend(tmp_path):
    smoke_job = _load_smoke_job_module()
    checkpoint_path = (
        tmp_path
        / "server"
        / "simulate_job"
        / "app_server"
        / smoke_job.DefaultCheckpointFileName.GLOBAL_MODEL
    )
    checkpoint_path.parent.mkdir(parents=True)
    torch.save({"model": {}, "meta_props": {}}, checkpoint_path)

    with pytest.raises(RuntimeError, match="missing 'adaptive_blend_factor'"):
        smoke_job._final_checkpoint_blend(str(tmp_path))
