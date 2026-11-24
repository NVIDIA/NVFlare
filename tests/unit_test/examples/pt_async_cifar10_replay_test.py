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

"""CIFAR replay example contracts without downloading the dataset."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nvflare.apis.simulation_plan import validate_simulation_plan


@pytest.fixture
def replay_package(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("tensorboard")
    folder = Path(__file__).resolve().parents[3] / "examples/advanced/collab"
    monkeypatch.syspath_prepend(str(folder))
    return importlib.import_module("pt_async_cifar10_reproducible")


def test_cifar_replay_plan_and_export(replay_package, tmp_path):
    job = importlib.import_module(f"{replay_package.__name__}.job")
    args = job.define_parser().parse_args(
        ["--num-clients", "1", "--num-logical-clients", "2", "--num-rounds", "2", "--buffer-size", "1"]
    )
    args.train_idx_root = str(tmp_path / "shards")
    plan = validate_simulation_plan(job.make_plan(args))
    execution = plan["async_execution"]
    assert execution["arrival_sequence"] == [1, 0]
    assert execution["logical_sequence"] == ["site-2", "site-1"]
    assert len(execution["train_seeds"]) == 2

    recipe = job.make_recipe(args)
    recipe.export(str(tmp_path / "job"))
    config = next((tmp_path / "job").rglob("config_fed_server.json"))
    server = next(item for item in json.loads(config.read_text())["components"] if item["id"] == "_server")
    assert server["path"] == "nvflare.collab.fedbuff_recipe.FedBuffWorkflow"
    assert server["args"]["num_rounds"] == 2
    assert "simulation_plan" not in config.read_text()

    args.plan_mode = "timed"
    args.horizon = 2
    timed = validate_simulation_plan(job.make_plan(args))["async_execution"]
    assert timed["timing"]["horizon"] == 2
    assert timed["timing"]["starts"] == [0.0, 0.0]


def test_cifar_replay_training_seed_resets_worker_state(replay_package, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    module = importlib.import_module(f"{replay_package.__name__}.trainer")
    site = SimpleNamespace(site_name="worker-a", is_aborted=False, call_info="test")
    monkeypatch.setattr(module, "collab", site)
    np.save(tmp_path / "site-1.npy", np.arange(8))
    client = module.Cifar10Trainer(
        data_root=str(tmp_path),
        train_idx_root=str(tmp_path),
        local_batch_size=2,
        aggregation_epochs=1,
        num_logical_clients=2,
    )
    client._dataset = torch.utils.data.TensorDataset(
        torch.arange(8, dtype=torch.float32).reshape(-1, 1),
        torch.arange(8) % 2,
    )
    client._local_model = torch.nn.Sequential(torch.nn.Dropout(p=0.5), torch.nn.Linear(1, 2))
    initial = {name: value.detach().clone() for name, value in client._local_model.state_dict().items()}

    def train(seed):
        return client.train(
            {"worker-a": 0},
            0,
            initial,
            {"worker-a": "site-1"},
            {"worker-a": seed},
        )

    first, _, first_meta = train(123)
    second, _, second_meta = train(123)
    third, _, _ = train(124)
    other_worker = module.Cifar10Trainer(
        data_root=str(tmp_path),
        train_idx_root=str(tmp_path),
        local_batch_size=2,
        aggregation_epochs=1,
        num_logical_clients=2,
    )
    other_worker._dataset = client._dataset
    other_worker._local_model = torch.nn.Sequential(torch.nn.Dropout(p=0.5), torch.nn.Linear(1, 2))
    site.site_name = "worker-b"
    on_other_worker, _, _ = other_worker.train({"worker-b": 0}, 0, initial, {"worker-b": "site-1"}, {"worker-b": 123})
    assert first_meta["train_seed"] == second_meta["train_seed"] == 123
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert all(torch.equal(first[name], on_other_worker[name]) for name in first)
    assert any(not torch.equal(first[name], third[name]) for name in first)
