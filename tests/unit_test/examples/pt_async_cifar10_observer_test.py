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
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("completed_version", [0, 1])
def test_horizon_stop_evaluates_only_published_cifar_snapshots(tmp_path, monkeypatch, completed_version):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("tensorboard")
    example_dir = Path(__file__).resolve().parents[3] / "examples/advanced/collab/pt_async_cifar10"
    model_spec = importlib.util.spec_from_file_location("pt_async_cifar10_model", example_dir / "model.py")
    model_module = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model_module)
    monkeypatch.setitem(sys.modules, "model", model_module)
    observer_spec = importlib.util.spec_from_file_location(
        "pt_async_cifar10_observer", example_dir / "async_aggregator.py"
    )
    observer_module = importlib.util.module_from_spec(observer_spec)
    observer_spec.loader.exec_module(observer_module)

    observer = observer_module.Cifar10Observer(data_root=str(tmp_path), num_rounds=3)
    observer._run_dir = str(tmp_path)
    observer._eval_snapshot_dir = str(tmp_path / "eval_snapshots")
    Path(observer._eval_snapshot_dir).mkdir()
    for version in range(completed_version + 1):
        torch.save({"w": torch.tensor([version])}, Path(observer._eval_snapshot_dir) / f"model_version_{version}.pt")
    evaluated = []
    monkeypatch.setattr(
        observer,
        "_evaluate_model",
        lambda snapshot, round_index: evaluated.append((snapshot["w"].item(), round_index)),
    )

    final_model = {"w": torch.tensor([99])}
    observer.on_finish(final_model, completed_version)

    assert evaluated == [(version, version - 1) for version in range(completed_version + 1)]
    saved = torch.load(tmp_path / "final_model/global_model.pt", weights_only=True)
    assert torch.equal(saved["w"], final_model["w"])
