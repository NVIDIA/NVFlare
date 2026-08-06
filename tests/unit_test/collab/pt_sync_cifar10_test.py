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

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HAS_PT_DEPS = all(importlib.util.find_spec(dep) is not None for dep in ("torch", "torchvision"))
pytestmark = pytest.mark.skipif(not HAS_PT_DEPS, reason="PyTorch example dependencies are not installed")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE_ROOT = _REPO_ROOT / "examples" / "advanced" / "collab" / "pt_sync_cifar10"
_MODULE_NAMES = ("data", "model", "fedavg", "fedprox", "scaffold", "prepare_data")


@pytest.fixture
def example_modules():
    original_modules = {name: sys.modules.pop(name, None) for name in _MODULE_NAMES}
    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_EXAMPLE_ROOT))
    try:
        data = importlib.import_module("data")
        fedavg = importlib.import_module("fedavg")
        scaffold = importlib.import_module("scaffold")
        prepare_data = importlib.import_module("prepare_data")
        yield SimpleNamespace(data=data, fedavg=fedavg, scaffold=scaffold, prepare_data=prepare_data)
    finally:
        sys.path[:] = original_sys_path
        for name in _MODULE_NAMES:
            sys.modules.pop(name, None)
            if original_modules[name] is not None:
                sys.modules[name] = original_modules[name]


class _FakeCifar10:
    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return len(self.targets)


def _prepare_args(data_root):
    return SimpleNamespace(data_root=str(data_root), num_clients=2, alpha=0.5, seed=7, overwrite=True)


def _seed_existing_generation(data_root):
    splits_dir = data_root / "splits"
    splits_dir.mkdir(parents=True)
    np.save(splits_dir / "site-1.npy", np.arange(10, dtype=np.int64))
    np.save(splits_dir / "site-2.npy", np.arange(10, 20, dtype=np.int64))
    np.save(splits_dir / "site-3.npy", np.array([20], dtype=np.int64))
    manifest = {
        "format_version": 1,
        "dataset": "CIFAR10",
        "num_clients": 3,
        "train_size": 21,
        "site_counts": {"site-1": 10, "site-2": 10, "site-3": 1},
    }
    (data_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _mock_cifar10_and_partitions(monkeypatch, module):
    train_dataset = _FakeCifar10(np.arange(20) % 2)
    test_dataset = _FakeCifar10(np.arange(4) % 2)
    partitions = [
        np.arange(0, 20, 2, dtype=np.int64),
        np.arange(1, 20, 2, dtype=np.int64),
    ]
    monkeypatch.setattr(
        module.datasets,
        "CIFAR10",
        lambda **kwargs: train_dataset if kwargs["train"] else test_dataset,
    )
    monkeypatch.setattr(module, "partition_data", lambda *args, **kwargs: partitions)
    return partitions


def test_model_and_scaffold_controls_use_local_step_weights(example_modules):
    torch = pytest.importorskip("torch")
    updates = {
        "site-1": {
            "weights": {"weight": torch.tensor([1.0, 3.0]), "counter": torch.tensor(1)},
            "control_delta": {"weight": torch.tensor([2.0])},
            "num_steps": 1,
            "train_loss": 0.5,
        },
        "site-2": {
            "weights": {"weight": torch.tensor([5.0, 7.0]), "counter": torch.tensor(2)},
            "control_delta": {"weight": torch.tensor([6.0])},
            "num_steps": 3,
            "train_loss": 1.5,
        },
    }

    averaged_weights, averaged_loss = example_modules.fedavg.weighted_average(updates, min_clients=2)
    averaged_controls = example_modules.scaffold.weighted_control_average(updates)

    torch.testing.assert_close(averaged_weights["weight"], torch.tensor([4.0, 6.0]))
    assert averaged_weights["counter"].item() == 1
    assert averaged_loss == pytest.approx(1.25)
    torch.testing.assert_close(averaged_controls["weight"], torch.tensor([5.0]))


def test_overwrite_publishes_manifest_last_and_validates_counts(monkeypatch, tmp_path, example_modules):
    data_root = tmp_path / "cifar10"
    _seed_existing_generation(data_root)
    expected_partitions = _mock_cifar10_and_partitions(monkeypatch, example_modules.prepare_data)
    manifest_path = data_root / "manifest.json"
    real_replace = example_modules.prepare_data.os.replace
    published = []

    def tracking_replace(source, destination):
        destination = Path(destination)
        if destination.suffix == ".npy":
            assert not manifest_path.exists()
        published.append(destination.name)
        real_replace(source, destination)

    monkeypatch.setattr(example_modules.prepare_data.os, "replace", tracking_replace)

    example_modules.prepare_data.prepare_data(_prepare_args(data_root))

    assert published == ["site-1.npy", "site-2.npy", "manifest.json"]
    assert not (data_root / "splits" / "site-3.npy").exists()
    for site_index, expected in enumerate(expected_partitions, start=1):
        np.testing.assert_array_equal(np.load(data_root / "splits" / f"site-{site_index}.npy"), expected)

    manifest = example_modules.data.validate_prepared_data(data_root, num_clients=2)
    assert manifest["split_seed"] == 7

    np.save(data_root / "splits" / "site-1.npy", expected_partitions[0][:-1])
    with pytest.raises(ValueError, match="manifest records 10"):
        example_modules.data.validate_prepared_data(data_root, num_clients=2)


def test_interrupted_overwrite_leaves_generation_invalid(monkeypatch, tmp_path, example_modules):
    data_root = tmp_path / "cifar10"
    _seed_existing_generation(data_root)
    _mock_cifar10_and_partitions(monkeypatch, example_modules.prepare_data)
    real_replace = example_modules.prepare_data.os.replace

    def fail_during_publish(source, destination):
        if Path(destination).name == "site-2.npy":
            raise OSError("simulated publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(example_modules.prepare_data.os, "replace", fail_during_publish)

    with pytest.raises(OSError, match="simulated publication failure"):
        example_modules.prepare_data.prepare_data(_prepare_args(data_root))

    assert not (data_root / "manifest.json").exists()
    with pytest.raises(FileNotFoundError, match="run prepare_data.py first"):
        example_modules.data.validate_prepared_data(data_root, num_clients=2)
