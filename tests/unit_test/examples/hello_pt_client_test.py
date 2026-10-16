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
from collections import Counter
from types import SimpleNamespace

import pytest

from tests.hello_pt_test_utils import load_hello_pt_module

HAS_PT = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_PT, reason="PyTorch is not installed")


@pytest.fixture(scope="module")
def torchvision_module():
    try:
        return pytest.importorskip("torchvision")
    except RuntimeError as error:
        pytest.skip(f"torchvision cannot be imported with this PyTorch installation: {error}")


def _load_hello_pt_module(file_name):
    with load_hello_pt_module(file_name) as module:
        return module


def test_hello_pt_evaluate_rejects_empty_data_loader():
    import torch

    client_module = _load_hello_pt_module("client.py")
    model = torch.nn.Linear(1, 2)

    with pytest.raises(ValueError, match="Evaluation data_loader produced no samples"):
        client_module.evaluate(net=model, data_loader=[], device="cpu")

    assert not model.training


def test_hello_pt_evaluate_uses_evaluation_mode():
    import torch

    client_module = _load_hello_pt_module("client.py")

    class ModeTrackingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.observed_modes = []

        def forward(self, inputs):
            self.observed_modes.append(self.training)
            return torch.zeros((len(inputs), 2))

    model = ModeTrackingModel()
    loader = [(torch.ones((2, 1)), torch.zeros(2, dtype=torch.long))]

    client_module.evaluate(model, loader, "cpu")

    assert model.observed_modes == [False]


def test_cifar_data_loaders_use_prepared_data_without_downloading(monkeypatch, tmp_path, torchvision_module):
    import torch

    client_module = _load_hello_pt_module("client.py")
    calls = []

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return torch.zeros((3, 32, 32)), index

    monkeypatch.setattr(torchvision_module.datasets, "CIFAR10", FakeDataset)
    batch_dir = tmp_path / "cifar-10-batches-py"
    batch_dir.mkdir()
    for name in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]:
        (batch_dir / name).write_bytes(b"test dataset stub")

    train_loader, test_loader = client_module.create_data_loaders(
        "cifar10", "site-1", 20, 10, 2, 0, data_root=str(tmp_path)
    )

    assert len(train_loader.dataset) == len(test_loader.dataset) == 2
    assert [(call["train"], call["download"]) for call in calls] == [(True, False), (False, False)]
    assert all(call["root"] == str(tmp_path) for call in calls)


@pytest.mark.parametrize("cache_state", ["missing", "partial", "empty"])
def test_cifar_data_loaders_explain_missing_preparation(tmp_path, cache_state):
    client_module = _load_hello_pt_module("client.py")
    if cache_state != "missing":
        batch_dir = tmp_path / "cifar-10-batches-py"
        batch_dir.mkdir()
        names = ["data_batch_1"]
        if cache_state == "empty":
            names = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]
        for name in names:
            (batch_dir / name).touch()

    with pytest.raises(FileNotFoundError, match="python prepare_data.py --data_root") as error:
        client_module.create_data_loaders("cifar10", "site-1", 20, 10, 2, 0, data_root=str(tmp_path))

    assert str(tmp_path) in str(error.value)
    assert "test_batch" in str(error.value)


def test_synthetic_data_is_deterministic_and_disjoint_by_site_and_split():
    import torch

    data_module = _load_hello_pt_module("prepare_data.py")
    dataset_type = data_module.SyntheticImageDataset

    train_1 = dataset_type(site_name="site-1", split="train", size=20)
    train_1_repeat = dataset_type(site_name="site-1", split="train", size=20)
    train_2 = dataset_type(site_name="site-2", split="train", size=20)
    eval_1 = dataset_type(site_name="site-1", split="eval", size=20)

    assert torch.equal(train_1.images, train_1_repeat.images)
    assert torch.equal(train_1.labels, train_1_repeat.labels)
    assert not torch.equal(train_1.images, train_2.images)
    assert not torch.equal(train_1.images, eval_1.images)

    id_sets = [set(dataset.sample_ids) for dataset in (train_1, train_2, eval_1)]
    assert all(left.isdisjoint(right) for index, left in enumerate(id_sets) for right in id_sets[index + 1 :])


def test_synthetic_data_is_balanced_and_encodes_the_label():
    data_module = _load_hello_pt_module("prepare_data.py")
    dataset = data_module.SyntheticImageDataset(site_name="site-1", split="train", size=100)

    assert Counter(dataset.labels.tolist()) == {label: 10 for label in range(data_module.NUM_CLASSES)}
    for image, label_tensor in dataset:
        label = label_tensor.item()
        row = 2 + (label // 5) * 16
        column = 1 + (label % 5) * 6
        assert image[:, row : row + 5, column : column + 5].mean().item() == pytest.approx(1.0)


def test_recipe_model_initialization_is_reproducible_and_isolated():
    import torch

    model_module = _load_hello_pt_module("model.py")

    rng_state = torch.random.get_rng_state()
    first_model = model_module.create_model()
    first = first_model.state_dict()
    assert torch.equal(torch.random.get_rng_state(), rng_state)

    second_model = model_module.create_model()
    second = second_model.state_dict()

    assert first_model.seed == second_model.seed == model_module.MODEL_SEED
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert model_module.SimpleNetwork().seed is None


def test_prepare_data_downloads_both_cifar_splits(monkeypatch, tmp_path, capsys, torchvision_module):
    data_module = _load_hello_pt_module("prepare_data.py")
    calls = []
    monkeypatch.setattr(torchvision_module.datasets, "CIFAR10", lambda **kwargs: calls.append(kwargs))

    data_module.main(["--data_root", str(tmp_path)])

    assert [(call["train"], call["download"]) for call in calls] == [(True, True), (False, True)]
    assert all(call["root"] == str(tmp_path) for call in calls)
    assert f"CIFAR-10 is ready under {tmp_path}" in capsys.readouterr().out


def test_cifar_preflight_rejects_present_but_empty_files(tmp_path):
    data_module = _load_hello_pt_module("prepare_data.py")
    batch_dir = tmp_path / "cifar-10-batches-py"
    batch_dir.mkdir()
    for name in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]:
        (batch_dir / name).touch()
    with pytest.raises(FileNotFoundError, match="Missing or empty CIFAR-10 files"):
        data_module.validate_cifar10(str(tmp_path))


@pytest.fixture
def task_client(tmp_path, monkeypatch):
    import torch

    with load_hello_pt_module("client_task.py") as module:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["client_task.py"])
        monkeypatch.setattr(module.flare, "init", lambda: None)
        monkeypatch.setattr(module.flare, "get_site_name", lambda: "site-1")
        monkeypatch.setattr(module.flare, "get_job_id", lambda: "job-1")
        monkeypatch.setattr(
            module.flare,
            "system_info",
            lambda: {
                "task_checkpoint": {
                    "input": str(tmp_path / "training_state.pt") if (tmp_path / "training_state.pt").exists() else None,
                    "output": str(tmp_path / "checkpoint.pending"),
                }
            },
        )
        monkeypatch.setattr(module.flare, "is_train", lambda: True)
        monkeypatch.setattr(module.flare, "is_evaluate", lambda: False)
        model = torch.nn.Linear(1, 2)
        monkeypatch.setattr(module, "create_model", lambda **kwargs: model)
        monkeypatch.setattr(
            module.flare, "receive", lambda: module.flare.FLModel(params=model.state_dict(), current_round=1)
        )
        loader = SimpleNamespace(generator=torch.Generator())
        monkeypatch.setattr(module, "create_data_loaders", lambda **kwargs: (loader, loader))
        monkeypatch.setattr(module, "evaluate", lambda *args: 0.0)
        monkeypatch.setattr(module, "train_one_round", lambda *args: 0)
        monkeypatch.setattr(module, "SummaryWriter", object)
        yield module


@pytest.mark.parametrize(
    "checkpoint,error",
    [
        (None, "Missing training checkpoint"),
        ({"job_id": "other", "site_name": "site-1", "round": 0}, "different job or site"),
        ({"job_id": "job-1", "site_name": "other", "round": 0}, "different job or site"),
        ({"job_id": "job-1", "site_name": "site-1", "round": -1}, "immediately preceding round"),
        ({"job_id": "job-1", "site_name": "site-1", "round": 1}, "immediately preceding round"),
    ],
)
def test_task_training_rejects_missing_stale_or_foreign_checkpoint(
    task_client, tmp_path, monkeypatch, checkpoint, error
):
    import torch

    if checkpoint is not None:
        torch.save(checkpoint, tmp_path / "training_state.pt")
    monkeypatch.setattr(task_client.flare, "send", lambda *args: pytest.fail("must not send a result"))
    with pytest.raises(RuntimeError, match=error):
        task_client.main()


@pytest.mark.parametrize("fresh_training", [False, True])
def test_task_evaluation_and_fresh_training_leave_checkpoint_untouched(
    task_client, tmp_path, monkeypatch, fresh_training
):
    checkpoint = tmp_path / "training_state.pt"
    checkpoint.write_bytes(b"not a PyTorch checkpoint")
    if fresh_training:
        monkeypatch.setattr(sys, "argv", ["client_task.py", "--no-resume_training"])
    else:
        monkeypatch.setattr(task_client.flare, "is_train", lambda: False)
        monkeypatch.setattr(task_client.flare, "is_evaluate", lambda: True)
    outputs = []
    monkeypatch.setattr(task_client.flare, "send", outputs.append)
    task_client.main()
    assert len(outputs) == 1
    assert bool(outputs[0].params) == fresh_training
    assert checkpoint.read_bytes() == b"not a PyTorch checkpoint"


def test_task_checkpoint_does_not_advance_when_send_fails(task_client, tmp_path, monkeypatch):
    import torch

    checkpoint = tmp_path / "training_state.pt"
    model = task_client.create_model()
    torch.save(
        {
            "job_id": "job-1",
            "site_name": "site-1",
            "round": 0,
            "optimizer": task_client.SGD(model.parameters(), lr=0.1, momentum=0.9).state_dict(),
            "train_rng": torch.Generator().get_state(),
            "torch_rng": torch.get_rng_state(),
        },
        checkpoint,
    )
    previous = checkpoint.read_bytes()

    def fail_send(*args):
        raise OSError("result disk is full")

    monkeypatch.setattr(task_client.flare, "send", fail_send)
    with pytest.raises(OSError, match="result disk is full"):
        task_client.main()
    assert checkpoint.read_bytes() == previous
    assert torch.load(tmp_path / "checkpoint.pending", weights_only=True)["round"] == 1


def test_task_training_writes_pending_state_without_replacing_committed_input(task_client, tmp_path, monkeypatch):
    import torch

    model = task_client.create_model()
    checkpoint = tmp_path / "training_state.pt"
    torch.save(
        {
            "job_id": "job-1",
            "site_name": "site-1",
            "round": 0,
            "optimizer": task_client.SGD(model.parameters(), lr=0.1, momentum=0.9).state_dict(),
            "train_rng": torch.Generator().get_state(),
            "torch_rng": torch.get_rng_state(),
        },
        checkpoint,
    )
    previous = checkpoint.read_bytes()
    outputs = []
    monkeypatch.setattr(task_client.flare, "send", outputs.append)
    task_client.main()
    assert len(outputs) == 1
    assert checkpoint.read_bytes() == previous
    candidate = torch.load(tmp_path / "checkpoint.pending", weights_only=True)
    assert candidate["round"] == 1
    assert candidate["job_id"] == "job-1"
