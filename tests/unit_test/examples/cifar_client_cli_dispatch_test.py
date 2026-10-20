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

import copy
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

ROOT = Path(__file__).parents[3]
CIFAR_ROOT = ROOT / "examples/advanced/cifar10/pt"
SOURCE_DIR = CIFAR_ROOT / "src"
ALGORITHMS = ("fedavg", "fedopt", "fedprox")


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("script", ["client.py", "job.py"])
def test_shipped_cifar_cli_help(algorithm, script):
    path = CIFAR_ROOT / "cifar10-sim" / f"cifar10_{algorithm}" / script
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(SOURCE_DIR), environment.get("PYTHONPATH", "")))
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=path.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    help_text = " ".join(result.stdout.split())
    if script == "job.py":
        assert "--execution_lifetime {job,task}" in help_text
        assert "CPU-only" in help_text and "NVFlare 2.10+" in help_text
        assert "data-worker RNG state is not restored" in help_text
    else:
        assert "or 1%" in help_text


@pytest.fixture(params=ALGORITHMS)
def client(request, monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE_DIR))

    def local_module(name):
        return name in {"data", "model", "train_utils", "task_checkpoint"} or name.startswith("data.")

    originals = {name: sys.modules.pop(name) for name in list(sys.modules) if local_module(name)}
    try:
        path = CIFAR_ROOT / "cifar10-sim" / f"cifar10_{request.param}" / "client.py"
        spec = importlib.util.spec_from_file_location(f"cifar_dispatch_{request.param}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        for name in list(sys.modules):
            if local_module(name):
                sys.modules.pop(name)
        sys.modules.update(originals)


def test_job_mode_submits_last_trained_model_after_global_evaluation(client, monkeypatch):
    from nvflare.app_common.abstract.fl_model import ParamsType
    from nvflare.app_common.app_constant import AlgorithmConstants

    model = torch.nn.Linear(2, 2)
    initial = copy.deepcopy(model.state_dict())
    evaluation_model = {name: value + 7 for name, value in initial.items()}
    training = client.flare.FLModel(
        params=initial, current_round=0, total_rounds=1, meta={AlgorithmConstants.FEDPROX_MU: 0.01}
    )
    assignments = iter(
        [
            ("train", training),
            ("validate", client.flare.FLModel(params=evaluation_model)),
            ("submit_model", client.flare.FLModel()),
        ]
    )
    current_task = None

    def receive():
        nonlocal current_task
        current_task, value = next(assignments)
        return value

    trained = {}

    def save(*args):
        trained.update(copy.deepcopy(model.state_dict()))

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.eye(2).repeat(2, 1), torch.tensor([0, 1, 0, 1])), batch_size=2
    )
    restore, checkpoint, send = Mock(), Mock(side_effect=save), Mock()
    monkeypatch.setattr(client, "DEVICE", "cpu")
    monkeypatch.setattr(client, "ModerateCNN", lambda: model)
    monkeypatch.setattr(client, "create_datasets", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(client, "create_data_loaders", lambda *args, **kwargs: (loader, object()))
    monkeypatch.setattr(client, "evaluate", lambda *args, **kwargs: 0.75)
    monkeypatch.setattr(client, "restore_training_state", restore)
    monkeypatch.setattr(client, "save_training_state", checkpoint)
    monkeypatch.setattr(client, "SummaryWriter", Mock)
    monkeypatch.setattr(client.flare, "init", Mock())
    monkeypatch.setattr(client.flare, "get_site_name", lambda: "site-1")
    monkeypatch.setattr(client.flare, "is_running", Mock(side_effect=[True, True, True, False]))
    monkeypatch.setattr(client.flare, "receive", receive)
    monkeypatch.setattr(client.flare, "is_train", lambda: current_task == "train")
    monkeypatch.setattr(client.flare, "is_evaluate", lambda: current_task == "validate")
    monkeypatch.setattr(client.flare, "is_submit_model", lambda: current_task == "submit_model")
    monkeypatch.setattr(client.flare, "get_task_name", lambda: current_task)
    monkeypatch.setattr(client.flare, "send", send)

    client.main(client.define_parser(["--num_workers", "0", "--aggregation_epochs", "1", "--no_lr_scheduler"]))

    assert send.call_count == 3
    assert send.call_args_list[0].args[0].params_type == ParamsType.DIFF
    assert send.call_args_list[1].args[0].params is None
    assert send.call_args_list[1].args[0].metrics == {"accuracy": 0.75}
    submission = send.call_args_list[2].args[0]
    assert submission.params_type == ParamsType.FULL
    assert trained.keys() == submission.params.keys()
    for name, value in trained.items():
        assert torch.equal(submission.params[name], value)
        assert not torch.equal(submission.params[name], evaluation_model[name])
    assert restore.call_count == checkpoint.call_count == 1
