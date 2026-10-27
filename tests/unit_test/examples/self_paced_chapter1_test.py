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
import math
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CHAPTER_CODE = (
    REPO_ROOT
    / "examples/tutorials/self-paced-training/part-1_federated_learning_introduction"
    / "chapter-1_running_federated_learning_applications/code"
)
pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")


@pytest.fixture
def lesson(monkeypatch):
    # Isolate the flat example modules from same-named modules in other examples.
    module_names = {path.stem for path in CHAPTER_CODE.glob("*.py")}
    original = {name: sys.modules[name] for name in module_names if name in sys.modules}
    for name in original:
        sys.modules.pop(name)
    monkeypatch.syspath_prepend(str(CHAPTER_CODE))
    monkeypatch.chdir(CHAPTER_CODE)
    try:
        yield importlib.import_module
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        sys.modules.update(original)


def test_site_and_split_data_are_reproducible_and_distinct(lesson):
    import torch

    dataset = lesson("data").SyntheticImageDataset
    first = dataset("site-1", "train", 200)
    repeated = dataset("site-1", "train", 200)
    other_site = dataset("site-2", "train", 200)
    evaluation = dataset("site-1", "eval", 100)
    assert torch.equal(first.images, repeated.images)
    assert torch.equal(first.labels, repeated.labels)
    assert not torch.equal(first.images, other_site.images)
    assert not torch.equal(first.images[:100], evaluation.images)
    assert set(first.sample_ids).isdisjoint(other_site.sample_ids)
    assert set(first.sample_ids).isdisjoint(evaluation.sample_ids)
    assert torch.bincount(first.labels).tolist() == [20] * 10


@pytest.mark.parametrize("batch_size,epochs,expected_steps", [(32, 1, 7), (16, 1, 13), (32, 2, 14)])
def test_training_reports_actual_steps_and_evaluation_preserves_weights(lesson, batch_size, epochs, expected_steps):
    import torch

    training = lesson("training")
    model = lesson("model").create_model()
    train_loader, eval_loader = training.make_loaders("site-1", batch_size)
    before = training.copy_weights(model)
    steps, loss = training.train(model, train_loader, epochs, 0.1)
    assert steps == expected_steps
    assert math.isfinite(loss)
    after = training.copy_weights(model)
    assert any(not torch.equal(before[key], after[key]) for key in before)
    accuracy = training.evaluate(model, eval_loader)
    assert 0 <= accuracy <= 100
    assert not model.training
    assert all(torch.equal(after[key], value) for key, value in model.state_dict().items())


def test_client_evaluation_does_not_train_and_training_reports_received_model_score(lesson, monkeypatch):
    from nvflare.app_common.abstract.fl_model import FLModel

    client = lesson("client")
    model = client.create_model()
    incoming = FLModel(params=client.copy_weights(model), current_round=0)
    tasks = iter(["train", "evaluate"])
    current = {"task": None}

    def receive():
        current["task"] = next(tasks)
        return incoming

    monkeypatch.setattr(sys, "argv", ["client.py"])
    monkeypatch.setattr(client.flare, "init", Mock())
    monkeypatch.setattr(client.flare, "get_site_name", lambda: "site-1")
    monkeypatch.setattr(client.flare, "is_running", Mock(side_effect=[True, True, False]))
    monkeypatch.setattr(client.flare, "receive", receive)
    monkeypatch.setattr(client.flare, "is_train", lambda: current["task"] == "train")
    monkeypatch.setattr(client.flare, "is_evaluate", lambda: current["task"] == "evaluate")
    send = Mock()
    monkeypatch.setattr(client.flare, "send", send)
    train = Mock(return_value=(7, 1.5))
    monkeypatch.setattr(client, "train", train)
    monkeypatch.setattr(client, "evaluate", Mock(side_effect=[12.5, 56.0, 31.0]))
    client.main()
    train.assert_called_once()
    trained, evaluated = [call.args[0] for call in send.call_args_list]
    assert trained.metrics == {"accuracy": 12.5}
    assert trained.meta["NUM_STEPS_CURRENT_ROUND"] == 7
    assert trained.params
    assert evaluated.params is None
    assert evaluated.metrics == {"accuracy": 31.0}
    assert evaluated.meta["num_eval_samples"] == 100


def test_minimal_recipe_does_not_schedule_evaluation(lesson, tmp_path):
    module = lesson("job")
    module.create_recipe(client_script="client_train_only.py").export(str(tmp_path))
    server_path = next(tmp_path.rglob("config_fed_server.json"))
    server = json.loads(server_path.read_text())
    assert len(server["workflows"]) == 1
    assert server["workflows"][0]["path"].endswith(".FedAvg")


def test_client_recipe_exports_distinct_arguments_for_each_site(lesson, tmp_path):
    lesson("job_custom_clients").create_recipe().export(str(tmp_path))
    job = tmp_path / "chapter1-custom-clients"
    meta = json.loads((job / "meta.json").read_text())
    expected = {
        "site-1": ["--learning_rate", "0.1", "--batch_size", "32"],
        "site-2": ["--learning_rate", "0.05", "--batch_size", "16"],
    }
    for site, arguments in expected.items():
        app = next(app for app, sites in meta["deploy_map"].items() if site in sites)
        config = json.loads((job / app / "config/config_fed_client.json").read_text())
        executor = config["executors"][0]["executor"]
        assert executor["args"]["task_script_args"] == arguments
        assert executor["args"]["task_script_path"] == "client.py"
        for module in ("client.py", "model.py", "data.py", "training.py"):
            assert (job / app / "custom" / module).is_file()


def test_server_recipe_exports_stopping_and_final_evaluation(lesson, tmp_path):
    lesson("job_custom_server").create_recipe(num_rounds=8, stop_accuracy=75.0, patience=3).export(str(tmp_path))
    config = json.loads(next(tmp_path.rglob("config_fed_server.json")).read_text())
    training, evaluation = config["workflows"]
    assert training["args"]["num_rounds"] == 8
    assert training["args"]["stop_cond"] == "accuracy >= 75.0"
    assert training["args"]["patience"] == 3
    assert evaluation["path"].endswith(".CrossSiteModelEval")
    assert evaluation["args"]["submit_model_task_name"] == ""
