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
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CHAPTER = (
    REPO_ROOT
    / "examples/tutorials/self-paced-training/part-1_federated_learning_introduction"
    / "chapter-1_running_federated_learning_applications"
)
SECTIONS = {
    "basic": "01.1_running_federated_learning_job",
    "conversion": "01.2_convert_deep_learning_to_federated_learning",
    "server": "01.3_server_side_customization",
    "clients": "01.4_client_side_customization",
    "tensorboard": "01.5_experiment_tracking/01.5.1_experiment_tracking_with_tensorboard",
    "mlflow": "01.5_experiment_tracking/01.5.2_experiment_tracking_with_mlflow",
    "structure": "01.6_job_structure_and_configuration",
    "logging": "01.7_logging",
    "recap": "01.8_recap",
}
pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")


@pytest.fixture
def lesson(monkeypatch, request, tmp_path):
    import torch

    original_threads = torch.get_num_threads()
    source = CHAPTER / SECTIONS[getattr(request, "param", "basic")] / "code"
    # Run from a copy with no sibling lessons or parent-level Python project available.
    code = tmp_path / "example-code"
    code.mkdir()
    for path in source.glob("*.py"):
        shutil.copy2(path, code / path.name)
    # Isolate the flat example modules from same-named modules in other examples.
    module_names = {path.stem for path in code.glob("*.py")}
    original = {name: sys.modules[name] for name in module_names if name in sys.modules}
    for name in original:
        sys.modules.pop(name)
    monkeypatch.syspath_prepend(str(code))
    monkeypatch.chdir(code)
    try:
        yield importlib.import_module
    finally:
        torch.set_num_threads(original_threads)
        for name in module_names:
            sys.modules.pop(name, None)
        sys.modules.update(original)


def test_site_and_split_data_are_reproducible_and_distinct(lesson):
    import torch

    dataset = lesson("prepare_data").SyntheticImageDataset
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

    training = lesson("client")
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


@pytest.mark.parametrize("lesson", ["conversion"], indirect=True)
def test_minimal_recipe_does_not_schedule_evaluation(lesson, tmp_path):
    module = lesson("job")
    module.create_recipe().export(str(tmp_path / "exported"))
    server_path = next(tmp_path.rglob("config_fed_server.json"))
    server = json.loads(server_path.read_text())
    assert len(server["workflows"]) == 1
    assert server["workflows"][0]["path"].endswith(".FedAvg")


@pytest.mark.parametrize("lesson", ["clients"], indirect=True)
def test_client_recipe_exports_distinct_arguments_for_each_site(lesson, tmp_path):
    lesson("job").create_recipe().export(str(tmp_path))
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
        for module in ("client.py", "model.py", "prepare_data.py"):
            assert (job / app / "custom" / module).is_file()


@pytest.mark.parametrize("lesson", ["server"], indirect=True)
@pytest.mark.parametrize("aggregation", ["median", "fedavg"])
def test_server_recipe_exports_aggregation_stopping_and_final_evaluation(lesson, tmp_path, aggregation):
    lesson("job").create_recipe(num_rounds=8, stop_accuracy=75.0, patience=3, aggregation=aggregation).export(
        str(tmp_path)
    )
    config_path = next(tmp_path.rglob("config_fed_server.json"))
    config = json.loads(config_path.read_text())
    training, evaluation = config["workflows"]
    if aggregation == "median":
        assert training["args"]["aggregator"]["path"] == "aggregator.MedianAggregator"
        packaged = config_path.parent.parent / "custom/aggregator.py"
        assert packaged.read_bytes() == (Path.cwd() / "aggregator.py").read_bytes()
    else:
        assert training["args"].get("aggregator") is None
    meta = json.loads((config_path.parents[2] / "meta.json").read_text())
    assert meta["min_clients"] == 3
    assert training["args"]["num_rounds"] == 8
    assert training["args"]["stop_cond"] == "accuracy >= 75.0"
    assert training["args"]["patience"] == 3
    assert evaluation["path"].endswith(".CrossSiteModelEval")
    assert evaluation["args"]["submit_model_task_name"] == ""


@pytest.mark.parametrize("lesson", ["server"], indirect=True)
def test_custom_server_uses_coordinate_medians_and_separate_mean_accuracy(lesson):
    import numpy as np

    from nvflare.apis.fl_constant import FLMetaKey
    from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

    aggregator = lesson("aggregator").MedianAggregator()
    clients = []
    for weights, bias, accuracy, steps in (([1, 30], 0, 20, 1), ([2, 10], 10, 40, 2), ([100, 20], 1, 90, 100)):
        model = FLModel(
            params={"weight": np.array(weights, dtype=np.float32), "bias": np.array([bias], dtype=np.float32)},
            metrics={"accuracy": accuracy},
            meta={FLMetaKey.NUM_STEPS_CURRENT_ROUND: steps},
        )
        clients.append(model)
        aggregator.accept_model(model)
    result = aggregator.aggregate_model()
    # The outlier's large step count must not dominate the custom aggregation.
    np.testing.assert_array_equal(result.params["weight"], [2, 20])
    np.testing.assert_array_equal(result.params["bias"], [1])
    assert result.params["weight"].dtype == np.float32
    assert result.params_type == ParamsType.FULL
    assert result.metrics == {"accuracy": 50.0}
    np.testing.assert_array_equal(clients[2].params["weight"], [100, 20])

    # A subsequent round must not reuse models or metrics from the previous round.
    aggregator.accept_model(FLModel(params={"weight": np.array([9.0])}, metrics={"accuracy": 75.0}))
    next_result = aggregator.aggregate_model()
    np.testing.assert_array_equal(next_result.params["weight"], [9.0])
    assert next_result.metrics == {"accuracy": 75.0}
    with pytest.raises(ValueError, match="No client models"):
        aggregator.aggregate_model()


@pytest.mark.parametrize("lesson", ["server"], indirect=True)
def test_custom_server_reset_and_full_weight_contract(lesson):
    import numpy as np

    from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

    aggregator = lesson("aggregator").MedianAggregator()
    aggregator.accept_model(FLModel(params={"weight": np.array([1.0])}, metrics={"accuracy": 10.0}))
    aggregator.reset_stats()
    with pytest.raises(ValueError, match="No client models"):
        aggregator.aggregate_model()
    with pytest.raises(ValueError, match="full model weights"):
        aggregator.accept_model(FLModel(params={"weight": np.array([1.0])}, params_type=ParamsType.DIFF))


@pytest.mark.parametrize(
    "lesson",
    [
        "basic",
        "conversion",
        "server",
        "clients",
        pytest.param(
            "tensorboard",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("tensorboard") is None, reason="TensorBoard is not installed"
            ),
        ),
        pytest.param(
            "mlflow",
            marks=pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="MLflow is not installed"),
        ),
        "structure",
        "logging",
        "recap",
    ],
    indirect=True,
)
def test_section_exports_its_own_client_and_dependencies(lesson, tmp_path):
    lesson("job").create_recipe().export(str(tmp_path / "exported"))
    configs = list((tmp_path / "exported").rglob("config_fed_client.json"))
    assert configs
    for path in configs:
        config = json.loads(path.read_text())
        executor = config["executors"][0]["executor"]
        assert executor["args"]["task_script_path"] == "client.py"
        custom = path.parent.parent / "custom"
        for name in ("client.py", "model.py", "prepare_data.py"):
            assert (custom / name).is_file()
            assert (custom / name).read_bytes() == (Path.cwd() / name).read_bytes()


@pytest.mark.parametrize("lesson", ["conversion"], indirect=True)
def test_conversion_evaluation_variant_exports_its_matching_client(lesson, tmp_path):
    lesson("job").create_recipe(client_script="client_evaluate.py").export(str(tmp_path / "exported"))
    server = json.loads(next((tmp_path / "exported").rglob("config_fed_server.json")).read_text())
    assert server["workflows"][1]["path"].endswith(".CrossSiteModelEval")
    path = next((tmp_path / "exported").rglob("config_fed_client.json"))
    config = json.loads(path.read_text())
    assert config["executors"][0]["executor"]["args"]["task_script_path"] == "client_evaluate.py"
    assert (path.parent.parent / "custom/client_evaluate.py").is_file()


@pytest.mark.parametrize("lesson", ["structure"], indirect=True)
def test_structure_per_site_variant_exports_both_sites(lesson, tmp_path):
    lesson("job").create_recipe(per_site=True).export(str(tmp_path / "exported"))
    folder = tmp_path / "exported/chapter1-structure-per-site"
    meta = json.loads((folder / "meta.json").read_text())
    for site, batch_size in (("site-1", "32"), ("site-2", "16")):
        app = next(app for app, targets in meta["deploy_map"].items() if site in targets)
        config = json.loads((folder / app / "config/config_fed_client.json").read_text())
        args = config["executors"][0]["executor"]["args"]["task_script_args"]
        assert args[args.index("--batch_size") + 1] == batch_size


@pytest.mark.parametrize("lesson", ["server"], indirect=True)
def test_median_aggregator_requires_accuracy_metric(lesson):
    from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

    aggregator = lesson("aggregator").MedianAggregator()
    with pytest.raises(ValueError, match="accuracy"):
        aggregator.accept_model(FLModel(params={"w": 1.0}, params_type=ParamsType.FULL))


@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="MLflow is not installed")
@pytest.mark.parametrize("lesson", ["mlflow"], indirect=True)
def test_mlflow_store_is_beside_job_script(lesson, tmp_path):
    module = lesson("job")
    module.create_recipe().export(str(tmp_path / "exported"))
    text = "".join(path.read_text() for path in (tmp_path / "exported").rglob("*.json"))
    assert str(Path(module.__file__).resolve().parent / "mlflow.db") in text
