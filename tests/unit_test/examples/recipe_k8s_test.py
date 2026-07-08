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
import json
import os
import sys

import pytest

HAS_PT_DEPS = all(importlib.util.find_spec(dep) is not None for dep in ("torch", "torchvision"))
pytestmark = pytest.mark.skipif(not HAS_PT_DEPS, reason="PyTorch example dependencies are not installed")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXAMPLE_DIR = os.path.join(REPO_ROOT, "examples", "advanced", "recipe-k8s")


@pytest.fixture
def load_example_module():
    loaded_modules = []
    original_model_module = sys.modules.pop("model", None)
    sys.path.insert(0, EXAMPLE_DIR)

    def _load(file_name: str, module_name: str):
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(EXAMPLE_DIR, file_name))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        loaded_modules.append(module_name)
        return module

    yield _load

    sys.path.pop(0)
    for module_name in loaded_modules:
        sys.modules.pop(module_name, None)
    if original_model_module is not None:
        sys.modules["model"] = original_model_module
    else:
        sys.modules.pop("model", None)


def _parse_validated(job_module, *extra_args):
    parser = job_module.define_parser()
    args = parser.parse_args(["--startup-kit", "/tmp/admin", *extra_args])
    job_module.validate_args(parser, args)
    return args


@pytest.mark.parametrize(
    "extra_args, expected_error",
    [
        (["--image", "shared", "--num-rounds", "0"], "must be greater than 0"),
        (["--image", "shared", "--site-2-name", "site-1"], "must identify different clients"),
        (["--image", "shared", "--site-1-name", "default"], "is reserved"),
        ([], "provide --image"),
        (["--image", "shared", "--site-1-gpus", "2"], "must be 0 or 1"),
        (["--image", "shared", "--data-dir", "/data/cifar ten"], "must not contain whitespace"),
    ],
)
def test_invalid_cli_values_use_argparse_errors(load_example_module, capsys, extra_args, expected_error):
    job_module = load_example_module("job.py", "recipe_k8s_job_invalid")

    with pytest.raises(SystemExit) as exc_info:
        _parse_validated(job_module, *extra_args)

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert expected_error in stderr
    assert "Traceback" not in stderr


def test_recipe_export_contains_cifar_code_and_k8s_metadata(load_example_module, monkeypatch, tmp_path):
    job_module = load_example_module("job.py", "recipe_k8s_job_export")
    monkeypatch.chdir(EXAMPLE_DIR)
    args = _parse_validated(
        job_module,
        "--image",
        "registry.example/shared:dev",
        "--site-2-image",
        "registry.example/site-2:dev",
        "--site-1-gpus",
        "1",
        "--server-image",
        "registry.example/server:dev",
        "--server-cpu",
        "4",
        "--server-memory",
        "8Gi",
    )

    recipe = job_module.create_recipe(args)
    recipe.export(str(tmp_path))
    job_root = tmp_path / "cifar10-k8s"

    with open(job_root / "meta.json") as meta_file:
        meta = json.load(meta_file)
    assert meta["deploy_map"] == {
        "app_server": ["server"],
        "app_site-1": ["site-1"],
        "app_site-2": ["site-2"],
    }
    assert meta["resource_spec"] == {
        "site-1": {"num_of_gpus": 1},
        "site-2": {"num_of_gpus": 0},
    }
    assert meta["launcher_spec"]["site-1"]["k8s"]["image"] == "registry.example/shared:dev"
    assert meta["launcher_spec"]["site-2"]["k8s"]["image"] == "registry.example/site-2:dev"
    assert meta["launcher_spec"]["default"]["k8s"]["image"] == "registry.example/server:dev"
    assert meta["launcher_spec"]["default"]["k8s"]["cpu"] == "4"
    assert meta["launcher_spec"]["default"]["k8s"]["memory"] == "8Gi"

    for app_name in ("app_server", "app_site-1", "app_site-2"):
        assert (job_root / app_name / "custom" / "model.py").is_file()
    for app_name in ("app_site-1", "app_site-2"):
        assert (job_root / app_name / "custom" / "client.py").is_file()

    with open(job_root / "app_site-1" / "config" / "config_fed_client.json") as config_file:
        site_1_config = json.load(config_file)
    executor_args = site_1_config["executors"][0]["executor"]["args"]
    assert executor_args["task_script_path"] == "client.py"
    assert "--site-index 0" in executor_args["task_script_args"]
    assert "--num-sites 2" in executor_args["task_script_args"]
    assert "--download" in executor_args["task_script_args"]
    assert "--require-gpu" in executor_args["task_script_args"]

    with open(job_root / "app_site-2" / "config" / "config_fed_client.json") as config_file:
        site_2_config = json.load(config_file)
    site_2_args = site_2_config["executors"][0]["executor"]["args"]["task_script_args"]
    assert "--site-index 1" in site_2_args
    assert "--num-sites 2" in site_2_args
    assert "--require-gpu" not in site_2_args


def test_cifar_model_and_partitions(load_example_module, monkeypatch):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    client_module = load_example_module("client.py", "recipe_k8s_client")
    model_module = sys.modules["model"]
    model = model_module.Cifar10Net()

    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 10)

    site_0 = client_module.partition_dataset(list(range(12)), site_index=0, num_sites=2, max_samples=3)
    site_1 = client_module.partition_dataset(list(range(12)), site_index=1, num_sites=2, max_samples=0)
    assert site_0.indices == [0, 2, 4]
    assert site_1.indices == [1, 3, 5, 7, 9, 11]
    assert set(site_0.indices).isdisjoint(site_1.indices)

    data_loader = DataLoader(
        TensorDataset(torch.rand(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])),
        batch_size=2,
    )
    loss = client_module.train(model, data_loader, torch.device("cpu"), local_epochs=1)
    accuracy = client_module.evaluate(model, data_loader, torch.device("cpu"))
    assert loss > 0
    assert 0 <= accuracy <= 1

    monkeypatch.setattr(client_module.torch.cuda, "is_available", lambda: False)
    assert client_module.select_device(require_gpu=False).type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        client_module.select_device(require_gpu=True)

    monkeypatch.setattr(client_module.torch.cuda, "is_available", lambda: True)
    assert client_module.select_device(require_gpu=False).type == "cpu"
    assert client_module.select_device(require_gpu=True).type == "cuda"
