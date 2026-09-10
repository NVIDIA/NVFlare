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

import argparse
import ast
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.hello_pt_test_utils import load_hello_pt_module

HAS_PT = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_PT, reason="PyTorch is not installed")


def _load_job_module():
    with load_hello_pt_module("job.py") as module:
        return module


@pytest.mark.parametrize("file_name", ["job.py", "client.py"])
def test_example_module_isolated_from_another_examples_cached_sibling(monkeypatch, file_name):
    conflicting_module = SimpleNamespace(DATASET_CHOICES=("other",), DATASET_PATH="other", DEFAULT_DATASET="other")
    monkeypatch.setitem(sys.modules, "prepare_data", conflicting_module)
    conflicting_model = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "model", conflicting_model)
    original_sys_path = list(sys.path)

    with load_hello_pt_module(file_name) as module:
        parser = argparse.ArgumentParser()
        module.add_dataset_arguments(parser)
        assert vars(parser.parse_args([])) == {"dataset": "synthetic", "data_root": "/tmp/nvflare/data"}
        assert parser.parse_args(["--dataset", "cifar10"]).dataset == "cifar10"

    assert sys.modules["prepare_data"] is conflicting_module
    assert sys.modules["model"] is conflicting_model
    assert sys.path == original_sys_path


def _load_web_python_snippet(name: str) -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    component_path = os.path.join(repo_root, "web", "src", "components", "code.astro")
    with open(component_path) as component_file:
        component = component_file.read()
    match = re.search(rf"const {name} = `\n(.*?)`;", component, flags=re.DOTALL)
    assert match, f"missing website snippet {name}"
    return match.group(1)


def test_zero_flag_defaults_are_portable_and_bounded():
    job_module = _load_job_module()

    args = job_module.define_parser().parse_args([])

    assert vars(args) == {
        "data_root": "/tmp/nvflare/data",
        "dataset": "synthetic",
        "n_clients": 2,
        "num_rounds": 3,
    }


def test_legacy_synthetic_flag_keeps_selecting_the_default_dataset():
    job_module = _load_job_module()

    args = job_module.define_parser().parse_args(["--synthetic_data"])

    assert args.dataset == "synthetic"


def test_help_includes_recipe_export_options():
    job_module = _load_job_module()

    help_text = job_module.define_parser().format_help()

    assert "--export" in help_text
    assert "--export-dir EXPORT_DIR" in help_text


@pytest.mark.parametrize(
    "cell_id, expected_rounds, expected_train_args",
    [
        ("nvflare-cli-export-code", 2, ["--dataset", "synthetic"]),
        ("nvflare-cli-abort-code", 20, ["--dataset", "synthetic", "--train_size", "8192"]),
    ],
)
def test_cli_notebook_exports_bundled_job_with_log_streaming(tmp_path, cell_id, expected_rounds, expected_train_args):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    tutorial_dir = os.path.join(repo_root, "examples", "tutorials")
    with open(os.path.join(tutorial_dir, "nvflare_cli.ipynb")) as notebook_file:
        notebook = json.load(notebook_file)
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)
    command = shlex.split(cell["source"][0].removeprefix("!"))
    assert command[0] == "python"
    command[0] = sys.executable
    command[command.index("--job-dir") + 1] = str(tmp_path)
    subprocess.run(
        command,
        cwd=tutorial_dir,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((repo_root, os.environ.get("PYTHONPATH", "")))},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )

    job_dir = tmp_path / "hello-pt"
    client = json.loads((job_dir / "app/config/config_fed_client.json").read_text())
    server = json.loads((job_dir / "app/config/config_fed_server.json").read_text())
    assert client["executors"][0]["executor"]["args"]["task_script_path"] == "client.py"
    assert client["executors"][0]["executor"]["args"]["task_script_args"] == expected_train_args
    assert server["workflows"][0]["args"]["num_rounds"] == expected_rounds
    assert any(component["path"].endswith(".JobLogStreamer") for component in client["components"])
    assert any(component["path"].endswith(".JobLogReceiver") for component in server["components"])
    assert (job_dir / "meta.json").is_file()
    assert {"client.py", "model.py", "prepare_data.py"} <= {path.name for path in (job_dir / "app/custom").glob("*.py")}


def test_website_pytorch_snippets_are_internally_consistent():
    client_source = _load_web_python_snippet("clientCode_pt")
    model_source = _load_web_python_snippet("modelCode_pt")
    job_source = _load_web_python_snippet("jobCode_pt")

    for name, source in (("client.py", client_source), ("model.py", model_source), ("job.py", job_source)):
        compile(source, name, "exec")

    model_functions = {node.name for node in ast.walk(ast.parse(model_source)) if isinstance(node, ast.FunctionDef)}
    assert "create_model" in model_functions

    client_options = {
        call.args[0].value
        for call in ast.walk(ast.parse(client_source))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_argument"
        and call.args
        and isinstance(call.args[0], ast.Constant)
    }
    train_args = [
        keyword.value.value
        for call in ast.walk(ast.parse(job_source))
        if isinstance(call, ast.Call)
        for keyword in call.keywords
        if keyword.arg == "train_args" and isinstance(keyword.value, ast.Constant)
    ]
    assert len(train_args) == 1
    assert {token for token in shlex.split(train_args[0]) if token.startswith("--")} <= client_options


def test_main_reports_simulation_success(tmp_path, monkeypatch, capsys):
    job_module = _load_job_module()
    result_dir = tmp_path / "simulation-result"
    result_dir.mkdir()
    calls = []

    run = SimpleNamespace(
        get_result=lambda: calls.append(("get_result",)) or str(result_dir),
    )
    env = object()
    recipe = SimpleNamespace(execute=lambda value: calls.append(("execute", value)) or run)
    monkeypatch.setattr(job_module, "create_recipe", lambda args: recipe)
    monkeypatch.setattr(job_module, "SimEnv", lambda num_clients: env)

    result = job_module.main([])

    assert result == str(result_dir)
    assert calls == [("execute", env), ("get_result",)]
    output = capsys.readouterr().out
    assert "Simulation completed successfully." in output
    assert "Job Status is: None" not in output


def test_default_recipe_uses_final_global_evaluation(monkeypatch):
    job_module = _load_job_module()
    calls = []
    recipe = object()
    recipe_kwargs = {}

    def make_recipe(**kwargs):
        recipe_kwargs.update(kwargs)
        return recipe

    monkeypatch.setattr(job_module, "FedAvgRecipe", make_recipe)
    monkeypatch.setattr(job_module, "create_model", lambda: "model")
    monkeypatch.setattr(job_module, "add_final_global_evaluation", lambda value: calls.append(("final", value)))

    result = job_module.create_recipe(job_module.define_parser().parse_args([]))

    assert result is recipe
    assert recipe_kwargs["model"] == "model"
    assert recipe_kwargs["min_clients"] == 2
    assert recipe_kwargs["num_rounds"] == 3
    assert recipe_kwargs["train_script"] == "client.py"
    assert recipe_kwargs["train_args"] == ["--dataset", "synthetic"]
    assert calls == [("final", recipe)]


def test_cifar_main_rejects_missing_data_before_creating_environment(tmp_path, monkeypatch):
    job_module = _load_job_module()
    monkeypatch.setattr(job_module, "SimEnv", lambda **kwargs: pytest.fail("simulation must not start"))
    monkeypatch.setattr(job_module, "create_recipe", lambda args: pytest.fail("recipe must not be constructed"))

    with pytest.raises(SystemExit, match="python prepare_data.py --data_root"):
        job_module.main(["--dataset", "cifar10", "--data_root", str(tmp_path)])


def test_cifar_cli_export_does_not_require_local_data(tmp_path):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    example_dir = os.path.join(repo_root, "examples", "hello-world", "hello-pt")
    remote_cache = str(tmp_path / "remote site's cache")
    subprocess.run(
        [
            sys.executable,
            "job.py",
            "--export",
            "--export-dir",
            str(tmp_path / "export"),
            "--dataset",
            "cifar10",
            "--data_root",
            remote_cache,
        ],
        cwd=example_dir,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((repo_root, os.environ.get("PYTHONPATH", "")))},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    client_config = tmp_path / "export" / "hello-pt" / "app" / "config" / "config_fed_client.json"
    executor_args = json.loads(client_config.read_text())["executors"][0]["executor"]["args"]
    assert executor_args["task_script_args"] == ["--dataset", "cifar10", "--data_root", remote_cache]
    assert not os.path.exists(remote_cache)


def test_cifar_recipe_preserves_data_root_with_spaces(monkeypatch):
    job_module = _load_job_module()
    recipe_kwargs = {}
    recipe = object()

    monkeypatch.setattr(job_module, "FedAvgRecipe", lambda **kwargs: recipe_kwargs.update(kwargs) or recipe)
    monkeypatch.setattr(job_module, "create_model", lambda: "model")
    monkeypatch.setattr(job_module, "add_final_global_evaluation", lambda value: None)

    args = job_module.define_parser().parse_args(["--dataset", "cifar10", "--data_root", "/data/cifar cache"])
    job_module.create_recipe(args)

    assert recipe_kwargs["train_args"] == [
        "--dataset",
        "cifar10",
        "--data_root",
        "/data/cifar cache",
    ]


def test_cifar_remains_an_explicit_option(monkeypatch):
    job_module = _load_job_module()
    calls = []
    recipe = object()
    recipe_kwargs = {}

    def make_recipe(**kwargs):
        recipe_kwargs.update(kwargs)
        return recipe

    monkeypatch.setattr(job_module, "FedAvgRecipe", make_recipe)
    monkeypatch.setattr(job_module, "create_model", lambda: "model")
    monkeypatch.setattr(job_module, "add_final_global_evaluation", lambda value: calls.append(("final", value)))
    args = job_module.define_parser().parse_args(["--dataset", "cifar10", "--data_root", "/data/cifar"])

    job_module.create_recipe(args)

    assert recipe_kwargs["train_args"] == ["--dataset", "cifar10", "--data_root", "/data/cifar"]
    assert calls == [("final", recipe)]


def test_export_serializes_the_recipe_model_seed(tmp_path, monkeypatch):
    example_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "hello-world", "hello-pt")
    )
    monkeypatch.chdir(example_dir)

    with load_hello_pt_module("job.py") as job_module:
        recipe = job_module.create_recipe(job_module.define_parser().parse_args([]))
        recipe.export(job_dir=str(tmp_path))

    config_path = tmp_path / "hello-pt" / "app" / "config" / "config_fed_server.json"
    with config_path.open() as config_file:
        config = json.load(config_file)
    model_configs = [
        component["args"]["model"]
        for component in config["components"]
        if component.get("path", "").endswith("PTFileModelPersistor")
    ]
    assert model_configs == [{"path": "model.SimpleNetwork", "args": {"seed": 202610}}]
