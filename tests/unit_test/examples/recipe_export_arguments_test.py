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

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MIGRATED_RECIPE_EXAMPLES = (
    "examples/advanced/experiment-tracking/mlflow/hello-lightning-mlflow/job.py",
    "examples/advanced/experiment-tracking/mlflow/hello-pt-mlflow-client/job.py",
    "examples/advanced/experiment-tracking/mlflow/hello-pt-mlflow/job.py",
    "examples/advanced/experiment-tracking/wandb/job.py",
    "examples/advanced/gnn/finance/job.py",
    "examples/advanced/gnn/protein/job.py",
    "examples/advanced/kaplan-meier-he/job.py",
    "examples/advanced/llm_hf/job.py",
    "examples/advanced/multi-gpu/lightning/job.py",
    "examples/advanced/multi-gpu/pt/job.py",
    "examples/advanced/psi/user_email_match/job.py",
    "examples/advanced/swarm_learning/swarm_pt/job.py",
    "examples/hello-world/hello-collab/job.py",
    "examples/hello-world/hello-flower/job.py",
    "examples/hello-world/hello-log-streaming/job.py",
    "examples/hello-world/hello-numpy/job.py",
)
_REMOVED_EXPORT_OPTIONS = {
    "--export",
    "--export-dir",
    "--export_config",
    "--export_dir",
    "--export_job",
    "--export_only",
    "--job_configs",
    "--job_dir",
}


def _calls_attribute(node: ast.AST, attribute: str, receiver: str = "recipe") -> bool:
    return any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == attribute
        and isinstance(item.func.value, ast.Name)
        and item.func.value.id == receiver
        for item in ast.walk(node)
    )


def test_recipe_examples_use_system_export_arguments():
    for relative_path in _MIGRATED_RECIPE_EXAMPLES:
        source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative_path)
        declared_options = {
            argument.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        }

        assert not declared_options.intersection(_REMOVED_EXPORT_OPTIONS), relative_path
        assert not _calls_attribute(tree, "export"), relative_path
        assert _calls_attribute(tree, "execute"), relative_path


def test_swarm_export_short_circuit_preserves_existing_workspace(monkeypatch, tmp_path):
    class FakeModel:
        def __init__(self, model_path):
            self.model_path = model_path

    class FakeRecipe:
        def __init__(self, **kwargs):
            pass

        def add_client_file(self, path):
            pass

        def add_server_config(self, config):
            pass

        def add_client_config(self, config):
            pass

        def execute(self, env):
            raise SystemExit(0)

    model_module = types.ModuleType("model")
    model_module.QwenLoRAModelWrapper = FakeModel
    recipe_module = types.ModuleType("nvflare.app_opt.pt.recipes.swarm")
    recipe_module.SwarmLearningRecipe = FakeRecipe
    monkeypatch.setitem(sys.modules, "model", model_module)
    monkeypatch.setitem(sys.modules, "nvflare.app_opt.pt.recipes.swarm", recipe_module)

    workspace = tmp_path / "simulation"
    marker = workspace / "ccwf_swarm_pt_lora" / "previous-result.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep", encoding="utf-8")

    job_path = _REPO_ROOT / "examples/advanced/swarm_learning/swarm_pt/job.py"
    spec = importlib.util.spec_from_file_location("swarm_export_job_test", job_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(job_path), "--workspace", str(workspace)])

    with pytest.raises(SystemExit, match="0"):
        module.main()

    assert marker.read_text(encoding="utf-8") == "keep"
