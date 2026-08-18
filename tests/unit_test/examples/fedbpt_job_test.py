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

import builtins
import importlib.util
import json
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from nvflare.recipe import PotentialSecretWarning

HAS_FEDBPT_EXPORT_DEPS = importlib.util.find_spec("cma") is not None and importlib.util.find_spec("torch") is not None
HAS_CMA = importlib.util.find_spec("cma") is not None

# Fake credential for testing the detector -- not a real token.
FAKE_GITHUB_TOKEN = "ghp_" + "Ab1" * 12


def _load_fedbpt_job_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    job_path = os.path.join(repo_root, "research", "fed-bpt", "job.py")
    spec = importlib.util.spec_from_file_location("fedbpt_job", job_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fedbpt_recipe_module(monkeypatch):
    class FakeComponent:
        def __init__(self, *args, **kwargs):
            pass

    fake_global_es_module = ModuleType("global_es")
    fake_global_es_module.GlobalES = FakeComponent
    fake_decomposer_module = ModuleType("decomposer_widget")
    fake_decomposer_module.RegisterDecomposer = FakeComponent
    monkeypatch.setitem(sys.modules, "global_es", fake_global_es_module)
    monkeypatch.setitem(sys.modules, "decomposer_widget", fake_decomposer_module)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    recipe_path = os.path.join(repo_root, "research", "fed-bpt", "fedbpt_recipe.py")
    spec = importlib.util.spec_from_file_location("fedbpt_recipe_for_test", recipe_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fedbpt_main_parses_export_flags_before_importing_recipe(monkeypatch, tmp_path):
    exported = []

    class FakeFedBPTRecipe:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]

        def export(self, export_dir):
            exported.append((export_dir, self.name))

    fake_recipe_module = ModuleType("fedbpt_recipe")
    fake_recipe_module.FedBPTRecipe = FakeFedBPTRecipe
    real_import = builtins.__import__

    def import_recipe_and_consume_export_flags(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fedbpt_recipe":
            # Match nvflare.recipe.spec's import-time behavior closely enough to
            # prove that job.py parsed these values before they disappear.
            export_index = sys.argv.index("--export")
            del sys.argv[export_index]
            export_dir_index = sys.argv.index("--export-dir")
            del sys.argv[export_dir_index : export_dir_index + 2]
            return fake_recipe_module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_recipe_and_consume_export_flags)
    monkeypatch.setattr(sys, "argv", ["job.py", "--export", "--export-dir", str(tmp_path)])

    job_module = _load_fedbpt_job_module()
    job_module.main()

    assert exported == [(str(tmp_path), "fedbpt")]


@pytest.mark.parametrize(
    "secret_args",
    [
        {"train_args": f"--api_key {FAKE_GITHUB_TOKEN}"},
        {"extra_train_args": ["--api_key", FAKE_GITHUB_TOKEN]},
    ],
    ids=["train_args", "extra_train_args"],
)
def test_fedbpt_run_warns_for_secret_in_combined_train_args(monkeypatch, secret_args):
    recipe_module = _load_fedbpt_recipe_module(monkeypatch)
    recipe = recipe_module.FedBPTRecipe(num_clients=1, **secret_args)
    env = MagicMock()
    env.deploy.return_value = "job-id"

    with pytest.warns(PotentialSecretWarning, match="train_args") as record:
        recipe.run(env)

    assert all(FAKE_GITHUB_TOKEN not in str(warning.message) for warning in record)
    env.deploy.assert_called_once()


def test_fedbpt_runner_uses_exported_train_script_name(monkeypatch):
    recipe_module = _load_fedbpt_recipe_module(monkeypatch)
    recipe = recipe_module.FedBPTRecipe(num_clients=1)

    client_app = recipe._job._deploy_map["@ALL"]
    executor = client_app.app_config.executors[0].executor

    assert executor._command[:3] == ["python3", "-u", "custom/fedbpt_train.py"]


@pytest.mark.skipif(not HAS_FEDBPT_EXPORT_DEPS, reason="FedBPT job export dependencies are not installed")
def test_fedbpt_job_exports_recipe_config(tmp_path):
    job_module = _load_fedbpt_job_module()
    parser = job_module.define_parser()
    args, extra_args = parser.parse_known_args(
        [
            "--num_clients",
            "2",
            "--num_rounds",
            "1",
            "--seed",
            "42",
            "--model_name",
            "roberta-base",
            "--k_shot",
            "1",
            "--local_popsize",
            "2",
            "--local_iter",
            "1",
            "--eval_clients",
            "none",
        ]
    )

    src_dir = os.path.join(job_module.FEDBPT_DIR, "src")
    src_path_count = sys.path.count(src_dir)
    recipe = job_module.create_recipe(args, extra_args)
    recipe.export(str(tmp_path))

    job_dir = tmp_path / "fedbpt"
    server_config = job_dir / "app" / "config" / "config_fed_server.json"
    client_config = job_dir / "app" / "config" / "config_fed_client.json"
    custom_dir = job_dir / "app" / "custom"

    assert (job_dir / "meta.json").exists()
    assert (custom_dir / "fedbpt_train.py").exists()
    assert (custom_dir / "cma_decomposer.py").exists()

    with open(server_config) as f:
        server = json.load(f)
    with open(client_config) as f:
        client = json.load(f)

    assert server["workflows"][0]["path"] == "global_es.GlobalES"
    assert server["workflows"][0]["args"]["num_clients"] == 2
    assert server["workflows"][0]["args"]["num_rounds"] == 1
    assert any(c["path"] == "decomposer_widget.RegisterDecomposer" for c in server["components"])
    assert any(c["path"] == "decomposer_widget.RegisterDecomposer" for c in client["components"])
    assert client["executors"][0]["executor"]["args"]["command"][:3] == [
        "python3",
        "-u",
        "custom/fedbpt_train.py",
    ]
    assert sys.path.count(src_dir) == src_path_count


@pytest.mark.skipif(not HAS_CMA, reason="cma is not installed")
def test_fedbpt_cma_decomposer_serializes_range_state():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    src_dir = os.path.join(repo_root, "research", "fed-bpt", "src")
    sys.path.insert(0, src_dir)
    try:
        import cma
        from cma_decomposer import register_decomposers

        from nvflare.fuel.utils import fobs
    finally:
        sys.path.remove(src_dir)

    register_decomposers()

    decoded_range = fobs.loads(fobs.dumps(range(1, 5, 2)))
    assert list(decoded_range) == [1, 3]

    strategy = cma.CMAEvolutionStrategy(4 * [5], 1, {"ftarget": 1e-9, "seed": 5})
    decoded_strategy = fobs.loads(fobs.dumps(strategy))

    assert decoded_strategy.N == strategy.N
    assert decoded_strategy.sigma == strategy.sigma
