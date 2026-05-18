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

import pytest

HAS_FEDBPT_EXPORT_DEPS = importlib.util.find_spec("cma") is not None and importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_FEDBPT_EXPORT_DEPS, reason="FedBPT job export dependencies are not installed")


def _load_fedbpt_job_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    job_path = os.path.join(repo_root, "research", "fed-bpt", "job.py")
    spec = importlib.util.spec_from_file_location("fedbpt_job", job_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert any("custom/fedbpt_train.py" in c["args"].get("script", "") for c in client["components"])
