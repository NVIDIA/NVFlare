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
import tempfile
from pathlib import Path


def _load_recipe_class():
    recipe_path = (
        Path(__file__).parents[5]
        / "examples"
        / "advanced"
        / "bionemo"
        / "task_fitting"
        / "job_inference"
        / "bionemo_inference_recipe.py"
    )
    spec = importlib.util.spec_from_file_location("bionemo_inference_recipe", recipe_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BioNeMoInferenceRecipe


def test_bionemo_inference_recipe_exports_model_free_task_config(tmp_path):
    BioNeMoInferenceRecipe = _load_recipe_class()
    client_script = tmp_path / "client.py"
    client_script.write_text("# test client script\n")

    recipe = BioNeMoInferenceRecipe(
        name="esm2_embeddings",
        min_clients=2,
        task_script=str(client_script),
        task_args="--checkpoint-path /tmp/model --data-root /tmp/data",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        recipe.job.export_job(tmpdir)
        job_dir = os.path.join(tmpdir, "esm2_embeddings")

        with open(os.path.join(job_dir, "app", "config", "config_fed_server.json")) as f:
            server_config = json.load(f)
        with open(os.path.join(job_dir, "app", "config", "config_fed_client.json")) as f:
            client_config = json.load(f)

    assert server_config["components"] == []
    assert server_config["workflows"][0]["path"] == (
        "nvflare.app_common.workflows.cmd_task_controller.CmdTaskController"
    )
    assert server_config["workflows"][0]["args"] == {
        "task_name": "infer",
        "persistor_id": "",
    }
    assert client_config["executors"][0]["tasks"] == ["infer"]
    assert client_config["executors"][0]["executor"]["args"]["params_exchange_format"] == "raw"
    assert client_config["executors"][0]["executor"]["args"]["server_expected_format"] == "raw"
