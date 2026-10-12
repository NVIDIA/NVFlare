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


import json

import numpy as np
import pytest

from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.job_config.api import FedJob
from nvflare.job_config.task_execution import prepare_task_execution
from nvflare.recipe.fedavg import FedAvgRecipe


def _job(lifetime="job", meta=None):
    job = FedJob(name="task-example", execution_lifetime=lifetime, meta_props=meta)
    job.to_server({"format_version": 2})
    return job


def _executor():
    return ClientAPIExecutor(execution_mode="in_process", task_script_path="train.py")


def test_export_preserves_job_default_and_supports_task_lifetime(tmp_path):
    job = _job()
    job.to_clients(_executor())
    job.export_job(str(tmp_path))
    path = tmp_path / job.name / "app" / "config" / "config_fed_client.json"
    assert "execution_lifetime" not in json.loads(path.read_text())
    job.set_execution_lifetime("task")
    converter = ConvertToFedEvent(["analytix_log_stats"])
    job.to_clients(converter, id="converter")
    job.export_job(str(tmp_path))
    config = json.loads(path.read_text())
    assert config["execution_lifetime"] == "task"
    assert prepare_task_execution(config).job_components[0]["id"] == "converter"
    assert "execution_scope" not in config["components"][0]
    assert config["executors"][0]["executor"]["path"].endswith(".ClientAPIExecutor")


def test_export_validates_gpu_only_for_task_app_sites(tmp_path):
    job = _job(meta={"resource_spec": {"site-2": {"num_of_gpus": 1}}})
    job.to(_executor(), "site-1")
    job.to(_executor(), "site-2")
    job.set_execution_lifetime("task", target="site-1")
    job.export_job(str(tmp_path))
    job.set_execution_lifetime("task", target="site-2")
    with pytest.raises(ValueError, match="GPUs"):
        job.export_job(str(tmp_path))


def test_all_clients_task_export_rejects_site_specific_gpu(tmp_path):
    job = _job("task", {"resource_spec": {"site-3": {"num_of_gpus": 1}}})
    job.to_clients(_executor())
    with pytest.raises(ValueError, match="GPUs"):
        job.export_job(str(tmp_path))


def test_invalid_task_configuration_keeps_previous_export(tmp_path):
    job = _job("task")
    job.to_clients(_executor())
    job.export_job(str(tmp_path))
    path = tmp_path / job.name / "app" / "config" / "config_fed_client.json"
    previous = path.read_bytes()
    job.to_clients({"task_execution": {"environment_variables": ["SECRET"]}})
    with pytest.raises(ValueError, match="site configuration"):
        job.export_job(str(tmp_path))
    assert path.read_bytes() == previous


def test_recipe_exports_native_task_profile_with_resident_federation_widget(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("pass\n")
    recipe = FedAvgRecipe(
        name="native-task",
        min_clients=1,
        num_rounds=2,
        framework=FrameworkType.NUMPY,
        model=np.zeros(2),
        train_script=str(script),
        execution_lifetime="task",
    )
    recipe._ensure_client_apps_prepared()
    recipe._job.export_job(str(tmp_path / "export"))
    config = json.loads((tmp_path / "export" / recipe.name / "app" / "config" / "config_fed_client.json").read_text())
    assert config["execution_lifetime"] == "task"
    assert "event_to_fed" in {c["id"] for c in prepare_task_execution(config).job_components}
    assert all("execution_scope" not in c for c in config["components"])
    assert config["executors"][0]["executor"]["args"]["execution_mode"] == "in_process"
