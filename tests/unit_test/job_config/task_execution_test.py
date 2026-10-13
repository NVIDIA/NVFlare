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
from contextlib import nullcontext

import numpy as np
import pytest

from nvflare.apis.filter import Filter
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.job_config.api import FedJob
from nvflare.job_config.defs import FilterType
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
    assert config["format_version"] == 3
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


def test_task_export_accepts_gpu_server_and_zero_gpu_client_memory_settings(tmp_path):
    job = _job(
        "task",
        {
            "resource_spec": {
                "server": {"num_of_gpus": 1},
                "@default": {"num_of_gpus": 0, "mem_per_gpu_in_GiB": 24},
            }
        },
    )
    job.to_clients(_executor())
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


def test_clearing_recipe_exec_params_preserves_task_lifetime_and_decomposer_registration(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("pass\n")
    recipe = FedAvgRecipe(
        min_clients=1,
        model=np.zeros(2),
        framework=FrameworkType.NUMPY,
        train_script=str(script),
        execution_lifetime="task",
    )
    recipe.add_decomposers(["nvflare.app_common.decomposers.common_decomposers.FLModelDecomposer"])
    recipe.export(str(tmp_path / "export"), client_exec_params={})
    config = json.loads((tmp_path / "export" / recipe.name / "app" / "config" / "config_fed_client.json").read_text())
    assert config["execution_lifetime"] == "task"
    assert config["format_version"] == 3
    plan = prepare_task_execution(config)
    assert "decomposer_reg" in {c["id"] for c in plan.job_components}
    assert "decomposer_reg" in {c["id"] for c in plan.executors[0].components}


def test_invalid_component_scope_does_not_attach_a_filter_before_raising():
    class Passthrough(Filter):
        def process(self, shareable, fl_ctx):
            return shareable

    job = _job()
    task_filter = Passthrough()
    with pytest.raises(ValueError, match="execution_scope"):
        job.to_clients(task_filter, filter_type=FilterType.TASK_RESULT, execution_scope="job")
    app = job._deploy_map["@ALL"].app_config
    assert app.task_result_filters == []
    job.to_clients(task_filter, filter_type=FilterType.TASK_RESULT)
    assert len(app.task_result_filters) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_fedjob_temporary_params_preserve_site_lifetimes_and_restore(tmp_path, fail):
    job = _job()
    for site, lifetime in (("site-1", "task"), ("site-2", "job")):
        job.to(_executor(), site)
        job.set_execution_lifetime(lifetime, target=site)
        job.to({"timeout": 600}, site)
    before = {target: dict(app.app_config.additional_params) for target, app in job._deploy_map.items()}

    expected = pytest.raises(RuntimeError, match="deployment failed") if fail else nullcontext()
    with expected:
        with job.temporary_exec_params(server_exec_params={}, client_exec_params={}):
            job.export_job(str(tmp_path))
            for site, lifetime, version in (("site-1", "task", 3), ("site-2", "job", 2)):
                path = tmp_path / job.name / f"app_{site}" / "config" / "config_fed_client.json"
                config = json.loads(path.read_text())
                assert config["execution_lifetime"] == lifetime
                assert config["format_version"] == version
                assert "timeout" not in config
            assert job._deploy_map["server"].app_config.additional_params == {}
            if fail:
                raise RuntimeError("deployment failed")

    after = {target: dict(app.app_config.additional_params) for target, app in job._deploy_map.items()}
    assert after == before


def test_fedjob_temporary_lifetime_override_restores_declared_task_job(tmp_path):
    job = _job("task")
    job.to_clients(_executor())
    with job.temporary_exec_params(client_exec_params={"execution_lifetime": "job"}):
        job.export_job(str(tmp_path / "temporary"))
        path = tmp_path / "temporary" / job.name / "app" / "config" / "config_fed_client.json"
        assert json.loads(path.read_text())["format_version"] == 2
    job.export_job(str(tmp_path / "restored"))
    path = tmp_path / "restored" / job.name / "app" / "config" / "config_fed_client.json"
    config = json.loads(path.read_text())
    assert config["execution_lifetime"] == "task"
    assert config["format_version"] == 3
