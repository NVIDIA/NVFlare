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
from argparse import Namespace
from types import SimpleNamespace

import pytest

from nvflare.apis.fl_constant import SystemConfigs
from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.app_common.default_component_policy import DEFAULT_CLASS_ALLOW_LIST
from nvflare.app_common.widgets.component_path_authorizer import CLASS_ALLOW_LIST, ComponentPathAuthorizer
from nvflare.fuel.common.excepts import ComponentNotAuthorized
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.job_config.task_execution import TASK_EXECUTOR_PATH
from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator
from nvflare.private.fed.client.task_worker_executor import TaskWorkerExecutor
from nvflare.private.json_configer import ConfigError


@pytest.fixture(autouse=True)
def _reset_config_service():
    ConfigService.reset()
    yield
    ConfigService.reset()


def _authorize(config_dict, config_ctx, node, *, authorizer):
    del config_ctx
    try:
        authorizer.authorize_component_config(config_dict, node)
    except UnsafeComponentError as e:
        return str(e)
    return ""


def _configurator(tmp_path, config, allow_list):
    config_file = tmp_path / "config_fed_client.json"
    config_file.write_text(json.dumps(config))
    args = Namespace(
        sp_scheme="grpc",
        sp_target="localhost:8002",
        client_name="site-1",
        parent_url=None,
        job_id="job-1",
        workspace=str(tmp_path),
    )
    workspace = SimpleNamespace(
        get_app_custom_dir=lambda job_id: str(tmp_path / job_id / "custom"),
        get_app_config_dir=lambda job_id: str(tmp_path / job_id / "config"),
    )
    configurator = ClientJsonConfigurator(
        workspace_obj=workspace,
        config_file_name=str(config_file),
        args=args,
        app_root=str(tmp_path),
    )
    ConfigService.add_section(
        SystemConfigs.RESOURCES_CONF,
        {CLASS_ALLOW_LIST: allow_list},
    )
    configurator.set_component_build_authorizer(
        _authorize,
        authorizer=ComponentPathAuthorizer(),
    )
    return configurator


def _task_config(executor):
    return {
        "format_version": 2,
        "execution_lifetime": "task",
        "executors": [{"tasks": ["train"], "executor": executor}],
        "components": [],
        "task_data_filters": [],
        "task_result_filters": [],
    }


def test_runtime_task_supervisor_rejects_original_executor_before_import(tmp_path, monkeypatch):
    marker = tmp_path / "executor-imported.txt"
    module_path = tmp_path / "forbidden_executor.py"
    module_path.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported')\n"
        "from nvflare.apis.executor import Executor\n"
        "class ForbiddenExecutor(Executor):\n"
        "    def execute(self, task_name, shareable, fl_ctx, abort_signal):\n"
        "        return shareable\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    forbidden_path = "forbidden_executor.ForbiddenExecutor"
    configurator = _configurator(tmp_path, _task_config({"path": forbidden_path, "args": {}}), allow_list=[])

    assert configurator.config_data["executors"][0]["executor"] == {"path": forbidden_path, "args": {}}

    with pytest.raises(ComponentNotAuthorized, match="forbidden_executor.ForbiddenExecutor.*allow_list"):
        configurator.configure()

    assert not marker.exists()


def test_runtime_task_supervisor_rejects_worker_component_before_import(tmp_path, monkeypatch):
    marker = tmp_path / "component-imported.txt"
    module_path = tmp_path / "forbidden_component.py"
    module_path.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported')\n"
        "class ForbiddenComponent:\n"
        "    pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    forbidden_path = "forbidden_component.ForbiddenComponent"
    config = _task_config(
        {
            "path": "nvflare.app_common.np.np_trainer.NPTrainer",
            "args": {"helper_id": "helper"},
        }
    )
    config["components"] = [{"id": "helper", "path": forbidden_path, "args": {}}]
    configurator = _configurator(
        tmp_path,
        config,
        allow_list=["nvflare.app_common.np.np_trainer.NPTrainer"],
    )

    with pytest.raises(ComponentNotAuthorized, match="forbidden_component.ForbiddenComponent.*allow_list"):
        configurator.configure()

    assert not marker.exists()


def test_runtime_creates_framework_supervisor_without_allow_list_entry(tmp_path):
    executor_path = "nvflare.app_common.np.np_trainer.NPTrainer"
    configurator = _configurator(
        tmp_path,
        _task_config({"path": executor_path, "args": {}}),
        allow_list=[executor_path],
    )

    configurator.configure()

    supervisor = configurator.runner_config.task_router.task_table["train"]
    assert type(supervisor) is TaskWorkerExecutor
    assert TASK_EXECUTOR_PATH == f"{TaskWorkerExecutor.__module__}.{TaskWorkerExecutor.__name__}"
    assert TaskWorkerExecutor.__module__ == "nvflare.private.fed.client.task_worker_executor"
    assert TASK_EXECUTOR_PATH not in DEFAULT_CLASS_ALLOW_LIST
    assert configurator.config_data["executors"][0]["executor"]["path"] == executor_path


def test_job_cannot_select_framework_supervisor(tmp_path):
    with pytest.raises(ConfigError, match="framework-managed and cannot be selected by a job"):
        _configurator(
            tmp_path,
            _task_config({"path": TASK_EXECUTOR_PATH, "args": {}}),
            allow_list=[TASK_EXECUTOR_PATH],
        )
