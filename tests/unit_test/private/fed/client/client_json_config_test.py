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
from types import SimpleNamespace

import pytest

from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator, _ExecutorDef
from nvflare.private.json_configer import ConfigError


class _PlainExecutor(Executor):
    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        return Shareable()


def _executor_def(tasks, executor):
    definition = _ExecutorDef()
    definition.tasks = tasks
    definition.executor = executor
    return definition


def _configurator_with(executors):
    # The validation is intentionally independent of scanner/workspace state. Constructing
    # only the executor list keeps these tests focused on the raw-config runtime policy.
    configurator = object.__new__(ClientJsonConfigurator)
    configurator.executors = executors
    return configurator


def _in_process_executor():
    return ClientAPIExecutor(execution_mode="in_process", task_script_path="train.py")


def _external_executor():
    return ClientAPIExecutor(execution_mode="external_process", command="python custom/train.py")


def _configurator_from_file(config_file):
    workspace_root = config_file.parent
    args = SimpleNamespace(
        sp_scheme="grpc",
        sp_target="localhost:8002",
        client_name="site-1",
        parent_url=None,
        job_id="job-1",
        workspace=str(workspace_root),
    )
    workspace = SimpleNamespace(
        get_app_custom_dir=lambda job_id: str(workspace_root / job_id / "custom"),
        get_app_config_dir=lambda job_id: str(workspace_root / job_id / "config"),
    )
    return ClientJsonConfigurator(
        workspace_obj=workspace,
        config_file_name=str(config_file),
        args=args,
        app_root=str(workspace_root),
    )


@pytest.mark.parametrize(
    "executor",
    [
        {"name": "MultiProcessExecutor"},
        {"name": "PTMultiProcessExecutor"},
        {"path": "nvflare.app_common.executors.multi_process_executor.MultiProcessExecutor"},
        {"path": "nvflare.app_opt.pt.multi_process_executor.PTMultiProcessExecutor"},
        {"path": "nvflare.app_opt.pt.PTMultiProcessExecutor"},
        {"class_path": "nvflare.app_common.pt.pt_multi_process_executor.PTMultiProcessExecutor"},
    ],
)
def test_removed_multi_process_executor_reports_migration_during_configure(tmp_path, executor, monkeypatch, caplog):
    monkeypatch.setenv("NVFLARE_SECURE_LOGGING", "true")
    config_file = tmp_path / "config_fed_client.json"
    config_file.write_text(json.dumps({"format_version": 2, "executors": [{"tasks": ["train"], "executor": executor}]}))
    configurator = _configurator_from_file(config_file)

    with pytest.raises(
        ConfigError, match=r"Executor .* was removed in NVFlare 2\.10.*Client API with external torchrun"
    ):
        configurator.configure()
    assert "Use the Client API with external torchrun." in caplog.text


class TestClientAPIExecutorRuntimeValidation:
    def test_raw_client_config_is_rejected_during_configure(self, tmp_path):
        config_file = tmp_path / "config_fed_client.json"
        config_file.write_text(
            json.dumps(
                {
                    "format_version": 2,
                    "executors": [
                        {
                            "tasks": ["train"],
                            "executor": {
                                "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
                                "args": {"execution_mode": "in_process", "task_script_path": "train.py"},
                            },
                        },
                        {
                            "tasks": ["evaluate"],
                            "executor": {
                                "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
                                "args": {
                                    "execution_mode": "external_process",
                                    "command": "python custom/train.py",
                                },
                            },
                        },
                    ],
                }
            )
        )
        configurator = _configurator_from_file(config_file)

        with pytest.raises(ConfigError, match="only one ClientAPIExecutor.*per client job"):
            configurator.configure()

    def test_rejects_two_executors_with_same_mode(self):
        configurator = _configurator_with(
            [
                _executor_def(["train"], _in_process_executor()),
                _executor_def(["evaluate"], _in_process_executor()),
            ]
        )

        with pytest.raises(ConfigError, match="only one ClientAPIExecutor.*per client job"):
            configurator._validate_client_api_executors()

    def test_rejects_two_executors_with_different_modes(self):
        configurator = _configurator_with(
            [
                _executor_def(["train"], _in_process_executor()),
                _executor_def(["evaluate"], _external_executor()),
            ]
        )

        with pytest.raises(ConfigError, match="configured modes:.*in_process.*external_process"):
            configurator._validate_client_api_executors()

    def test_does_not_restrict_other_executor_types(self):
        configurator = _configurator_with(
            [
                _executor_def(["train"], _PlainExecutor()),
                _executor_def(["evaluate"], _PlainExecutor()),
                _executor_def(["submit_model"], _external_executor()),
            ]
        )

        configurator._validate_client_api_executors()
