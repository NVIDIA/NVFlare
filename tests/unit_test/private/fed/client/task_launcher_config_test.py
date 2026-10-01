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

import pytest

from nvflare.apis.fl_constant import SystemConfigs
from nvflare.apis.launcher import LauncherMode
from nvflare.apis.task_launcher_spec import TaskLauncherSpec
from nvflare.app_common.task_launcher.process_launcher import ProcessTaskLauncher
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.private.fed.client.client_runner import ClientRunnerConfig, TaskRouter
from nvflare.private.fed.client.task_launcher_config import build_task_launcher, configure_task_launchers
from nvflare.private.fed.client.task_worker_executor import TaskWorkerExecutor


class ConfiguredTaskLauncher(TaskLauncherSpec):
    launch_mode = LauncherMode.PROCESS.value

    def __init__(self, value=None):
        super().__init__()
        self.value = value

    def launch_task(self, request):
        raise NotImplementedError()


class JobSuppliedTaskWorkerExecutor(TaskWorkerExecutor):
    pass


@pytest.fixture(autouse=True)
def _reset_config_service():
    ConfigService.reset()
    yield
    ConfigService.reset()


def _supervisor(supervisor_type=TaskWorkerExecutor):
    return supervisor_type(
        executor={"path": "example.Executor", "args": {}},
        components=[],
    )


def _runner_config(executor):
    router = TaskRouter()
    router.add_executor(["train", "validate"], executor)
    return ClientRunnerConfig(router, {}, {})


def test_missing_site_setting_selects_process_launcher():
    assert isinstance(build_task_launcher(), ProcessTaskLauncher)


def test_site_resources_select_custom_launcher_and_arguments(monkeypatch):
    ConfigService.add_section(
        SystemConfigs.RESOURCES_CONF,
        {"task_launcher": {"path": "site.launcher.ConfiguredTaskLauncher", "args": {"value": 7}}},
    )
    monkeypatch.setattr(
        "nvflare.private.fed.client.task_launcher_config.load_class", lambda _path: ConfiguredTaskLauncher
    )

    launcher = build_task_launcher()

    assert type(launcher) is ConfiguredTaskLauncher
    assert launcher.value == 7


@pytest.mark.parametrize(
    "config, error",
    [
        ("process", TypeError),
        ({"path": ""}, ValueError),
        ({"args": []}, TypeError),
        ({"backend": "process"}, ValueError),
    ],
)
def test_invalid_site_launcher_configuration_is_rejected(config, error):
    with pytest.raises(error):
        build_task_launcher(config)


def test_configured_class_must_implement_task_launcher_spec(monkeypatch):
    monkeypatch.setattr("nvflare.private.fed.client.task_launcher_config.load_class", lambda _path: object)

    with pytest.raises(TypeError, match="must implement TaskLauncherSpec"):
        build_task_launcher({"path": "site.NotALauncher"})


def test_runtime_injects_one_shared_launcher_and_registers_lifecycle_handler(monkeypatch):
    supervisor = _supervisor()
    runner_config = _runner_config(supervisor)
    launcher = ConfiguredTaskLauncher()
    monkeypatch.setattr("nvflare.private.fed.client.task_launcher_config.build_task_launcher", lambda: launcher)

    configured = configure_task_launchers(runner_config, job_launcher_mode=LauncherMode.PROCESS.value)

    assert configured is launcher
    assert supervisor._get_task_launcher() is launcher
    assert runner_config.handlers.count(launcher) == 1


def test_runtime_does_not_inject_privileged_launcher_into_job_subclass(monkeypatch):
    runner_config = _runner_config(_supervisor(JobSuppliedTaskWorkerExecutor))
    monkeypatch.setattr(
        "nvflare.private.fed.client.task_launcher_config.build_task_launcher",
        lambda: pytest.fail("launcher must not be built for a job-supplied subclass"),
    )

    assert configure_task_launchers(runner_config) is None


def test_runtime_rejects_mismatched_job_and_task_launcher_modes(monkeypatch):
    runner_config = _runner_config(_supervisor())
    launcher = ConfiguredTaskLauncher()
    launcher.launch_mode = LauncherMode.DOCKER.value
    monkeypatch.setattr("nvflare.private.fed.client.task_launcher_config.build_task_launcher", lambda: launcher)

    with pytest.raises(RuntimeError, match="JobLauncher mode 'process'.*TaskLauncher mode 'docker'.*same launch mode"):
        configure_task_launchers(runner_config, job_launcher_mode=LauncherMode.PROCESS.value)


@pytest.mark.parametrize("job_mode", [None, "", "custom"])
def test_runtime_requires_supported_job_launcher_mode_for_task_execution(job_mode):
    runner_config = _runner_config(_supervisor())

    with pytest.raises(ValueError, match="selected JobLauncher mode must be one of"):
        configure_task_launchers(runner_config, job_launcher_mode=job_mode)


def test_runtime_requires_supported_task_launcher_mode(monkeypatch):
    runner_config = _runner_config(_supervisor())
    launcher = ConfiguredTaskLauncher()
    launcher.launch_mode = None
    monkeypatch.setattr("nvflare.private.fed.client.task_launcher_config.build_task_launcher", lambda: launcher)

    with pytest.raises(ValueError, match="configured TaskLauncher mode must be one of"):
        configure_task_launchers(runner_config, job_launcher_mode=LauncherMode.PROCESS.value)


def test_only_site_config_injects_worker_environment_names():
    ConfigService.add_section(
        SystemConfigs.RESOURCES_CONF, {"task_launcher": {"environment_variables": ["SITE_DATA_KEY"]}}
    )
    supervisor = _supervisor()
    configure_task_launchers(_runner_config(supervisor), job_launcher_mode=LauncherMode.PROCESS.value)
    assert supervisor._environment_variables == ("SITE_DATA_KEY",)


@pytest.mark.parametrize("names", ["KEY", ["invalid=name"], [7], ["NVFLARE_JOB_AUTH_TOKEN"]])
def test_invalid_site_environment_policy_is_rejected_before_launcher_import(names, monkeypatch):
    monkeypatch.setattr(
        "nvflare.private.fed.client.task_launcher_config.load_class",
        lambda _path: pytest.fail("invalid policy must fail before import"),
    )
    with pytest.raises(ValueError, match="environment_variables"):
        build_task_launcher({"environment_variables": names})
