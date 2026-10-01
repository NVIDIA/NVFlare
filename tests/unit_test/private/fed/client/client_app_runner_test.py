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

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.private.fed.client import client_app_runner
from nvflare.private.fed.simulator.simulator_app_runner import SimulatorClientAppRunner


@pytest.mark.parametrize("mode", ["process", "docker", "k8s", "slurm"])
def test_client_runner_injects_site_launcher_with_selected_job_backend(tmp_path, monkeypatch, mode):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    args = Namespace(
        workspace=str(tmp_path),
        client_name="site-1",
        job_id="job-1",
        client_config="config_fed_client.json",
        set=[],
        launch_mode=mode,
    )
    config = SimpleNamespace(handlers=[])
    conf = Mock(runner_config=config)
    monkeypatch.setattr(client_app_runner, "ClientJsonConfigurator", lambda **_kwargs: conf)
    inject = Mock()
    monkeypatch.setattr(client_app_runner, "configure_task_launchers", inject)
    privacy = Mock(is_policy_defined=lambda: False)
    monkeypatch.setattr(client_app_runner, "create_privacy_manager", lambda *_args, **_kwargs: privacy)
    monkeypatch.setattr(client_app_runner.PrivacyService, "initialize", Mock())
    runner = client_app_runner.ClientAppRunner()
    manager = Mock(new_context=lambda: FLContext())
    monkeypatch.setattr(runner, "create_run_manager", lambda *_args: manager)
    client_runner = Mock()
    monkeypatch.setattr(client_app_runner, "ClientRunner", lambda **_kwargs: client_runner)
    client = Mock()

    assert runner.create_client_runner(str(tmp_path), args, "config", client, False) is client_runner
    conf.configure.assert_called_once()
    inject.assert_called_once_with(config, job_launcher_mode=mode)
    assert client.runner_config is config
    manager.add_handler.assert_called_once_with(client_runner)


def test_simulator_reports_process_backend_even_without_cli_mode():
    assert SimulatorClientAppRunner().get_job_launcher_mode(Namespace()) == "process"
