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

import argparse
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from nvflare.dashboard import cli as dashboard_cli


def _parse_dashboard_args(argv):
    parser = argparse.ArgumentParser(prog="nvflare dashboard")
    dashboard_cli.define_dashboard_parser(parser)
    return parser.parse_args(argv)


def test_start_uses_installed_package_entrypoint(capsys):
    args = _parse_dashboard_args(["--start", "-i", "nvflare-parent:test", "--cred", "admin@example.com:pw:org"])
    container = SimpleNamespace(id="container-1", status="running", reload=Mock(), logs=Mock())
    client = SimpleNamespace(images=SimpleNamespace(pull=Mock()), containers=SimpleNamespace(run=Mock()))
    client.containers.run.return_value = container

    with (
        patch.object(dashboard_cli.docker, "from_env", return_value=client),
        patch.object(dashboard_cli.time, "sleep"),
    ):
        dashboard_cli.start(args)

    client.containers.run.assert_called_once()
    run_kwargs = client.containers.run.call_args.kwargs
    assert run_kwargs["entrypoint"] == ["python3", "-m", "nvflare.dashboard.wsgi"]
    assert run_kwargs["volumes"][str(dashboard_cli.os.getcwd())]["mode"] == "rw"
    assert "model" not in run_kwargs["volumes"][str(dashboard_cli.os.getcwd())]
    assert "Dashboard container started" in capsys.readouterr().out


@pytest.mark.parametrize("status", ["exited", "removing"])
def test_start_reports_immediate_container_exit(capsys, status):
    args = _parse_dashboard_args(["--start", "-i", "nvflare-parent:test", "--cred", "admin@example.com:pw:org"])
    container = SimpleNamespace(
        id="container-1",
        status=status,
        reload=Mock(),
        logs=Mock(return_value=b"python: can't open file '/app/nvflare/dashboard/wsgi.py'"),
    )
    client = SimpleNamespace(images=SimpleNamespace(pull=Mock()), containers=SimpleNamespace(run=Mock()))
    client.containers.run.return_value = container

    with (
        patch.object(dashboard_cli.docker, "from_env", return_value=client),
        patch.object(dashboard_cli.time, "sleep"),
        pytest.raises(SystemExit) as exc_info,
    ):
        dashboard_cli.start(args)

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert f"Dashboard container exited immediately with status: {status}" in output
    assert "Container logs:" in output
    assert "can't open file" in output


def test_start_reports_auto_removed_container_exit(capsys):
    """auto_remove=True often removes the container before reload() returns;
    that path raises docker.errors.NotFound and must exit 1 with a clear message."""
    args = _parse_dashboard_args(["--start", "-i", "nvflare-parent:test", "--cred", "admin@example.com:pw:org"])
    container = SimpleNamespace(
        id="container-1",
        status="running",
        reload=Mock(side_effect=dashboard_cli.docker.errors.NotFound("gone")),
        logs=Mock(),
    )
    client = SimpleNamespace(images=SimpleNamespace(pull=Mock()), containers=SimpleNamespace(run=Mock()))
    client.containers.run.return_value = container

    with (
        patch.object(dashboard_cli.docker, "from_env", return_value=client),
        patch.object(dashboard_cli.time, "sleep"),
        pytest.raises(SystemExit) as exc_info,
    ):
        dashboard_cli.start(args)

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert "Dashboard container exited immediately and was removed." in output
    container.logs.assert_not_called()
