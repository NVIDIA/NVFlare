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

import os
import shlex
import sys
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_common.metrics_exchange.metrics_sender import ANALYTICS_BOOTSTRAP_ENV
from nvflare.app_opt.flower.applet import FlowerClientApplet, FlowerServerApplet
from nvflare.app_opt.flower.defs import Constant as FlowerConstant


def _make_server_applet(run_config=None):
    applet = FlowerServerApplet(
        database="",
        superlink_ready_timeout=1.0,
        run_config=run_config,
    )
    applet.exec_api_addr = "127.0.0.1:9093"
    applet.flower_app_dir = "/tmp/flower app"
    applet.flwr_home_dir = "/tmp/flwr-home"
    return applet


@patch("nvflare.app_opt.flower.applet._validate_flower_executable")
def test_flower_run_command_formats_toml_scalars(_validate_executable):
    applet = _make_server_applet(
        {
            "learning-rate": 0.01,
            "use-gpu": True,
            "optimizer": "adam",
            "experiment-name": "hello world",
        }
    )

    cmd = applet._flower_command("run")

    assert shlex.split(cmd) == [
        os.path.join(os.path.dirname(sys.executable), "flwr"),
        "run",
        ".",
        "nvflare",
        "--run-config",
        "learning-rate=0.01",
        "--run-config",
        "use-gpu=true",
        "--run-config",
        'optimizer="adam"',
        "--run-config",
        'experiment-name="hello world"',
        "--format",
        "json",
    ]


@patch("nvflare.app_opt.flower.applet._validate_flower_executable")
def test_flower_run_command_rejects_unsupported_run_config_values(_validate_executable):
    applet = _make_server_applet({"layers": [32, 64]})

    with pytest.raises(TypeError, match=r"values must be bool, int, float, or str"):
        applet._flower_command("run")


@patch("nvflare.app_opt.flower.applet._validate_flower_executable")
def test_flower_stop_command_does_not_include_run_config(_validate_executable):
    applet = _make_server_applet({"use-gpu": True})

    cmd = applet._flower_command("stop", "run-id-123")

    assert shlex.split(cmd) == [
        os.path.join(os.path.dirname(sys.executable), "flwr"),
        "stop",
        "run-id-123",
        "nvflare",
        "--format",
        "json",
    ]


def test_flower_predeployed_constant_value():
    """FLOWER_PREDEPLOYED constant has correct value."""
    assert FlowerConstant.FLOWER_PREDEPLOYED == "flower_predeployed"


@patch("nvflare.app_opt.flower.applet._validate_flower_executable")
def test_flower_list_command_uses_superlink_name(_validate_executable):
    applet = _make_server_applet()

    cmd = applet._flower_command("list")

    assert shlex.split(cmd) == [
        os.path.join(os.path.dirname(sys.executable), "flwr"),
        "list",
        "nvflare",
        "--format",
        "json",
    ]


def test_prepare_flwr_home_writes_config(tmp_path):
    applet = _make_server_applet()

    flwr_home_dir = applet._prepare_flwr_home(str(tmp_path))
    config_path = tmp_path / "flwr_home" / "config.toml"

    assert flwr_home_dir == str(tmp_path / "flwr_home")
    assert config_path.read_text() == (
        "[superlink]\n"
        'default = "nvflare"\n'
        "\n"
        "[superlink.nvflare]\n"
        'address = "127.0.0.1:9093"\n'
        "insecure = true\n"
    )


@patch("nvflare.app_opt.flower.applet.run_command", return_value='{"success": true}')
def test_run_flower_command_sets_flwr_home_and_cwd(mock_run_command):
    applet = _make_server_applet()

    applet._run_flower_command("flwr run --format json . nvflare", cwd=applet.flower_app_dir)

    cmd_desc = mock_run_command.call_args.args[0]
    assert cmd_desc.cwd == applet.flower_app_dir
    assert cmd_desc.env["FLWR_HOME"] == applet.flwr_home_dir


@patch("nvflare.app_opt.flower.applet._validate_flower_executable")
def test_client_applet_passes_exact_metrics_bootstrap_without_credentials(_validate_executable):
    workspace = MagicMock(spec=Workspace)
    workspace.get_app_dir.return_value = "/workspace/run_job/app_site-1"
    workspace.get_app_config_dir.return_value = "/workspace/run_job/app_site-1/config"
    engine = MagicMock()
    engine.get_workspace.return_value = workspace
    engine.get_clients.return_value = [MagicMock(name="site-1")]
    engine.get_clients.return_value[0].name = "site-1"
    fl_ctx = FLContext()
    fl_ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
    fl_ctx.set_prop(ReservedKey.RUN_NUM, "job", private=True, sticky=False)
    fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-1", private=True, sticky=False)
    applet = FlowerClientApplet(
        extra_env={"MY_VAR": "value"},
    )

    cmd_desc = applet.get_command(
        {
            FlowerConstant.APP_CTX_SUPERLINK_ADDR: "127.0.0.1:9092",
            FlowerConstant.APP_CTX_CLIENTAPP_API_ADDR: "127.0.0.1:9094",
            FlowerConstant.APP_CTX_FL_CONTEXT: fl_ctx,
        }
    )

    assert cmd_desc.env[ANALYTICS_BOOTSTRAP_ENV] == "/workspace/run_job/app_site-1/config/analytics_bootstrap.json"
    assert cmd_desc.env["MY_VAR"] == "value"
    argv = shlex.split(cmd_desc.cmd)
    assert "--grpc-adapter" not in argv
    assert "--clientappio-api-address" not in argv
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "9094"
    assert "--allow-runtime-dependency-installation" not in argv
    assert "auth_token" not in repr(cmd_desc.env).lower()
    assert "bearer" not in repr(cmd_desc.env).lower()


@patch("nvflare.app_opt.flower.applet._validate_flower_executable")
def test_list_targets_exact_run_with_option(_validate_executable):
    command = shlex.split(_make_server_applet()._flower_command("list", "12345678901234567890"))
    assert command[1:] == ["list", "--run-id", "12345678901234567890", "nvflare", "--format", "json"]


@pytest.mark.parametrize("applet_type", [FlowerServerApplet, FlowerClientApplet])
@patch("nvflare.app_opt.flower.applet._validate_flwr_runtime", side_effect=RuntimeError("unsupported Flower"))
def test_deployed_runtime_rejects_version_before_start(_validate, applet_type):
    applet = _make_server_applet() if applet_type is FlowerServerApplet else FlowerClientApplet()
    with pytest.raises(RuntimeError, match="unsupported Flower"):
        applet.start({})


@pytest.mark.parametrize("runtime_deps", [False, True])
@patch("nvflare.app_opt.flower.applet._validate_flower_executable")
@patch("nvflare.app_opt.flower.applet._validate_flwr_runtime")
@patch("nvflare.app_opt.flower.applet._wait_for_superlink")
@patch("nvflare.app_opt.flower.applet.start_process")
@patch("nvflare.app_opt.flower.applet.run_command", return_value='{"success": true, "run-id": "123"}')
def test_server_http_launcher_and_dependency_policy(_run, start, ready, _version, _executable, runtime_deps, tmp_path):
    workspace = MagicMock(spec=Workspace)
    workspace.get_run_dir.return_value = str(tmp_path)
    workspace.get_app_custom_dir.return_value = str(tmp_path / "custom")
    engine = MagicMock()
    engine.get_workspace.return_value = workspace
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
    ctx.set_prop(ReservedKey.RUN_NUM, "job", private=True, sticky=False)
    applet = FlowerServerApplet("", 10, allow_runtime_dependency_installation=runtime_deps)
    applet.start({FlowerConstant.APP_CTX_SUPERLINK_ADDR: "127.0.0.1:19092", FlowerConstant.APP_CTX_FL_CONTEXT: ctx})
    descriptor = start.call_args.args[0]
    argv = shlex.split(descriptor.cmd)
    assert argv[1:6] == ["--insecure", "--host", "127.0.0.1", "--port", "19092"]
    expected_flag = (
        "--allow-runtime-dependency-installation" if runtime_deps else "--disable-runtime-dependency-installation"
    )
    assert argv[6:] == [expected_flag]
    assert descriptor.env["FLWR_HOME"] == str(tmp_path / "flwr_home")
    ready.assert_called_once_with("127.0.0.1:19092", start.return_value, 10)
    assert applet.run_id == "123"


@pytest.mark.parametrize(
    "status,expected",
    [
        ("running", (False, 0)),
        ("finished:completed", (True, 0)),
        ("finished:failed", (True, 1)),
        ("finished:stopped", (True, 1)),
    ],
)
def test_flower_completion_status(status, expected):
    from nvflare.app_common.tie.defs import Constant

    applet = _make_server_applet()
    applet.run_id = "123"
    applet._flower_command = MagicMock(return_value="list command")
    applet._run_flower_command = MagicMock(return_value={"runs": [{"status": status}]})
    stopped, code = applet._query_for_run_status()
    assert stopped == expected[0]
    assert code == (Constant.EXIT_CODE_FAILED if expected[1] else 0)
    applet._flower_command.assert_called_once_with("list", "123")


@patch("nvflare.app_opt.flower.applet.run_command", return_value="ServerApp training summary\n")
def test_server_log_is_saved_before_superlink_stops(run_command, tmp_path):
    applet = _make_server_applet()
    applet.run_id = "123"
    applet.run_dir = str(tmp_path)
    applet.flower_run_finished = True
    manager = MagicMock()
    manager.poll.return_value = None
    applet._superlink_process_mgr = manager

    def stop():
        assert (tmp_path / "server_app_log.txt").read_text() == "ServerApp training summary\n"

    manager.stop.side_effect = stop
    applet.stop()
    assert shlex.split(run_command.call_args.args[0].cmd)[1:] == ["log", "123", "nvflare", "--show"]
    manager.stop.assert_called_once()
