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
from tests.flower_test_utils import make_client_fl_ctx, make_server_start_ctx


def _make_server_applet(run_config=None):
    applet = FlowerServerApplet(
        database="",
        superlink_ready_timeout=1.0,
        run_config=run_config,
    )
    applet.superlink_api_addr = "127.0.0.1:9093"
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
    assert "auth_token" not in repr(cmd_desc.env).lower()
    assert "bearer" not in repr(cmd_desc.env).lower()


@patch("nvflare.app_opt.flower.applet._validate_flower_executable")
def test_supernode_command_uses_proxy_and_explicit_host_and_port(_validate_executable):
    """The SuperNode must connect to the local proxy and bind its ClientApp API to an explicit free port.

    Flower's default ClientApp API port (9094) collides between clients on one host, and the removed
    --grpc-adapter / --clientappio-api-address flags make Flower 1.43 exit at argument parsing.
    """
    applet = FlowerClientApplet(supernode_max_wait_time=120)

    cmd_desc = applet.get_command(
        {
            FlowerConstant.APP_CTX_SUPERLINK_ADDR: "127.0.0.1:9092",
            FlowerConstant.APP_CTX_CLIENTAPP_API_ADDR: "127.0.0.1:9094",
            FlowerConstant.APP_CTX_FL_CONTEXT: make_client_fl_ctx(),
        }
    )

    args = shlex.split(cmd_desc.cmd)
    assert os.path.basename(args[0]) == "flower-supernode"
    assert "--insecure" in args
    assert args[args.index("--superlink") + 1] == "127.0.0.1:9092"
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--port") + 1] == "9094"
    assert args[args.index("--max-wait-time") + 1] == "120"
    assert "--grpc-adapter" not in args
    assert "--clientappio-api-address" not in args
    assert "--allow-runtime-dependency-installation" not in args


@patch("nvflare.app_opt.flower.applet._validate_flower_executable")
def test_supernode_command_options_are_optional(_validate_executable):
    applet = FlowerClientApplet(allow_runtime_dependency_installation=True, supernode_max_wait_time=None)

    cmd_desc = applet.get_command(
        {
            FlowerConstant.APP_CTX_SUPERLINK_ADDR: "127.0.0.1:9092",
            FlowerConstant.APP_CTX_CLIENTAPP_API_ADDR: "127.0.0.1:9094",
            FlowerConstant.APP_CTX_FL_CONTEXT: make_client_fl_ctx(),
        }
    )

    args = shlex.split(cmd_desc.cmd)
    assert "--max-wait-time" not in args
    assert "--allow-runtime-dependency-installation" in args


@patch("nvflare.app_opt.flower.applet._validate_flower_executable")
def test_supernode_command_quotes_executable_path_with_spaces(_validate_executable):
    applet = FlowerClientApplet()

    with patch.object(sys, "executable", "/opt/my envs/py env/bin/python"):
        cmd_desc = applet.get_command(
            {
                FlowerConstant.APP_CTX_SUPERLINK_ADDR: "127.0.0.1:9092",
                FlowerConstant.APP_CTX_CLIENTAPP_API_ADDR: "127.0.0.1:9094",
                FlowerConstant.APP_CTX_FL_CONTEXT: make_client_fl_ctx(),
            }
        )

    args = shlex.split(cmd_desc.cmd)
    assert args[0] == "/opt/my envs/py env/bin/flower-supernode"
    assert args[1] == "--insecure"


@pytest.mark.parametrize("database", ["", "/tmp/flwr state.db"])
@patch("nvflare.app_opt.flower.applet.run_command", return_value='{"success": true, "run-id": "42"}')
@patch("nvflare.app_opt.flower.applet._validate_flower_executable")
def test_superlink_command_uses_single_loopback_address(_validate_executable, _run_command, database, tmp_path):
    """The SuperLink serves Fleet, ServerApp and CLI control calls on one port, passed explicitly on loopback."""
    applet = FlowerServerApplet(database=database, superlink_ready_timeout=1.0)
    process_mgr = MagicMock()
    process_mgr.poll.return_value = None

    with patch.object(applet, "_start_process", return_value=process_mgr) as start_process:
        with patch.object(applet, "_wait_for_superlink_ready") as wait_ready:
            applet.start(make_server_start_ctx(tmp_path))

    cmd_desc = start_process.call_args.kwargs["cmd_desc"]
    args = shlex.split(cmd_desc.cmd)
    assert os.path.basename(args[0]) == "flower-superlink"
    assert "--insecure" in args
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--port") + 1] == "9091"
    if database:
        assert args[args.index("--database") + 1] == database
    else:
        assert "--database" not in args
    for removed in (
        "--fleet-api-type",
        "--fleet-api-address",
        "--serverappio-api-address",
        "--control-api-address",
    ):
        assert removed not in args
    # Flower 1.43's SuperLink installs the app's dependencies by default; it must be disabled explicitly
    assert "--disable-runtime-dependency-installation" in args
    assert "--allow-runtime-dependency-installation" not in args
    wait_ready.assert_called_once_with("127.0.0.1:9091")
    assert applet.run_id == "42"
    # "flwr run" reaches the SuperLink through the same address
    assert 'address = "127.0.0.1:9091"' in (tmp_path / "run" / "flwr_home" / "config.toml").read_text()


class _FakeResponse:
    def __init__(self, status):
        self.status = status


class _FakeHealthConnection:
    """Replacement for http.client.HTTPConnection that answers GET /health from a scripted list."""

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.requests = []

    def __call__(self, host, port, timeout=None):
        self.target = (host, port)
        return self

    def request(self, method, path):
        self.requests.append((method, path))

    def getresponse(self):
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(outcome)

    def close(self):
        pass


def _ready_applet(timeout=5.0, running=True):
    applet = FlowerServerApplet(database="", superlink_ready_timeout=timeout)
    process_mgr = MagicMock()
    process_mgr.poll.return_value = None if running else 3
    applet._superlink_process_mgr = process_mgr
    return applet


def test_wait_for_superlink_ready_polls_health_until_200():
    applet = _ready_applet()
    fake = _FakeHealthConnection([ConnectionRefusedError("not yet"), 503, 200])

    with (
        patch("nvflare.app_opt.flower.applet.http.client.HTTPConnection", fake),
        patch("nvflare.app_opt.flower.applet.time.sleep"),
    ):
        applet._wait_for_superlink_ready("127.0.0.1:9091")

    assert fake.target == ("127.0.0.1", 9091)
    assert fake.requests == [("GET", "/health")] * 3


def test_wait_for_superlink_ready_fails_fast_when_process_exits():
    applet = _ready_applet(running=False)
    fake = _FakeHealthConnection([200])

    with patch("nvflare.app_opt.flower.applet.http.client.HTTPConnection", fake):
        with pytest.raises(RuntimeError, match="exited with code 3"):
            applet._wait_for_superlink_ready("127.0.0.1:9091")

    assert fake.requests == []


def test_wait_for_superlink_ready_times_out_with_last_error():
    applet = _ready_applet(timeout=0.3)
    fake = _FakeHealthConnection([503])

    with pytest.raises(RuntimeError, match="not ready after 0.3 seconds: HTTP 503"):
        with patch("nvflare.app_opt.flower.applet.http.client.HTTPConnection", fake):
            applet._wait_for_superlink_ready("127.0.0.1:9091")


@patch("nvflare.app_opt.flower.applet.run_command", return_value='{"success": true, "run-id": "42"}')
@patch("nvflare.app_opt.flower.applet._validate_flower_executable")
def test_superlink_runtime_dependency_installation_can_be_allowed(_validate_executable, _run_command, tmp_path):
    applet = FlowerServerApplet(database="", superlink_ready_timeout=1.0, allow_runtime_dependency_installation=True)
    process_mgr = MagicMock()
    process_mgr.poll.return_value = None

    with patch.object(applet, "_start_process", return_value=process_mgr) as start_process:
        with patch.object(applet, "_wait_for_superlink_ready"):
            applet.start(make_server_start_ctx(tmp_path))

    args = shlex.split(start_process.call_args.kwargs["cmd_desc"].cmd)
    assert "--allow-runtime-dependency-installation" in args
    assert "--disable-runtime-dependency-installation" not in args
