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
from unittest.mock import patch

import pytest

from nvflare.app_opt.flower.applet import FlowerServerApplet


def _make_server_applet(run_config=None):
    applet = FlowerServerApplet(
        database="",
        superlink_ready_timeout=1.0,
        run_config=run_config,
    )
    applet.exec_api_addr = "127.0.0.1:9093"
    applet.flower_app_dir = "/tmp/flower app"
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
        "--federation-config",
        'address="127.0.0.1:9093"',
        "/tmp/flower app",
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
        "--format",
        "json",
        "--federation-config",
        'address="127.0.0.1:9093"',
        "run-id-123",
        "/tmp/flower app",
    ]
