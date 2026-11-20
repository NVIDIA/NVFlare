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
"""Shared factories for the Flower integration tests (not collected as tests)."""

from unittest.mock import MagicMock, patch

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.flower.applet import FlowerClientApplet, FlowerServerApplet
from nvflare.app_opt.flower.defs import Constant


def make_client_fl_ctx() -> FLContext:
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
    return fl_ctx


def make_server_start_ctx(run_dir, superlink_addr="127.0.0.1:9091") -> dict:
    workspace = MagicMock(spec=Workspace)
    workspace.get_app_custom_dir.return_value = str(run_dir / "custom")
    workspace.get_run_dir.return_value = str(run_dir / "run")
    engine = MagicMock()
    engine.get_workspace.return_value = workspace
    fl_ctx = FLContext()
    fl_ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
    fl_ctx.set_prop(ReservedKey.RUN_NUM, "job", private=True, sticky=False)
    return {Constant.APP_CTX_SUPERLINK_API_ADDR: superlink_addr, Constant.APP_CTX_FL_CONTEXT: fl_ctx}


def build_supernode_command(applet: FlowerClientApplet):
    """Return the CommandDescriptor the client applet would launch."""
    with patch("nvflare.app_opt.flower.applet._validate_flower_executable"):
        return applet.get_command(
            {
                Constant.APP_CTX_SUPERLINK_ADDR: "127.0.0.1:9092",
                Constant.APP_CTX_CLIENTAPP_API_ADDR: "127.0.0.1:9094",
                Constant.APP_CTX_FL_CONTEXT: make_client_fl_ctx(),
            }
        )


def build_superlink_command(applet: FlowerServerApplet, run_dir, superlink_addr="127.0.0.1:9091"):
    """Return the CommandDescriptor the server applet would launch for the SuperLink (nothing is started)."""
    process_mgr = MagicMock()
    process_mgr.poll.return_value = None
    with (
        patch.object(applet, "_start_process", return_value=process_mgr) as start_process,
        patch.object(applet, "_wait_for_superlink_ready"),
        patch.object(applet, "_run_flower_command", return_value={"run-id": "1"}),
        patch("nvflare.app_opt.flower.applet._validate_flower_executable"),
    ):
        applet.start(make_server_start_ctx(run_dir, superlink_addr))
    return start_process.call_args.kwargs["cmd_desc"]
