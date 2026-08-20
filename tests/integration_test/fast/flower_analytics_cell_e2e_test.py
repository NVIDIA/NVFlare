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
import os
import subprocess
import sys
import uuid
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.metrics_exchange.metrics_sender import ANALYTICS_BOOTSTRAP_FILE
from nvflare.app_common.widgets.metric_relay import MetricRelay
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.utils.network_utils import get_open_ports

_CLIENT_SCRIPT = r"""
import json
import sys
from pathlib import Path

from nvflare.client import tracking
from nvflare.client.tracking import SummaryWriter

client = tracking.init(config_file=sys.argv[1])
try:
    writer = SummaryWriter(ctx=client)
    writer.add_scalar("loss", 0.125, global_step=7)
    Path(sys.argv[2]).write_text(json.dumps({"sent": True}))
finally:
    tracking.shutdown(client)
"""


class _Engine:
    def __init__(self, cell, config_dir):
        self.cell = cell
        self.workspace = MagicMock()
        self.workspace.get_app_config_dir.return_value = str(config_dir)

    def get_cell(self):
        return self.cell

    def get_workspace(self):
        return self.workspace

    def new_context(self):
        return nullcontext(MagicMock(spec=FLContext))


@pytest.mark.parametrize("internal_transport", ["tcp", "shared-file"])
def test_analytics_record_crosses_a_real_direct_cell_session(tmp_path, monkeypatch, internal_transport):
    if internal_transport == "shared-file":
        internal_root = tmp_path / "cellnet"
        internal_root.mkdir(mode=0o770)
        internal_resources = {
            "root_dir": str(internal_root),
            DriverParams.CONNECTION_SECURITY.value: "clear",
        }
    else:
        internal_root = None
        internal_resources = {
            "host": "127.0.0.1",
            DriverParams.CONNECTION_SECURITY.value: "clear",
        }
    monkeypatch.setattr(CommConfigurator, "_config_loaded", True)
    monkeypatch.setattr(
        CommConfigurator,
        "_configuration",
        {
            "internal": {
                "scheme": internal_transport,
                "resources": internal_resources,
            }
        },
    )
    server_url = f"tcp://127.0.0.1:{get_open_ports(1)[0]}"
    root_fqcn = f"site-{uuid.uuid4().hex[:8]}"
    server = Cell(
        fqcn=root_fqcn,
        root_url=server_url,
        secure=False,
        credentials={},
        create_internal_listener=False,
    )
    cj = None
    relay = None
    fl_ctx = None
    try:
        server.start()
        cj = Cell(
            fqcn=f"{root_fqcn}.job",
            root_url=server_url,
            secure=False,
            credentials={},
            create_internal_listener=False,
        )
        cj.start()
        engine = _Engine(cj, tmp_path)
        fl_ctx = MagicMock(spec=FLContext)
        fl_ctx.get_engine.return_value = engine
        fl_ctx.get_job_id.return_value = "job"
        fl_ctx.get_identity_name.return_value = "site-1"
        fl_ctx.get_prop.return_value = None
        fl_ctx.get_peer_context.return_value = None
        relay = MetricRelay()
        relay._start(fl_ctx)
        result_path = tmp_path / "client_result.json"
        env = os.environ.copy()
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        with patch("nvflare.app_common.widgets.metric_relay.send_analytic_dxo") as send:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _CLIENT_SCRIPT,
                    str(tmp_path / ANALYTICS_BOOTSTRAP_FILE),
                    str(result_path),
                ],
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
            )

        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        client_result = json.loads(result_path.read_text())
        assert client_result["sent"] is True
        assert send.call_count == 1
        dxo = send.call_args.args[1]
        assert dxo.data["track_key"] == "loss"
        assert dxo.data["track_value"] == 0.125
        assert dxo.data["global_step"] == 7
    finally:
        if relay and fl_ctx:
            relay._stop(fl_ctx)
        if cj:
            cj.stop()
        server.stop()
    if internal_root:
        assert list(internal_root.iterdir()) == []
