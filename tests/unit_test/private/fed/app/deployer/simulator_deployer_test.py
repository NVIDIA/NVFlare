# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pytest

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.fuel.f3.cellnet.defs import CellChannel
from nvflare.fuel.f3.cellnet.identity import ADMIN_LISTENER_KEY
from nvflare.fuel.hci.server.authz import AuthorizationService
from nvflare.fuel.sec.audit import AuditService
from nvflare.private.fed.app.deployer.simulator_deployer import SimulatorDeployer
from nvflare.private.fed.app.simulator.simulator import define_simulator_parser
from nvflare.private.fed.client.fed_client import FederatedClient
from nvflare.private.fed.server.run_manager import RunManager
from nvflare.private.fed.simulator.simulator_server import SimulatorServer

# from nvflare.private.fed.simulator.simulator_server import SimulatorServer
from nvflare.security.security import EmptyAuthorizer


@pytest.mark.xdist_group(name="simulator_deploy")
class TestSimulatorDeploy(unittest.TestCase):
    def setUp(self) -> None:
        self.deployer = SimulatorDeployer()
        AuthorizationService.initialize(EmptyAuthorizer())
        AuditService.initialize(audit_file_name=WorkspaceConstants.AUDIT_LOG)

    def tearDown(self) -> None:
        self.deployer.close()

    def _create_parser(self):
        parser = argparse.ArgumentParser()
        define_simulator_parser(parser)

        parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")

        return parser

    @patch("nvflare.private.fed.client.fed_client.FederatedClient.register")
    # @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.start_heartbeat")
    # @patch("nvflare.private.fed.app.deployer.simulator_deployer.FedAdminAgent")
    def test_create_client(self, mock_register):
        workspace = tempfile.mkdtemp()
        parser = self._create_parser()
        args = parser.parse_args(["job_folder", "-w" + workspace, "-n 2", "-t 1"])
        client, _, _, _ = self.deployer.create_fl_client("client0", args)
        assert isinstance(client, FederatedClient)
        client.cell.stop()
        shutil.rmtree(workspace)

    @patch("nvflare.fuel.hci.server.hci.FileStreamer.register_stream_processing")
    @patch("nvflare.private.fed.simulator.simulator_server.SimulatorServer._register_cellnet_cbs")
    @patch("nvflare.private.fed.server.fed_server.Cell")
    def test_create_server(self, mock_cell, mock_simulator_server, mock_register_stream):
        workspace = tempfile.mkdtemp()
        os.mkdir(os.path.join(workspace, "local"))
        os.mkdir(os.path.join(workspace, "startup"))
        parser = self._create_parser()
        args = parser.parse_args(["job_folder", "-w" + workspace, "-n 2", "-t 1"])
        args.config_folder = "config"
        server_config, server = self.deployer.create_fl_server(args)

        assert isinstance(server, SimulatorServer)
        assert isinstance(server.engine.run_manager, RunManager)
        assert len(self.deployer.open_ports) == 1
        assert "admin_host" not in server_config
        assert "admin_port" not in server_config
        root_urls = mock_cell.call_args.kwargs["root_url"]
        assert root_urls == [f"tcp://0:{self.deployer.open_ports[0]}"]
        assert all(ADMIN_LISTENER_KEY not in url for url in root_urls)
        assert server.admin_server.enable_hci is False
        assert server.admin_server.cmd_reg is None
        assert server.admin_server.sess_mgr is None
        assert server.admin_server.net_agent is None
        assert server.admin_server.net_mgr is None
        registered_channels = {call.kwargs["channel"] for call in server.cell.register_request_cb.call_args_list}
        assert CellChannel.HCI not in registered_channels
        assert "_net_manager" not in registered_channels
        mock_register_stream.assert_not_called()
        server.cell.core_cell.register_request_cb.assert_not_called()

        server.cell.stop()
        server.close()
        shutil.rmtree(workspace)
