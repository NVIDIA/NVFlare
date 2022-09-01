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
from unittest.mock import patch
from argparse import Namespace

import pytest

from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner


class TestSimulatorRunner:
    def setup_method(self) -> None:
        self.workspace = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.workspace)

    def _create_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("job_folder")
        parser.add_argument("--workspace", "-o", type=str, help="WORKSPACE folder", required=True)
        parser.add_argument("--clients", "-n", type=int, help="number of clients")
        parser.add_argument("--client_list", "-c", type=str, help="client names list")
        parser.add_argument("--threads", "-t", type=int, help="number of running threads", required=True)
        parser.add_argument("--gpu", "-gpu", type=str, help="list of GPUs")
        parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")

        return parser

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    @patch("nvflare.private.fed.app.server.server_train.FedAdminServer")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.register")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.start_heartbeat")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FedAdminAgent")
    def test_valid_job_simulate_setup(self, mock_server,  mock_admin, mock_register, mock_heartbeat, mock_agent):
        workspace = tempfile.mkdtemp()
        parser = self._create_parser()
        job_folder = os.path.join(os.path.dirname(__file__), "../../../../data/jobs/valid_job")
        args = parser.parse_args([job_folder, '-o' + workspace, '-t 1'])
        runner = SimulatorRunner(args)
        assert runner.setup()

        expected_clients = ["site-1", "site-2"]
        client_names = []
        for client in runner.client_names:
            client_names.append(client)
        assert sorted(client_names) == sorted(expected_clients)

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    @patch("nvflare.private.fed.app.server.server_train.FedAdminServer")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.register")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.start_heartbeat")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FedAdminAgent")
    def test_client_names_setup(self, mock_server,  mock_admin, mock_register, mock_heartbeat, mock_agent):
        workspace = tempfile.mkdtemp()
        parser = self._create_parser()
        job_folder = os.path.join(os.path.dirname(__file__), "../../../../data/jobs/valid_job")
        args = parser.parse_args([job_folder, '-o' + workspace, '-c site-1', '-t 1'])
        runner = SimulatorRunner(args)
        assert runner.setup()

        expected_clients = ["site-1"]
        client_names = []
        for client in runner.client_names:
            client_names.append(client)
        assert sorted(client_names) == sorted(expected_clients)

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    @patch("nvflare.private.fed.app.server.server_train.FedAdminServer")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.register")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.start_heartbeat")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FedAdminAgent")
    def test_no_app_for_client(self, mock_server,  mock_admin, mock_register, mock_heartbeat, mock_agent):
        workspace = tempfile.mkdtemp()
        parser = self._create_parser()
        job_folder = os.path.join(os.path.dirname(__file__), "../../../../data/jobs/valid_job")
        args = parser.parse_args([job_folder, '-o' + workspace, '-n 3', '-t 1'])
        runner = SimulatorRunner(args)
        assert not runner.setup()

    @pytest.mark.parametrize("client_names, gpus, expected_split_names", [
                                (["1", "2", "3", "4"], ["0", "1"], [["1", "3"], ["2", "4"]]),
                                (["1", "2", "3", "4", "5"], ["0", "1"], [["1", "3", "5"], ["2", "4"]]),
                                (["1", "2", "3", "4", "5"], ["0", "1", "2"], [["1", "4"], ["2", "5"], ["3"]])
                                ]
                             )
    def test_split_names(self, client_names, gpus, expected_split_names):
        runner = SimulatorRunner(Namespace())
        split_names = runner.split_names(client_names, gpus)
        assert sorted(split_names) == sorted(expected_split_names)