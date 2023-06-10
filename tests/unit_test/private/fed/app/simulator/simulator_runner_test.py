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

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner
from nvflare.private.fed.utils.fed_utils import split_gpus


class TestSimulatorRunner:
    def setup_method(self) -> None:
        self.workspace = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.workspace)

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    @patch("nvflare.private.fed.app.utils.FedAdminServer")
    @patch("nvflare.private.fed.client.fed_client.FederatedClient.register")
    # @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.start_heartbeat")
    # @patch("nvflare.private.fed.app.deployer.simulator_deployer.FedAdminAgent")
    def test_valid_job_simulate_setup(self, mock_server, mock_admin, mock_register):
        workspace = tempfile.mkdtemp()
        job_folder = os.path.join(os.path.dirname(__file__), "../../../../data/jobs/valid_job")
        runner = SimulatorRunner(job_folder=job_folder, workspace=workspace, threads=1)
        assert runner.setup()

        expected_clients = ["site-1", "site-2"]
        client_names = []
        for client in runner.client_names:
            client_names.append(client)
        assert sorted(client_names) == sorted(expected_clients)

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    @patch("nvflare.private.fed.app.utils.FedAdminServer")
    @patch("nvflare.private.fed.client.fed_client.FederatedClient.register")
    # @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.start_heartbeat")
    # @patch("nvflare.private.fed.app.deployer.simulator_deployer.FedAdminAgent")
    def test_client_names_setup(self, mock_server, mock_admin, mock_register):
        workspace = tempfile.mkdtemp()
        job_folder = os.path.join(os.path.dirname(__file__), "../../../../data/jobs/valid_job")
        runner = SimulatorRunner(job_folder=job_folder, workspace=workspace, clients="site-1", threads=1)
        assert runner.setup()

        expected_clients = ["site-1"]
        client_names = []
        for client in runner.client_names:
            client_names.append(client)
        assert sorted(client_names) == sorted(expected_clients)

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    @patch("nvflare.private.fed.app.utils.FedAdminServer")
    @patch("nvflare.private.fed.client.fed_client.FederatedClient.register")
    # @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.start_heartbeat")
    # @patch("nvflare.private.fed.app.deployer.simulator_deployer.FedAdminAgent")
    def test_no_app_for_client(self, mock_server, mock_admin, mock_register):
        workspace = tempfile.mkdtemp()
        job_folder = os.path.join(os.path.dirname(__file__), "../../../../data/jobs/valid_job")
        runner = SimulatorRunner(job_folder=job_folder, workspace=workspace, n_clients=3, threads=1)
        assert not runner.setup()

    @pytest.mark.parametrize(
        "client_names, gpus, expected_split_names",
        [
            (["1", "2", "3", "4"], ["0", "1"], [["1", "3"], ["2", "4"]]),
            (["1", "2", "3", "4", "5"], ["0", "1"], [["1", "3", "5"], ["2", "4"]]),
            (["1", "2", "3", "4", "5"], ["0", "1", "2"], [["1", "4"], ["2", "5"], ["3"]]),
        ],
    )
    def test_split_names(self, client_names, gpus, expected_split_names):
        runner = SimulatorRunner(job_folder="", workspace="")
        split_names = runner.split_clients(client_names, gpus)
        assert sorted(split_names) == sorted(expected_split_names)

    @pytest.mark.parametrize(
        "gpus, expected_gpus",
        [
            ("[0,1],[1, 2]", ["0,1", "1,2"]),
            ("[0,1],[3]", ["0,1", "3"]),
            ("[0,1],[ 3 ]", ["0,1", "3"]),
            ("[02,1],[ a ]", ["02,1", "a"]),
            ("[]", [""]),
            ("[0,1],3", ["0,1", "3"]),
            ("[0,1],[1,2,3],3", ["0,1", "1,2,3", "3"]),
            ("0,1,2", ["0", "1", "2"]),
        ],
    )
    def test_split_gpus_success(self, gpus, expected_gpus):
        splitted_gpus = split_gpus(gpus)
        assert splitted_gpus == expected_gpus

    @pytest.mark.parametrize(
        "gpus",
        [
            "[0,1],3]",
            "0,1,[2",
            "[0,1]extra",
            "[1, [2, 3], 4]",
        ],
    )
    def test_split_gpus_fail(self, gpus):
        with pytest.raises(ValueError):
            split_gpus(gpus)
