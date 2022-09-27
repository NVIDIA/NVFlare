# Copyright (c) 2022, NVIDIA CORPORATION.
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
from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner


class TestSimulateFedStatsJob:
    def setup_method(self) -> None:
        self.workspace = tempfile.mkdtemp()
        current_workdir = os.getcwd()
        self.stats_job_folder = "./df_stats_job"

    def teardown_method(self) -> None:
        shutil.rmtree(self.workspace)

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    @patch("nvflare.private.fed.app.server.server_train.FedAdminServer")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.register")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.start_heartbeat")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FedAdminAgent")
    def test_fed_stats_job_simulate_setup(self, mock_server, mock_admin, mock_register, mock_heartbeat, mock_agent):
        runner = SimulatorRunner(job_folder=self.stats_job_folder,
                                 workspace=self.workspace,
                                 clients="site-1, site-2",
                                 threads=2)
        assert runner.setup()

        expected_clients = ["site-1", "site-2"]
        client_names = []
        for client in runner.client_names:
            client_names.append(client.strip())
        assert sorted(client_names) == sorted(expected_clients)

    # @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    # @patch("nvflare.private.fed.app.server.server_train.FedAdminServer")
    # @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.register")
    # @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.start_heartbeat")
    # @patch("nvflare.private.fed.app.deployer.simulator_deployer.FedAdminAgent")
    # def test_fed_stats_job_simulate_setup(self, mock_server, mock_admin, mock_register, mock_heartbeat, mock_agent):
    #     runner = SimulatorRunner(job_folder=self.stats_job_folder,
    #                              workspace=self.workspace,
    #                              clients="site-1",
    #                              threads=1)
    #     # note this test may not work due to 1) configuration failure, 2) data not available 3) dependency not available
    #     # but the code doesn't stop even in failure conditions.
    #     status = runner.run()
    #     assert(status == 0)
