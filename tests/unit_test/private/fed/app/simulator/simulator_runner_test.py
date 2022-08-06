import argparse
import tempfile
import unittest
import os
import shutil
from unittest.mock import patch


from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.workspace = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.workspace, 'transfer'))

    def tearDown(self) -> None:
        super().tearDown()
        shutil.rmtree(self.workspace)

    def _create_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("job_folder")
        parser.add_argument("--workspace", "-o", type=str, help="WORKSPACE folder", required=True)
        parser.add_argument("--clients", "-n", type=int, help="number of clients")
        parser.add_argument("--client_list", "-c", type=str, help="client names list")
        parser.add_argument("--threads", "-t", type=int, help="number of running threads", required=True)
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

        expected_clients = ["sit-1", "site-2"]
        client_names = []
        for client in runner.federated_clients:
            client_names.append(client.client_name)
        assert client_names.sort() == expected_clients.sort()

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

        expected_clients = ["sit-1"]
        client_names = []
        for client in runner.federated_clients:
            client_names.append(client.client_name)
        assert client_names.sort() == expected_clients.sort()

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    @patch("nvflare.private.fed.app.server.server_train.FedAdminServer")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.register")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.start_heartbeat")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FedAdminAgent")
    def test_no_app_for_client(self, mock_server,  mock_admin, mock_register, mock_heartbeat, mock_agent):
        workspace = tempfile.mkdtemp()
        parser = self._create_parser()
        job_folder = os.path.join(os.path.dirname(__file__), "../../../../data/jobs/valid_job")
        args = parser.parse_args([job_folder, '-o' + workspace, '-n 2', '-t 1'])
        runner = SimulatorRunner(args)
        assert not runner.setup()
