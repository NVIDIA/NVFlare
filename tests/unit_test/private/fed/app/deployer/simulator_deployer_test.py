import argparse
import os
import tempfile
import unittest
from unittest.mock import patch

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.fuel.hci.server.authz import AuthorizationService
from nvflare.fuel.sec.audit import AuditService
from nvflare.private.fed.app.deployer.simulator_deployer import SimulatorDeploy
from nvflare.private.fed.client.fed_client import FederatedClient
from nvflare.private.fed.simulator.simulator_server import SimulatorServer
from nvflare.security.security import EmptyAuthorizer


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.deployer = SimulatorDeploy()
        AuthorizationService.initialize(EmptyAuthorizer())
        AuditService.initialize(audit_file_name=WorkspaceConstants.AUDIT_LOG)

    def tearDown(self) -> None:
        self.deployer.close()

    def _create_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("job_folder")
        parser.add_argument("--workspace", "-o", type=str, help="WORKSPACE folder", required=True)
        parser.add_argument("--clients", "-n", type=int, help="number of clients", required=True)
        parser.add_argument("--threads", "-t", type=int, help="number of running threads", required=True)

        parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")

        return parser

    def test_create_server(self):
        with patch("nvflare.private.fed.app.server.server_train.FedAdminServer") as mock_admin:
            workspace = tempfile.mkdtemp()
            os.makedirs(os.path.join(workspace, 'transfer'))
            parser = self._create_parser()
            args = parser.parse_args(['job_folder', '-o' + workspace, '-n 2', '-t 1'])
            _, server = self.deployer.create_fl_server(args)
            assert isinstance(server, SimulatorServer)

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.register")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FederatedClient.start_heartbeat")
    @patch("nvflare.private.fed.app.deployer.simulator_deployer.FedAdminAgent")
    def test_create_client(self, mock_register, mock_heartbeat, mock_agent):
        workspace = tempfile.mkdtemp()
        os.makedirs(os.path.join(workspace, 'transfer'))
        parser = self._create_parser()
        args = parser.parse_args(['job_folder', '-o' + workspace, '-n 2', '-t 1'])
        client = self.deployer.create_fl_client("client0", args)
        assert isinstance(client, FederatedClient)
