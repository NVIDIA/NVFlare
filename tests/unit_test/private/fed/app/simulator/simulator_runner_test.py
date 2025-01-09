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
import copy
import os
import shutil
import sys
import threading
import time
import uuid
from argparse import Namespace
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest

from nvflare.apis.fl_constant import FLContextKey, MachineStatus, WorkspaceConstants
from nvflare.private.fed.app.simulator.simulator_runner import SimulatorClientRunner, SimulatorRunner
from nvflare.private.fed.utils.fed_utils import split_gpus


class MockCell:
    def get_root_url_for_child(self):
        return "tcp://0:123"

    def get_internal_listener_url(self):
        return "tcp://0:123"


class TestSimulatorRunner:
    def setup_method(self, method):
        self.workspace_name = str(uuid.uuid4())
        self.cwd = os.getcwd()
        os.makedirs(os.path.join(self.cwd, self.workspace_name, WorkspaceConstants.STARTUP_FOLDER_NAME))

    def teardown_method(self, method):
        os.chdir(self.cwd)
        shutil.rmtree(os.path.join(self.cwd, self.workspace_name))

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    @patch("nvflare.private.fed.app.utils.FedAdminServer")
    @patch("nvflare.private.fed.client.fed_client.FederatedClient.register")
    @patch("nvflare.private.fed.server.fed_server.BaseServer.get_cell", return_value=MockCell())
    def test_valid_job_simulate_setup(self, mock_deploy, mock_admin, mock_register, mock_cell):
        job_folder = os.path.join(os.path.dirname(__file__), "../../../../data/jobs/valid_job")
        runner = SimulatorRunner(job_folder=job_folder, workspace=self.workspace_name, threads=1)
        assert runner.setup()

        expected_clients = ["site-1", "site-2"]
        client_names = []
        for client in runner.client_names:
            client_names.append(client)
        assert sorted(client_names) == sorted(expected_clients)

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    @patch("nvflare.private.fed.app.utils.FedAdminServer")
    @patch("nvflare.private.fed.client.fed_client.FederatedClient.register")
    @patch("nvflare.private.fed.server.fed_server.BaseServer.get_cell", return_value=MockCell())
    def test_client_names_setup(self, mock_deploy, mock_admin, mock_register, mock_cell):
        job_folder = os.path.join(os.path.dirname(__file__), "../../../../data/jobs/valid_job")
        runner = SimulatorRunner(job_folder=job_folder, workspace=self.workspace_name, clients="site-1", threads=1)
        assert runner.setup()

        expected_clients = ["site-1"]
        client_names = []
        for client in runner.client_names:
            client_names.append(client)
        assert sorted(client_names) == sorted(expected_clients)

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    @patch("nvflare.private.fed.app.utils.FedAdminServer")
    @patch("nvflare.private.fed.client.fed_client.FederatedClient.register")
    @patch("nvflare.private.fed.server.fed_server.BaseServer.get_cell", return_value=MockCell())
    def test_no_app_for_client(self, mock_deploy, mock_admin, mock_register, mock_cell):
        job_folder = os.path.join(os.path.dirname(__file__), "../../../../data/jobs/valid_job")
        runner = SimulatorRunner(job_folder=job_folder, workspace=self.workspace_name, n_clients=3, threads=1)
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

    @patch("nvflare.private.fed.app.deployer.simulator_deployer.SimulatorServer.deploy")
    @patch("nvflare.private.fed.app.utils.FedAdminServer")
    @patch("nvflare.private.fed.client.fed_client.FederatedClient.register")
    @patch("nvflare.private.fed.server.fed_server.BaseServer.get_cell", return_value=MockCell())
    def test_start_server_app(self, mock_deploy, mock_admin, mock_register, mock_cell):
        with TemporaryDirectory() as workspace:
            job_folder = os.path.join(os.path.dirname(__file__), "../../../../data/jobs/valid_job")
            runner = SimulatorRunner(
                job_folder=job_folder,
                workspace=workspace,
            )
            runner.setup()

            with patch("nvflare.private.fed.simulator.simulator_server.SimulatorServer.run_engine"):
                with patch("nvflare.private.fed.simulator.simulator_server.SimulatorServer.create_job_cell"):
                    server_thread = threading.Thread(target=runner.start_server_app, args=[runner.args])
                    server_thread.start()

                    while runner.server.engine.engine_info.status != MachineStatus.STARTED:
                        time.sleep(1.0)
                        if not server_thread.is_alive():
                            raise RuntimeError("Could not start the Server App.")
                    fl_ctx = runner.server.engine.new_context()
                    workspace_obj = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
                    assert workspace_obj.get_root_dir() == os.path.join(workspace, "server")

                    runner.server.logger = Mock()
                    runner.server.engine.asked_to_stop = True

    def test_get_new_sys_path_with_empty(self):
        args = Namespace(workspace="/tmp")
        args.set = []
        runner = SimulatorClientRunner(None, args, [], None, None, None)
        old_sys_path = copy.deepcopy(sys.path)
        sys.path.insert(0, "")
        sys.path.append("/temp/test")
        new_sys_path = runner._get_new_sys_path()
        assert old_sys_path == new_sys_path
        sys.path = old_sys_path

    def test_get_new_sys_path_with_multiple_empty(self):
        args = Namespace(workspace="/tmp")
        args.set = []
        runner = SimulatorClientRunner(None, args, [], None, None, None)
        old_sys_path = copy.deepcopy(sys.path)
        sys.path.insert(0, "")
        if len(sys.path) > 2:
            sys.path.insert(2, "")
        sys.path.append("/temp/test")
        new_sys_path = runner._get_new_sys_path()
        assert old_sys_path == new_sys_path
        sys.path = old_sys_path

    def test_get_new_sys_path(self):
        args = Namespace(workspace="/tmp")
        args.set = []
        runner = SimulatorClientRunner(None, args, [], None, None, None)
        old_sys_path = copy.deepcopy(sys.path)
        sys.path.append("/temp/test")
        new_sys_path = runner._get_new_sys_path()
        assert old_sys_path == new_sys_path
        sys.path = old_sys_path
