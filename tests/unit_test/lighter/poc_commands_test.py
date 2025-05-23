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
import collections

import pytest

from nvflare.cli_exception import CLIException
from nvflare.lighter.utils import update_project_server_name_config
from nvflare.tool.poc.poc_commands import (
    client_gpu_assignments,
    get_gpu_ids,
    get_service_command,
    get_service_config,
    prepare_builders,
    update_clients,
)
from nvflare.tool.poc.service_constants import FlareServiceConstants as SC


class TestPOCCommands:
    def test_client_gpu_assignments(self):
        clients = [f"site-{i}" for i in range(0, 12)]
        gpu_ids = [0, 1, 2, 0, 3]
        assignments = client_gpu_assignments(clients, gpu_ids)
        assert assignments == {
            "site-0": [0],
            "site-1": [1],
            "site-2": [2],
            "site-3": [0],
            "site-4": [3],
            "site-5": [0],
            "site-6": [1],
            "site-7": [2],
            "site-8": [0],
            "site-9": [3],
            "site-10": [0],
            "site-11": [1],
        }

    clients = [f"site-{i}" for i in range(0, 4)]
    gpu_ids = []
    assignments = client_gpu_assignments(clients, gpu_ids)
    assert assignments == {"site-0": [], "site-1": [], "site-2": [], "site-3": []}
    clients = [f"site-{i}" for i in range(0, 4)]
    gpu_ids = [0, 1, 2, 0, 3]
    assignments = client_gpu_assignments(clients, gpu_ids)
    assert assignments == {"site-0": [0, 3], "site-1": [1], "site-2": [2], "site-3": [0]}

    def test_get_gpu_ids(self):
        host_gpu_ids = [0]
        gpu_ids = get_gpu_ids(-1, host_gpu_ids)
        assert gpu_ids == [0]
        gpu_ids = get_gpu_ids([0], host_gpu_ids)
        assert gpu_ids == [0]
        with pytest.raises(CLIException) as e:
            # gpu id =1 is not valid GPU ID as the host only has 1 gpu where id = 0
            gpu_ids = get_gpu_ids([0, 1], host_gpu_ids)

    def test_get_package_command(self):
        cmd = get_service_command(SC.CMD_START, "/tmp/nvflare/poc", SC.FLARE_SERVER, {})
        assert "/tmp/nvflare/poc/server/startup/start.sh" == cmd

        cmd = get_service_command(SC.CMD_START, "/tmp/nvflare/poc", SC.FLARE_PROJ_ADMIN, {})
        assert "/tmp/nvflare/poc/admin@nvidia.com/startup/fl_admin.sh" == cmd

        cmd = get_service_command(SC.CMD_START, "/tmp/nvflare/poc", "site-2000", {})
        assert "/tmp/nvflare/poc/site-2000/startup/start.sh" == cmd

    def test_get_package_command2(self):
        project_config = {
            "api_version": 3,
            "name": "example_project",
            "description": "NVIDIA FLARE sample project yaml file",
            "participants": [
                {"name": "server", "type": "server", "org": "nvidia", "fed_learn_port": 8002, "admin_port": 8003},
                {"name": "admin@nvidia.com", "type": "admin", "org": "nvidia", "role": "project_admin"},
                {"name": "lead@nvidia.com", "type": "admin", "org": "nvidia", "role": "lead"},
                {"name": "site-1", "type": "client", "org": "nvidia"},
                {"name": "site-2000", "type": "client", "org": "nvidia"},
            ],
            "builders": [
                {
                    "path": "nvflare.lighter.impl.static_file.StaticFileBuilder",
                    "args": {"config_folder": "config", "docker_image": "nvflare/nvflare"},
                },
            ],
        }

        project_config = collections.OrderedDict(project_config)
        global_packages = get_service_config(project_config)
        assert global_packages[SC.IS_DOCKER_RUN] is True

        cmd = get_service_command(SC.CMD_START, "/tmp/nvflare/poc", SC.FLARE_SERVER, global_packages)
        assert "/tmp/nvflare/poc/server/startup/docker.sh -d" == cmd

        cmd = get_service_command(SC.CMD_START, "/tmp/nvflare/poc", SC.FLARE_PROJ_ADMIN, global_packages)
        assert "/tmp/nvflare/poc/admin@nvidia.com/startup/fl_admin.sh" == cmd

        cmd = get_service_command(SC.CMD_START, "/tmp/nvflare/poc", "site-2000", global_packages)
        assert "/tmp/nvflare/poc/site-2000/startup/docker.sh -d" == cmd

        cmd = get_service_command(SC.CMD_STOP, "/tmp/nvflare/poc", SC.FLARE_SERVER, global_packages)
        assert "docker stop server" == cmd

        cmd = get_service_command(SC.CMD_STOP, "/tmp/nvflare/poc", SC.FLARE_PROJ_ADMIN, global_packages)
        assert "touch /tmp/nvflare/poc/admin@nvidia.com/shutdown.fl" == cmd

        cmd = get_service_command(SC.CMD_STOP, "/tmp/nvflare/poc", "site-2000", global_packages)
        assert "docker stop site-2000" == cmd

    def test_update_server_name(self):

        project_config = {
            "participants": [
                {"name": "server1", "org": "nvidia", "type": "server"},
                {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin", "type": "admin"},
                {"name": "client-1", "org": "nvidia", "type": "client"},
            ]
        }

        project_config = collections.OrderedDict(project_config)

        old_server_name = "server1"
        server_name = "server"
        update_project_server_name_config(project_config, old_server_name, server_name)
        servers = [p for p in project_config["participants"] if p["type"] == "server"]
        assert len(servers) == 1
        assert servers[0]["name"] == server_name

    def test_update_clients(self):
        project_config = {
            "participants": [
                {"name": "server1", "org": "nvidia", "type": "server"},
                {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin", "type": "admin"},
                {"name": "client-1", "org": "nvidia", "type": "client"},
            ]
        }

        project_config = collections.OrderedDict(project_config)
        clients = []
        n_clients = 3
        project_config = update_clients(clients, n_clients, project_config)
        result_clients = [p["name"] for p in project_config["participants"] if p["type"] == "client"]
        assert len(result_clients) == 3
        assert result_clients == ["site-1", "site-2", "site-3"]

        clients = ["client-1", "client-2", "client-3", "client-4"]
        n_clients = 3
        project_config = update_clients(clients, n_clients, project_config)
        result_clients = [p["name"] for p in project_config["participants"] if p["type"] == "client"]
        assert len(result_clients) == len(clients)
        assert result_clients == clients

    def test_prepare_builders(self):
        project_config = {
            "participants": [
                {"name": "server1", "org": "nvidia", "type": "server"},
                {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin", "type": "admin"},
                {"name": "client-1", "org": "nvidia", "type": "client"},
            ],
            "builders": [
                {
                    "path": "nvflare.lighter.impl.static_file.StaticFileBuilder",
                    "args": {"overseer_agent": {"args": {"sp_end_point": "server1: 8002: 8003"}}},
                },
                {"path": "nvflare.lighter.impl.cert.CertBuilder", "args": {}},
            ],
        }

        project_config = collections.OrderedDict(project_config)
        builders = prepare_builders(project_config)
        assert len(builders) == 2

    def test_get_packages_config(self):
        project_config = {
            "participants": [
                {"name": "server1", "org": "nvidia", "type": "server"},
                {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin", "type": "admin"},
                {"name": "client-1", "org": "nvidia", "type": "client"},
                {"name": "client-2", "org": "nvidia", "type": "client"},
            ],
            "builders": [
                {
                    "path": "nvflare.lighter.impl.static_file.StaticFileBuilder",
                    "args": {"overseer_agent": {"args": {"sp_end_point": "server1: 8002: 8003"}}},
                },
                {"path": "nvflare.lighter.impl.cert.CertBuilder", "args": {}},
            ],
        }

        project_config = collections.OrderedDict(project_config)

        global_config = get_service_config(project_config)
        assert "server1" == global_config[SC.FLARE_SERVER]
        assert "admin@nvidia.com" == global_config[SC.FLARE_PROJ_ADMIN]
        assert ["client-1", "client-2"] == global_config[SC.FLARE_CLIENTS]
