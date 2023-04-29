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

import pytest

from nvflare.cli_exception import CLIException
from nvflare.lighter.poc_commands import client_gpu_assignments, get_gpu_ids, get_package_command, update_clients, \
    prepare_builders
from nvflare.lighter.service_constants import FlareServiceConstants as SC
from nvflare.lighter.utils import update_project_server_name_config


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
        cmd = get_package_command(SC.CMD_START, "/tmp/nvflare/poc", SC.FLARE_SERVER)
        assert "/tmp/nvflare/poc/server/startup/start.sh" == cmd

        cmd = get_package_command(SC.CMD_START, "/tmp/nvflare/poc", SC.FLARE_PROJ_ADMIN)
        assert "/tmp/nvflare/poc/admin@nvidia.com/startup/fl_admin.sh" == cmd

        cmd = get_package_command(SC.CMD_START, "/tmp/nvflare/poc", "site-2000")
        assert "/tmp/nvflare/poc/site-2000/startup/start.sh" == cmd

    def test_update_server_name(self):
        project_config = {
            "participants": [
                {
                    "name": "server1",
                    "org": "nvidia",
                    "type": "server"
                },
                {
                    "name": "admin@nvidia.com",
                    "org": "nvidia",
                    "role": "project_admin",
                    "type": "admin"
                },
                {
                    "name": "client-1",
                    "org": "nvidia",
                    "type": "client"
                }
            ]
        }

        old_server_name = "server1"
        server_name = "server"
        update_project_server_name_config(project_config, old_server_name, server_name)
        servers = [p for p in project_config["participants"] if p["type"] == "server"]
        assert (len(servers) == 1)
        assert (servers[0]["name"] == server_name)

        overseer_agent_builder = {
            "args": {
                "overseer_agent": {
                    "args": {
                        "sp_end_point": "server1: 8002: 8003"
                    }

                }
            }
        }
        project_config["builders"] = [overseer_agent_builder]

        update_project_server_name_config(project_config, old_server_name, server_name)

        assert (project_config["builders"][0]["args"]["overseer_agent"]["args"]["sp_end_point"] == "server: 8002: 8003")

    def test_update_clients(self):
        project_config = {
            "participants": [
                {
                    "name": "server1",
                    "org": "nvidia",
                    "type": "server"
                },
                {
                    "name": "admin@nvidia.com",
                    "org": "nvidia",
                    "role": "project_admin",
                    "type": "admin"
                },
                {
                    "name": "client-1",
                    "org": "nvidia",
                    "type": "client"
                }
            ]
        }

        clients = []
        n_clients = 3
        project_config = update_clients(clients, n_clients, project_config)
        result_clients = [p["name"] for p in project_config["participants"] if p["type"] == "client"]
        assert (len(result_clients) == 3)
        assert (result_clients == ["site-1", "site-2", "site-3"])

        clients = ["client-1", "client-2", "client-3", "client-4"]
        n_clients = 3
        project_config = update_clients(clients, n_clients, project_config)
        result_clients = [p["name"] for p in project_config["participants"] if p["type"] == "client"]
        assert (len(result_clients) == len(clients))
        assert (result_clients == clients)

    def test_prepare_builders(self):
        project_config = {
            "builders": [
                {
                    "path": "nvflare.lighter.impl.static_file.StaticFileBuilder",
                    "args": {
                        "overseer_agent": {
                            "args": {
                                "sp_end_point": "server1: 8002: 8003"
                            }

                        }
                    }
                },
                {
                    "path": "nvflare.lighter.impl.cert.CertBuilder",
                    "args": {}
                },
            ]
        }

        builders = prepare_builders(project_config)
        assert (len(builders) == 2)
        for c in builders:
            assert (c.__class__.__name__ == "LocalStaticFileBuilder" or c.__class__.__name__ == "LocalCertBuilder")
            if c.__class__.__name__ == "LocalStaticFileBuilder":
                assert(c.get_server_name() == "localhost")
