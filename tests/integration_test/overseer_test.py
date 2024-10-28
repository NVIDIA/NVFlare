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

import time

import pytest

from tests.integration_test.src import OALauncher


@pytest.mark.xdist_group(name="overseer_tests_group")
class TestOverseer:
    def test_overseer_server_down_and_up(self):
        oa_launcher = OALauncher()
        try:
            oa_launcher.start_overseer()
            time.sleep(1)
            server_agent_list = oa_launcher.start_servers(2)
            client_agent_list = oa_launcher.start_clients(4)
            time.sleep(10)
            psp = oa_launcher.get_primary_sp(client_agent_list[0])
            assert psp.name == "server00"
            oa_launcher.pause_server(server_agent_list[0])
            time.sleep(20)
            psp = oa_launcher.get_primary_sp(client_agent_list[0])
            assert psp.name == "server01"
            oa_launcher.resume_server(server_agent_list[0])
            psp = oa_launcher.get_primary_sp(client_agent_list[0])
            assert psp.name == "server01"
            time.sleep(10)
            psp = oa_launcher.get_primary_sp(client_agent_list[0])
            assert psp.name == "server01"
        finally:
            oa_launcher.stop_clients()
            oa_launcher.stop_servers()
            oa_launcher.stop_overseer()

    def test_overseer_client_down_and_up(self):
        oa_launcher = OALauncher()
        try:
            oa_launcher.start_overseer()
            time.sleep(10)
            _ = oa_launcher.start_servers(1)
            client_agent_list = oa_launcher.start_clients(1)
            time.sleep(10)
            psp = oa_launcher.get_primary_sp(client_agent_list[0])
            assert psp.name == "server00"
            oa_launcher.pause_client(client_agent_list[0])
            time.sleep(10)
            psp = oa_launcher.get_primary_sp(client_agent_list[0])
            assert psp.name == "server00"
            oa_launcher.resume_client(client_agent_list[0])
            time.sleep(10)
            psp = oa_launcher.get_primary_sp(client_agent_list[0])
            assert psp.name == "server00"
        finally:
            oa_launcher.stop_clients()
            oa_launcher.stop_servers()
            oa_launcher.stop_overseer()

    def test_overseer_overseer_down_and_up(self):
        oa_launcher = OALauncher()
        try:
            oa_launcher.start_overseer()
            time.sleep(10)
            _ = oa_launcher.start_servers(1)
            client_agent_list = oa_launcher.start_clients(4)
            time.sleep(10)
            for client_agent in client_agent_list:
                psp = oa_launcher.get_primary_sp(client_agent)
                assert psp.name == "server00"
            oa_launcher.stop_overseer()
            time.sleep(10)
            oa_launcher.start_overseer()
            time.sleep(20)
            for client_agent in client_agent_list:
                psp = oa_launcher.get_primary_sp(client_agent)
                assert psp.name == "server00"
        finally:
            oa_launcher.stop_clients()
            oa_launcher.stop_servers()
            oa_launcher.stop_overseer()
