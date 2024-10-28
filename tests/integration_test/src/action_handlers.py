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

from __future__ import annotations

import typing

from nvflare.fuel.hci.client.api_status import APIStatus

if typing.TYPE_CHECKING:
    from .nvf_test_driver import NVFTestDriver

import time
from abc import ABC, abstractmethod

from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI
from tests.integration_test.src.utils import check_job_done, run_admin_api_tests


class _CmdHandler(ABC):
    @abstractmethod
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: FLAdminAPI):
        pass


class _StartHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: FLAdminAPI):
        if command_args[0] == "server":
            if len(command_args) == 2:
                # if server id is provided
                server_ids = [command_args[1]]
            else:
                # start all servers
                server_ids = list(admin_controller.site_launcher.server_properties.keys())
            for sid in server_ids:
                admin_controller.site_launcher.start_server(sid)
            admin_controller.super_admin_api.login(username=admin_controller.super_admin_user_name)
        elif command_args[0] == "client":
            if len(command_args) == 2:
                # if client id is provided
                client_ids = [command_args[1]]
            else:
                # start all clients
                client_ids = list(admin_controller.site_launcher.client_properties.keys())
            for cid in client_ids:
                admin_controller.site_launcher.start_client(cid)
        elif command_args[0] == "overseer":
            admin_controller.site_launcher.start_overseer()
        else:
            raise RuntimeError(f"Target {command_args[0]} is not supported.")


class _KillHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: FLAdminAPI):
        if command_args[0] == "server":
            if len(command_args) == 2:
                # if server id is provided
                server_id = command_args[1]
            else:
                # kill active server
                server_id = admin_controller.site_launcher.get_active_server_id(admin_api.port)
            admin_controller.site_launcher.stop_server(server_id)
        elif command_args[0] == "overseer":
            admin_controller.site_launcher.stop_overseer()
        elif command_args[0] == "client":
            if len(command_args) == 2:
                # if client id is provided
                client_ids = [command_args[1]]
            else:
                # close all clients
                client_ids = list(admin_controller.site_launcher.client_properties.keys())
            for cid in client_ids:
                admin_api.remove_client([admin_controller.site_launcher.client_properties[cid].name])
                admin_controller.site_launcher.stop_client(cid)


class _SleepHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: FLAdminAPI):
        time.sleep(int(command_args[0]))


class _AdminCommandsHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: FLAdminAPI):
        run_admin_api_tests(admin_api)


class _NoopHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: FLAdminAPI):
        pass


class _TestDoneHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: FLAdminAPI):
        admin_controller.test_done = True


class _SubmitJobHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: FLAdminAPI):
        job_name = str(command_args[0])

        response = admin_api.submit_job(job_name)
        if response["status"] == APIStatus.ERROR_RUNTIME:
            admin_controller.admin_api_response = response.get("raw", {}).get("data")
        elif response["status"] == APIStatus.ERROR_AUTHORIZATION:
            admin_controller.admin_api_response = response["details"]
        elif response["status"] == APIStatus.SUCCESS:
            admin_controller.job_id = response["details"]["job_id"]
            admin_controller.last_job_name = job_name


class _CloneJobHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: FLAdminAPI):
        response = admin_api.clone_job(admin_controller.job_id)
        if response["status"] == APIStatus.ERROR_RUNTIME:
            admin_controller.admin_api_response = response.get("raw", {}).get("data")
        elif response["status"] == APIStatus.ERROR_AUTHORIZATION:
            admin_controller.admin_api_response = response["details"]
        if response["status"] == APIStatus.SUCCESS:
            admin_controller.job_id = response["details"]["job_id"]


class _AbortJobHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: FLAdminAPI):
        response = admin_api.abort_job(admin_controller.job_id)
        if response["status"] == APIStatus.ERROR_RUNTIME:
            admin_controller.admin_api_response = response.get("raw", {}).get("data")
        elif response["status"] == APIStatus.ERROR_AUTHORIZATION:
            admin_controller.admin_api_response = response["details"]


class _ListJobHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: FLAdminAPI):
        response = admin_api.list_jobs()
        assert response["status"] == APIStatus.SUCCESS


class _ShellCommandHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: FLAdminAPI):
        if str(command_args[0]) == "ls":
            response = admin_api.ls_target(str(command_args[1]))
            if response["status"] == APIStatus.ERROR_RUNTIME:
                admin_controller.admin_api_response = response.get("raw", {}).get("data")
            elif response["status"] == APIStatus.ERROR_AUTHORIZATION:
                admin_controller.admin_api_response = response["details"]["message"]
            elif response["status"] == APIStatus.SUCCESS:
                admin_controller.admin_api_response = " ".join(response["details"]["message"].splitlines())


class _CheckJobHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: FLAdminAPI):
        timeout = 1
        if command_args:
            timeout = float(command_args[0])
        start_time = time.time()
        result = False
        if admin_controller.job_id:
            while time.time() - start_time < timeout:
                result = check_job_done(job_id=admin_controller.job_id, admin_api=admin_controller.super_admin_api)
                if result:
                    break
                time.sleep(0.5)
        admin_controller.test_done = result
