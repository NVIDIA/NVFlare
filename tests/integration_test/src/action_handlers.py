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

from nvflare.fuel.flare_api.api_spec import AuthorizationError, InvalidJobDefinition
from nvflare.fuel.flare_api.flare_api import Session

if typing.TYPE_CHECKING:
    from .nvf_test_driver import NVFTestDriver

import time
from abc import ABC, abstractmethod

from tests.integration_test.src.utils import (
    build_authorization_error_details,
    build_authorization_error_message,
    build_error_response_items,
    check_job_done,
    normalize_invalid_job_definition_message,
    run_admin_api_tests,
)


class _CmdHandler(ABC):
    @abstractmethod
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: Session):
        pass


class _StartHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: Session):
        if command_args[0] == "server":
            if len(command_args) == 2:
                # if server id is provided
                server_ids = [command_args[1]]
            else:
                # start all servers
                server_ids = list(admin_controller.site_launcher.server_properties.keys())
            for sid in server_ids:
                admin_controller.site_launcher.start_server(sid)
            admin_controller.super_admin_api.try_connect(10.0)
        elif command_args[0] == "client":
            if len(command_args) == 2:
                # if client id is provided
                client_ids = [command_args[1]]
            else:
                # start all clients
                client_ids = list(admin_controller.site_launcher.client_properties.keys())
            for cid in client_ids:
                admin_controller.site_launcher.start_client(cid)
        else:
            raise RuntimeError(f"Target {command_args[0]} is not supported.")


class _KillHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: Session):
        if command_args[0] == "server":
            if len(command_args) == 2:
                # if server id is provided
                server_id = command_args[1]
            else:
                # kill active server
                server_id = admin_controller.site_launcher.get_active_server_id(admin_api.api.port)
            admin_controller.site_launcher.stop_server(server_id)
        elif command_args[0] == "client":
            if len(command_args) == 2:
                # if client id is provided
                client_ids = [command_args[1]]
            else:
                # close all clients
                client_ids = list(admin_controller.site_launcher.client_properties.keys())
            for cid in client_ids:
                admin_controller.remove_client(admin_api, admin_controller.site_launcher.client_properties[cid].name)
                admin_controller.site_launcher.stop_client(cid)


class _SleepHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: Session):
        time.sleep(int(command_args[0]))


class _AdminCommandsHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: Session):
        run_admin_api_tests(admin_api)


class _NoopHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: Session):
        pass


class _TestDoneHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: Session):
        admin_controller.test_done = True


class _SubmitJobHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: Session):
        admin_controller.admin_api_response = None
        job_name = str(command_args[0])
        job_path = admin_controller.resolve_job_path(job_name)
        try:
            admin_controller.job_id = admin_api.submit_job(job_path)
            admin_controller.last_job_name = job_name
        except AuthorizationError:
            admin_controller.admin_api_response = build_authorization_error_details(admin_api, "submit_job")
        except InvalidJobDefinition as e:
            admin_controller.admin_api_response = build_error_response_items(
                normalize_invalid_job_definition_message(str(e))
            )


class _CloneJobHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: Session):
        admin_controller.admin_api_response = None
        try:
            admin_controller.job_id = admin_api.clone_job(admin_controller.job_id)
        except AuthorizationError:
            admin_controller.admin_api_response = build_authorization_error_details(admin_api, "clone_job")


class _AbortJobHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: Session):
        admin_controller.admin_api_response = None
        try:
            admin_api.abort_job(admin_controller.job_id)
        except AuthorizationError:
            admin_controller.admin_api_response = build_authorization_error_details(admin_api, "abort_job")


class _ListJobHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: Session):
        admin_api.list_jobs()


class _ShellCommandHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: Session):
        admin_controller.admin_api_response = None
        if str(command_args[0]) == "ls":
            try:
                response = admin_api.ls_target(str(command_args[1]))
                admin_controller.admin_api_response = " ".join(response.splitlines())
            except AuthorizationError:
                admin_controller.admin_api_response = build_authorization_error_message(
                    admin_api, "ls", include_authz_prefix=False
                )


class _CheckJobHandler(_CmdHandler):
    def handle(self, command_args: list, admin_controller: NVFTestDriver, admin_api: Session):
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
