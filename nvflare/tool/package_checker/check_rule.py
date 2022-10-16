# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
from abc import ABC, abstractmethod

from nvflare.tool.package_checker.utils import (
    NVFlareConfig,
    NVFlareRole,
    check_grpc_server_running,
    check_overseer_running,
    check_response,
    check_socket_server_running,
    parse_overseer_agent_args,
    try_bind_address,
    try_write_dir,
)

CHECK_PASSED = "PASSED"


class CheckResult:
    def __init__(self, problem="", solution="", data=None):
        self.problem = problem
        self.solution = solution
        self.data = data


class CheckRule(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, package_path: str, data) -> CheckResult:
        """Returns problem and solution.

        Returns:
            A "CheckResult".
        """
        pass


class CheckOverseerRunning(CheckRule):
    def __init__(self, name: str, role: str):
        super().__init__(name)
        if role not in [NVFlareRole.SERVER, NVFlareRole.CLIENT, NVFlareRole.ADMIN]:
            raise RuntimeError(f"role {role} is not supported.")
        self.role = role

    def __call__(self, package_path, data=None):
        startup = os.path.join(package_path, "startup")
        if self.role == NVFlareRole.SERVER:
            nvf_config = NVFlareConfig.SERVER
        elif self.role == NVFlareRole.CLIENT:
            nvf_config = NVFlareConfig.CLIENT
        else:
            nvf_config = NVFlareConfig.ADMIN

        fed_config_file = os.path.join(startup, nvf_config)
        with open(fed_config_file, "r") as f:
            fed_config = json.load(f)

        if self.role == NVFlareRole.ADMIN:
            overseer_agent_conf = fed_config["admin"]["overseer_agent"]
        else:
            overseer_agent_conf = fed_config["overseer_agent"]

        required_args = ["overseer_end_point", "role", "project", "name"]
        if self.role == NVFlareRole.SERVER:
            required_args.extend(["fl_port", "admin_port"])

        overseer_agent_args = parse_overseer_agent_args(overseer_agent_conf, required_args)
        resp, err = check_overseer_running(startup=startup, overseer_agent_args=overseer_agent_args, role=self.role)
        if err:
            return CheckResult(
                f"Can't connect to overseer ({overseer_agent_args['overseer_end_point']}): {err}",
                "1) Please check if overseer is up or certificates are correct."
                + "2) Please check if overseer hostname in project.yml is available."
                + "3) if running in local machine, check if overseer defined in project.yml is defined in /etc/hosts",
            )
        elif not check_response(resp):
            return CheckResult(
                f"Can't connect to overseer ({overseer_agent_args['overseer_end_point']})",
                "1) Please check if overseer is up or certificates are correct."
                + "2) Please check if overseer hostname in project.yml is available."
                + "3) if running in local machine, check if overseer defined in project.yml is defined in /etc/hosts",
            )
        return CheckResult(CHECK_PASSED, "N/A", resp)


class CheckAddressBinding(CheckRule):
    def __init__(self, name: str, get_host_and_port_from_package):
        super().__init__(name)
        self.get_host_and_port_from_package = get_host_and_port_from_package

    def __call__(self, package_path, data=None):
        host, port = self.get_host_and_port_from_package(package_path)
        e = try_bind_address(host, port)
        if e:
            return CheckResult(
                f"Can't bind to address ({host}:{port}): {e}",
                "Please check the DNS and port.",
            )
        return CheckResult(CHECK_PASSED, "N/A")


class CheckWriting(CheckRule):
    def __init__(self, name: str, get_filename_from_package):
        super().__init__(name)
        self.get_filename_from_package = get_filename_from_package

    def __call__(self, package_path, data=None):
        path_to_write = self.get_filename_from_package(package_path)
        e = None
        if path_to_write:
            e = try_write_dir(path_to_write)
        if e:
            return CheckResult(
                f"Can't write to {path_to_write}: {e}.",
                "Please check the user permission.",
            )
        return CheckResult(CHECK_PASSED, "N/A")


class CheckPrimarySPInResponse(CheckRule):
    def __call__(self, package_path, data):
        data = data.json()
        psp = data.get("primary_sp")
        if not psp:
            return CheckResult(
                "Can't get primary service provider from overseer",
                "Please contact NVFLARE system admin and make sure at least one of the FL servers"
                + " is up and can connect to overseer.",
            )
        return CheckResult(CHECK_PASSED, "N/A", psp)


class CheckSPSocketServerAvailable(CheckRule):
    def __call__(self, package_path, data):
        startup = os.path.join(package_path, "startup")
        sp_end_point = data["sp_end_point"]
        sp_name, grpc_port, admin_port = sp_end_point.split(":")
        if not check_socket_server_running(startup=startup, host=sp_name, port=int(admin_port)):
            return CheckResult(
                f"Can't connect to ({sp_end_point})/DNS can't resolve its ip",
                f" 1) If ({sp_end_point}) is public, check internet connection, try ping  ({sp_end_point})."
                + f" 2) If ({sp_end_point}) is private, then you need to add its ip to the etc/hosts."
                + "3) If network is good, Please contact NVFLARE system admin and make sure the primary FL server"
                + "is running.",
            )
        return CheckResult(CHECK_PASSED, "N/A", data)


class CheckSPGRPCServerAvailable(CheckRule):
    def __call__(self, package_path, data):
        startup = os.path.join(package_path, "startup")
        sp_end_point = data["sp_end_point"]
        sp_name, grpc_port, admin_port = sp_end_point.split(":")

        if not check_grpc_server_running(startup=startup, host=sp_name, port=int(grpc_port)):
            return CheckResult(
                f"Can't connect to primary service provider's ({sp_end_point}) grpc server",
                "Please check if server is up.",
            )
        return CheckResult(CHECK_PASSED, "N/A", data)
