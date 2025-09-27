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

import json
import os
from abc import ABC, abstractmethod

from nvflare.tool.package_checker.utils import (
    NVFlareConfig,
    NVFlareRole,
    check_grpc_server_running,
    construct_dummy_overseer_response,
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
    def __init__(self, name: str, required: bool = True):
        """Creates a CheckRule.

        Args:
            name (str): name of the rule
            required (bool): whether this rule is required to pass.
        """
        self.name = name
        self.required = required

    @abstractmethod
    def __call__(self, package_path: str, data) -> CheckResult:
        """Returns problem and solution.

        Returns:
            A "CheckResult".
        """
        pass


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


def _get_primary_sp(sp_list):
    for sp in sp_list:
        if sp["primary"]:
            return sp
    return None


class CheckGRPCServerAvailable(CheckRule):
    def __init__(self, name: str, role: str):
        super().__init__(name)
        if role not in [NVFlareRole.SERVER, NVFlareRole.CLIENT, NVFlareRole.ADMIN]:
            raise RuntimeError(f"role {role} is not supported.")
        self.role = role

    def __call__(self, package_path, data):
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
            admin = fed_config["admin"]
            host = admin["host"]
            port = admin["port"]
            overseer_agent_conf = {
                "path": "nvflare.ha.dummy_overseer_agent.DummyOverseerAgent",
                "args": {"sp_end_point": f"{host}:{port}:{port}"},
            }
        else:
            overseer_agent_conf = fed_config["overseer_agent"]

        resp = construct_dummy_overseer_response(overseer_agent_conf=overseer_agent_conf, role=self.role)
        resp = resp.json()
        sp_list = resp.get("sp_list", [])
        psp = _get_primary_sp(sp_list)
        sp_end_point = psp["sp_end_point"]
        sp_name, grpc_port, admin_port = sp_end_point.split(":")

        if not check_grpc_server_running(startup=startup, host=sp_name, port=int(grpc_port)):
            return CheckResult(
                f"Can't connect to grpc server ({sp_name}:{grpc_port})",
                "Please check if server is up.",
            )
        return CheckResult(CHECK_PASSED, "N/A")
