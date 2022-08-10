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
    try_bind_address,
    try_write_dir,
)


class CheckResult:
    def __init__(self, problem="", solution="", data=None):
        self.data = data
        self.problem = problem
        self.solution = solution


class CheckRule(ABC):
    @abstractmethod
    def __call__(self, package_path: str, data) -> CheckResult:
        """Returns problem and solution.

        Returns:
            A "CheckResult".
        """
        pass


class CheckOverseerRunning(CheckRule):
    def __init__(self, role: str):
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

        resp = check_overseer_running(startup=startup, overseer_agent_conf=overseer_agent_conf, role=self.role)
        if not check_response(resp):
            return CheckResult("Can't connect to overseer", "Please check if overseer is up.")
        return CheckResult("", "", resp)


class CheckAddressBinding(CheckRule):
    def __init__(self, get_host_and_port_from_package):
        self.get_host_and_port_from_package = get_host_and_port_from_package

    def __call__(self, package_path, data=None):
        host, port = self.get_host_and_port_from_package(package_path)
        e = try_bind_address(host, port)
        if e:
            return CheckResult(
                f"Can't bind to address ({host}:{port}): {e}",
                "Please check the DNS and port.",
            )
        return CheckResult("", "")


class CheckWriting(CheckRule):
    def __init__(self, get_filename_from_package):
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
        return CheckResult("", "")


class CheckPrimarySPInResponse(CheckRule):
    def __call__(self, package_path, data):
        data = data.json()
        psp = data.get("primary_sp")
        if not psp:
            return CheckResult(
                f"Can't get primary service provider ({psp}) from overseer",
                "Please check if server is up.",
            )
        return CheckResult("", "", psp)


class CheckSPSocketServerAvailable(CheckRule):
    def __call__(self, package_path, data):
        startup = os.path.join(package_path, "startup")
        sp_end_point = data["sp_end_point"]
        sp_name, grpc_port, admin_port = sp_end_point.split(":")
        if not check_socket_server_running(startup=startup, host=sp_name, port=int(admin_port)):
            return CheckResult(
                f"Can't connect to primary service provider's ({sp_end_point}) socketserver",
                "Please check if server is up.",
            )
        return CheckResult("", "", data)


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
        return CheckResult("", "", data)
