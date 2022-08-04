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
import signal
import sys
from subprocess import TimeoutExpired

from .package_checker import PackageChecker
from .utils import (
    check_grpc_server_running,
    check_overseer_running,
    check_response,
    check_socket_server_running,
    run_command_in_subprocess,
)

CLIENT_SCRIPT = "nvflare.private.fed.app.client.client_train"


class ClientPackageChecker(PackageChecker):
    NVF_CONFIG = "fed_client.json"

    def should_be_checked(self, package_path) -> bool:
        """Check if this package should be checked by this checker."""
        startup = os.path.join(package_path, "startup")
        if os.path.exists(os.path.join(startup, self.NVF_CONFIG)):
            return True
        return False

    def _check_overseer_and_service_provider_running_and_accessible(self, package_path: str, role: str):
        startup = os.path.join(package_path, "startup")
        fed_config_file = os.path.join(startup, self.NVF_CONFIG)
        with open(fed_config_file, "r") as f:
            fed_config = json.load(f)

        overseer_agent_conf = (
            fed_config["overseer_agent"] if role == "client" else fed_config["admin"]["overseer_agent"]
        )

        # check overseer connecting
        resp = check_overseer_running(startup=startup, overseer_agent_conf=overseer_agent_conf, role=role)
        if not check_response(resp):
            self.add_report(package_path, "Can't connect to overseer", "Please check if overseer is up.")
        else:
            data = resp.json()
            psp = data.get("primary_sp")
            if not psp:
                self.add_report(
                    package_path,
                    f"Can't get primary service provider ({psp}) from overseer",
                    "Please check if server is up.",
                )
            else:
                sp_end_point = psp["sp_end_point"]
                sp_name, grpc_port, admin_port = sp_end_point.split(":")
                if not check_socket_server_running(startup=startup, host=sp_name, port=int(admin_port)):
                    self.add_report(
                        package_path,
                        f"Can't connect to primary service provider's ({sp_end_point}) socketserver",
                        "Please check if server is up.",
                    )

                if not check_grpc_server_running(startup=startup, host=sp_name, port=int(grpc_port)):
                    self.add_report(
                        package_path,
                        f"Can't connect to primary service provider's ({sp_end_point}) grpc server",
                        "Please check if server is up.",
                    )

    def _check_dry_run(self, package_path: str):
        command = (
            f"{sys.executable} -m {CLIENT_SCRIPT}"
            f" -m {package_path} -s {self.NVF_CONFIG}"
            " --set secure_train=false config_folder=config"
        )
        process = run_command_in_subprocess(command)
        try:
            out, _ = process.communicate(timeout=3)
            self.add_report(
                package_path,
                f"Can't start client successfully: \n{out}",
                "Please check the error message of dry run.",
            )
        except TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)

    def check(self, package_path):
        """Checks if the package is runnable on the current system."""
        self._check_overseer_and_service_provider_running_and_accessible(package_path=package_path, role="client")

        # check if client can run
        if len(self.report[package_path]) == 0:
            self._check_dry_run(package_path=package_path)

    def dry_run(self, package_path):
        command = (
            f"{sys.executable} -m {CLIENT_SCRIPT}"
            f" -m {package_path} -s {self.NVF_CONFIG}"
            " --set secure_train=false config_folder=config"
        )
        self.dry_run_process = run_command_in_subprocess(command)
