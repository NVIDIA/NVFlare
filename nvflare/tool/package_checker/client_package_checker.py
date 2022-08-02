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
from .utils import check_overseer_running, check_response, run_command_in_subprocess

CLIENT_SCRIPT = "nvflare.private.fed.app.client.client_train"
CLIENT_NVF_CONFIG = "fed_client.json"


class ClientPackageChecker(PackageChecker):
    def should_be_checked(self, package_path) -> bool:
        """Check if this package should be checked by this checker."""
        startup = os.path.join(package_path, "startup")
        if os.path.exists(os.path.join(startup, CLIENT_NVF_CONFIG)):
            return True
        return False

    def check(self, package_path):
        """Checks if the package is runnable on the current system."""
        startup = os.path.join(package_path, "startup")
        fed_config_file = os.path.join(startup, CLIENT_NVF_CONFIG)
        with open(fed_config_file, "r") as f:
            fed_config = json.load(f)

        # check overseer
        resp = check_overseer_running(startup=startup, overseer_agent_conf=fed_config["overseer_agent"], role="client")
        if not check_response(resp):
            self.add_report(package_path, "Can't connect to overseer", "Please check if overseer is up.")
        else:
            data = resp.json()
            psp = data.get("primary_sp")
            if not psp:
                self.add_report(package_path, "Can't get primary sp from overseer", "Please check if server is up.")

        # TODO:: check if the primary_sp's GRPC port or ADMIN port can be access
        #   primary_sp => "server_host_name:GRPC_PORT:ADMIN_PORT"

        # check if client can run
        if len(self.report[package_path]) == 0:
            command = (
                f"{sys.executable} -m {CLIENT_SCRIPT}"
                f" -m {package_path} -s {CLIENT_NVF_CONFIG}"
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

    def dry_run(self, package_path):
        command = (
            f"{sys.executable} -m {CLIENT_SCRIPT}"
            f" -m {package_path} -s {CLIENT_NVF_CONFIG}"
            " --set secure_train=false config_folder=config"
        )
        self.dry_run_process = run_command_in_subprocess(command)
