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

import os
import signal
from subprocess import TimeoutExpired

from .client_package_checker import ClientPackageChecker
from .utils import run_command_in_subprocess


class AdminConsolePackageChecker(ClientPackageChecker):
    NVF_CONFIG = "fed_admin.json"

    def _check_dry_run(self, package_path: str):
        command = os.path.join(package_path, "startup", "fl_admin.sh")
        process = run_command_in_subprocess(command)
        try:
            out, _ = process.communicate(timeout=5)
            self.add_report(
                package_path,
                f"Can't start admin console successfully: \n{out}",
                "Please check the error message of dry run.",
            )
        except TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)

    def check(self, package_path):
        """Checks if the package is runnable on the current system."""
        self._check_overseer_and_service_provider_running_and_accessible(package_path=package_path, role="admin")

        # check if client can run
        if len(self.report[package_path]) == 0:
            self._check_dry_run(package_path=package_path)

    def dry_run(self, package_path):
        command = os.path.join(package_path, "startup", "fl_admin.sh")
        self.dry_run_process = run_command_in_subprocess(command)
