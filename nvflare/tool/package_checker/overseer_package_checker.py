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

from .package_checker import PackageChecker
from .utils import run_command_in_subprocess, try_bind_address


class OverseerPackageChecker(PackageChecker):
    def should_be_checked(self, package_path) -> bool:
        """Check if this package should be checked by this checker."""
        gunicorn_conf_file = os.path.join(package_path, "startup", "gunicorn.conf.py")
        if os.path.exists(gunicorn_conf_file):
            return True
        return False

    def check(self, package_path):
        """Checks if the package is runnable on the current system."""
        gunicorn_conf_file = os.path.join(package_path, "startup", "gunicorn.conf.py")
        gunicorn_conf = {}
        try:
            with open(gunicorn_conf_file, "r") as f:
                lines = f.read().splitlines()
                for line in lines:
                    k, v = line.split("=")
                    if v[0] == '"' and v[-1] == '"':
                        v = str(v[1:-1])
                    gunicorn_conf[k] = v
            address = gunicorn_conf["bind"]
            host, port = address.split(":")
            e = try_bind_address(host, int(port))
            if e:
                self.add_report(
                    package_path,
                    f"Can't bind to address ({address}) for overseer service: {e}",
                    "Please check the DNS and port.",
                )

            if len(self.report[package_path]) == 0:
                command = os.path.join(package_path, "startup", "start.sh")
                process = run_command_in_subprocess(command)
                try:
                    out, _ = process.communicate(timeout=10)
                    self.add_report(
                        package_path,
                        f"Can't start overseer successfully: \n{out}",
                        "Please check the error message of dry run.",
                    )
                except TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)

        except Exception as e:
            print(f"Package format is not correct: {e}")

    def dry_run(self, package_path):
        command = os.path.join(package_path, "startup", "start.sh")
        self.dry_run_process = run_command_in_subprocess(command)
