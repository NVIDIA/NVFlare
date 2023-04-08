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

import os

from .check_rule import CheckAddressBinding
from .package_checker import PackageChecker
from .utils import NVFlareConfig


def _get_overseer_host_and_port(package_path: str):
    gunicorn_conf_file = os.path.join(package_path, "startup", NVFlareConfig.OVERSEER)
    gunicorn_conf = {}

    with open(gunicorn_conf_file, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            k, v = line.split("=")
            if v[0] == '"' and v[-1] == '"':
                v = str(v[1:-1])
            gunicorn_conf[k] = v
    address = gunicorn_conf["bind"]
    host, port = address.split(":")
    return host, int(port)


class OverseerPackageChecker(PackageChecker):
    def should_be_checked(self) -> bool:
        """Check if this package should be checked by this checker."""
        gunicorn_conf_file = os.path.join(self.package_path, "startup", NVFlareConfig.OVERSEER)
        if os.path.exists(gunicorn_conf_file):
            return True
        return False

    def init_rules(self, package_path):
        self.dry_run_timeout = 5
        self.rules = [
            CheckAddressBinding(
                name="Check overseer port binding", get_host_and_port_from_package=_get_overseer_host_and_port
            ),
        ]

    def get_dry_run_command(self) -> str:
        return os.path.join(self.package_path, "startup", "start.sh")
