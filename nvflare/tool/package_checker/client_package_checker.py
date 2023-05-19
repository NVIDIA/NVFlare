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
import sys

from .check_rule import (
    CheckNonPrimarySPGRPCServerAvailable,
    CheckNonPrimarySPSocketServerAvailable,
    CheckOverseerRunning,
    CheckPrimarySPGRPCServerAvailable,
    CheckPrimarySPSocketServerAvailable,
    CheckSPListInResponse,
)
from .package_checker import PackageChecker
from .utils import NVFlareConfig, NVFlareRole

CLIENT_SCRIPT = "nvflare.private.fed.app.client.client_train"


class ClientPackageChecker(PackageChecker):
    NVF_CONFIG = NVFlareConfig.CLIENT
    NVF_ROLE = NVFlareRole.CLIENT

    def should_be_checked(self) -> bool:
        """Check if this package should be checked by this checker."""
        startup = os.path.join(self.package_path, "startup")
        if os.path.exists(os.path.join(startup, self.NVF_CONFIG)):
            return True
        return False

    def init_rules(self, package_path):
        self.rules = [
            [
                CheckOverseerRunning(name="Check overseer running", role=self.NVF_ROLE),
                CheckSPListInResponse(name="Check service provider list available"),
                CheckPrimarySPSocketServerAvailable(name="Check primary SP's socket server available"),
                CheckPrimarySPGRPCServerAvailable(name="Check primary SP's GRPC server available"),
                CheckNonPrimarySPSocketServerAvailable(
                    name="Check non-primary SP's socket server available", required=False
                ),
                CheckNonPrimarySPGRPCServerAvailable(
                    name="Check non-primary SP's GRPC server available", required=False
                ),
            ]
        ]

    def get_dry_run_command(self) -> str:
        command = (
            f"{sys.executable} -m {CLIENT_SCRIPT}"
            f" -m {self.package_path} -s {self.NVF_CONFIG}"
            " --set secure_train=true config_folder=config"
        )
        return command
