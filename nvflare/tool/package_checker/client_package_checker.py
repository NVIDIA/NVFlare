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
import re
import sys

from .check_rule import CheckGRPCServerAvailable
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
            CheckGRPCServerAvailable(name="Check GRPC server available", role=self.NVF_ROLE),
        ]

    def get_uid_from_startup_script(self) -> str:
        """Extract uid from sub_start.sh"""

        sub_start_path = os.path.join(self.package_path, "startup", "sub_start.sh")

        try:
            with open(sub_start_path, "r") as f:
                content = f.read()

            # Look for uid=value in the python command
            match = re.search(r"uid=([^\s]+)", content)
            if match:
                return match.group(1)

        except Exception as e:
            raise RuntimeError(f"Error reading {sub_start_path}: {e}")

        return None

    def get_dry_run_command(self) -> str:
        uid = self.get_uid_from_startup_script()
        if not uid:
            raise ValueError(
                f"Could not extract uid from {self.package_path}/startup/sub_start.sh. "
                "Possible reasons: the file may be missing, unreadable, or not in the expected format. "
                "Please check that the file exists, has the correct permissions, and contains a line with 'uid=<value>' in the Python command."
            )

        command = (
            f"{sys.executable} -m {CLIENT_SCRIPT}"
            f" -m {self.package_path} -s {self.NVF_CONFIG}"
            f" --set secure_train=true uid={uid} config_folder=config"
        )
        return command
