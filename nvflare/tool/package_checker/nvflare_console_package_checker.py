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

from .check_rule import CHECK_PASSED
from .client_package_checker import ClientPackageChecker
from .package_checker import CheckStatus
from .utils import NVFlareConfig, NVFlareRole


class NVFlareConsolePackageChecker(ClientPackageChecker):
    NVF_CONFIG = NVFlareConfig.ADMIN
    NVF_ROLE = NVFlareRole.ADMIN

    def get_dry_run_command(self) -> str:
        return os.path.join(self.package_path, "startup", "fl_admin.sh")

    def get_dry_run_inputs(self):
        return os.path.basename(os.path.normpath(self.package_path))

    def check_dry_run(self) -> CheckStatus:
        startup = os.path.join(self.package_path, "startup")
        try:
            with open(os.path.join(startup, NVFlareConfig.ADMIN), "r") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            return super().check_dry_run()
        if config.get("admin", {}).get("ephemeral_admin_cert"):
            self.add_report("Check dry run", CHECK_PASSED, "N/A")
            return CheckStatus.PASS
        return super().check_dry_run()
