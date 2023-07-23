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
from typing import Dict, Tuple

from nvflare.apis.app_validation import AppValidationKey, AppValidator
from nvflare.apis.fl_constant import JobConstants, SiteType, WorkspaceConstants
from nvflare.fuel.utils.config_factory import ConfigFactory


def _config_exists(app_root: str, config_folder: str, site_type: str) -> str:
    if site_type == SiteType.SERVER:
        config_to_check = JobConstants.SERVER_JOB_CONFIG
    elif site_type == SiteType.CLIENT:
        config_to_check = JobConstants.CLIENT_JOB_CONFIG
    else:
        config_to_check = None

    if config_to_check and ConfigFactory.load_config(config_to_check, [os.path.join(app_root, config_folder)]) is None:
        return f"Missing required config {config_to_check} inside {os.path.join(app_root, config_folder)} folder."
    return ""


class DefaultAppValidator(AppValidator):
    def __init__(self, site_type: str, config_folder: str = "config"):
        self._site_type = site_type
        self._config_folder = config_folder

    def validate(self, app_folder: str) -> Tuple[str, Dict]:
        result = dict()
        app_root = os.path.abspath(app_folder)
        if not os.path.exists(os.path.join(app_root, self._config_folder)):
            return "Missing config folder inside app folder.", {}

        err = _config_exists(app_root=app_root, config_folder=self._config_folder, site_type=self._site_type)
        if err:
            return err, {}

        if os.path.exists(os.path.join(app_root, WorkspaceConstants.CUSTOM_FOLDER_NAME)):
            result[AppValidationKey.BYOC] = True
        return "", result
