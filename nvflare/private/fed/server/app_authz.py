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

from nvflare.apis.app_validation import AppValidationKey, AppValidator
from nvflare.fuel.hci.server.authz import AuthzContext
from nvflare.security.security import Action, FLAuthzContext


class AppAuthzService(object):

    app_validator = None

    @staticmethod
    def initialize(app_validator):
        if app_validator and not isinstance(app_validator, AppValidator):
            raise TypeError(f"app_validator must be an instance of AppValidator, but get {type(app_validator)}.")
        AppAuthzService.app_validator = app_validator

    @staticmethod
    def _authorize_actions(app_path: str, sites: [str], actions) -> (str, AuthzContext):
        if AppAuthzService.app_validator:
            err, info = AppAuthzService.app_validator.validate(app_path)
            if err:
                return err, None

            byoc = info.get(AppValidationKey.BYOC, False)
            custom_datalist = info.get(AppValidationKey.CUSTOM_DATA_LIST, False)
            if byoc:
                actions.append(Action.BYOC)
            if custom_datalist:
                actions.append(Action.CUSTOM_DATALIST)

        return "", FLAuthzContext.new_authz_context(site_names=sites, actions=actions)

    @staticmethod
    def authorize_upload(app_path: str) -> (str, AuthzContext):
        return AppAuthzService._authorize_actions(app_path, ["server"], [Action.UPLOAD])

    @staticmethod
    def authorize_deploy(app_path: str, sites: [str]) -> (str, AuthzContext):
        return AppAuthzService._authorize_actions(app_path, sites, [Action.DEPLOY])
