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

from nvflare.apis.app_validation import AppValidationKey, AppValidator
from nvflare.fuel.sec.authz import AuthorizationService, AuthzContext, Person

_RIGHT_BYOC = "byoc"


class AppAuthzService(object):

    app_validator = None

    @staticmethod
    def initialize(app_validator):
        if app_validator and not isinstance(app_validator, AppValidator):
            raise TypeError(f"app_validator must be an instance of AppValidator, but get {type(app_validator)}.")
        AppAuthzService.app_validator = app_validator

    @staticmethod
    def authorize(
        app_path: str,
        submitter_name: str,
        submitter_org: str,
        submitter_role: str,
    ) -> (bool, str):
        if not AppAuthzService.app_validator:
            return True, ""

        err, app_info = AppAuthzService.app_validator.validate(app_path)
        if err:
            return False, err

        app_has_custom_code = app_info.get(AppValidationKey.BYOC, False)
        if not app_has_custom_code:
            return True, ""

        ctx = AuthzContext(
            user=Person(submitter_name, submitter_org, submitter_role),
            submitter=Person(submitter_name, submitter_org, submitter_role),
            right=_RIGHT_BYOC,
        )

        authorized, err = AuthorizationService.authorize(ctx)
        if not authorized:
            return False, "BYOC not permitted"

        return True, ""
