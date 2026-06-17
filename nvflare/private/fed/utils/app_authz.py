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
from nvflare.app_opt.flower.defs import Constant as FlowerConstant
from nvflare.fuel.sec.authz import AuthorizationService, AuthzContext, Person

_RIGHT_BYOC = "byoc"
_RIGHT_FLOWER_PREDEPLOYED = "server-predeployed-flwr"


class AppAuthzService(object):

    app_validator = None

    @staticmethod
    def initialize(app_validator):
        if app_validator and not isinstance(app_validator, AppValidator):
            raise TypeError(f"app_validator must be an instance of AppValidator, but get {type(app_validator)}.")
        AppAuthzService.app_validator = app_validator

    @staticmethod
    def validate_app(app_path: str) -> (str, dict):
        if not AppAuthzService.app_validator:
            return "", {}

        err, app_info = AppAuthzService.app_validator.validate(app_path)
        if err:
            return err, {}
        if not isinstance(app_info, dict):
            return f"app validator must return app info as dict but got {type(app_info)}", {}

        return "", app_info

    @staticmethod
    def authorize_app_info(
        app_info: dict,
        submitter_name: str,
        submitter_org: str,
        submitter_role: str,
        job_meta: dict = None,
    ) -> (bool, str):
        app_has_custom_code = app_info.get(AppValidationKey.BYOC, False)
        if app_has_custom_code:
            ctx = AuthzContext(
                user=Person(submitter_name, submitter_org, submitter_role),
                submitter=Person(submitter_name, submitter_org, submitter_role),
                right=_RIGHT_BYOC,
            )
            authorized, err = AuthorizationService.authorize(ctx)
            if not authorized:
                return False, "BYOC not permitted"

        # Guard for server-predeployed Flower app mode.
        # If job_meta is provided, check the flag directly.
        is_flower_predeployed = job_meta.get(FlowerConstant.FLOWER_PREDEPLOYED, False) if job_meta else False

        if is_flower_predeployed:
            ctx = AuthzContext(
                user=Person(submitter_name, submitter_org, submitter_role),
                submitter=Person(submitter_name, submitter_org, submitter_role),
                right=_RIGHT_FLOWER_PREDEPLOYED,
            )
            authorized, err = AuthorizationService.authorize(ctx)
            if not authorized:
                return False, (
                    "Server-predeployed Flower app mode is not permitted on this site. "
                    "A site admin must grant the 'server-predeployed-flwr' right in "
                    "authorization.json to enable this mode."
                )

        return True, ""

    @staticmethod
    def authorize(
        app_path: str,
        submitter_name: str,
        submitter_org: str,
        submitter_role: str,
        job_meta: dict = None,
    ) -> (bool, str):
        err, app_info = AppAuthzService.validate_app(app_path)
        if err:
            return False, err

        return AppAuthzService.authorize_app_info(
            app_info=app_info,
            submitter_name=submitter_name,
            submitter_org=submitter_org,
            submitter_role=submitter_role,
            job_meta=job_meta,
        )
