# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Tuple

import json
import jwt
import requests

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey


class CustomSecurityHandler(FLComponent):
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.AUTHORIZE_COMMAND_CHECK:
            result, reason = self.authorize(fl_ctx=fl_ctx)
            if not result:
                fl_ctx.set_prop(FLContextKey.AUTHORIZATION_RESULT, False, sticky=False)
                fl_ctx.set_prop(FLContextKey.AUTHORIZATION_REASON, reason, sticky=False)

    def _validate_token(self, token):
        try:
            response = requests.get("http://localhost:8080/realms/myrealm")
            public_key = "-----BEGIN PUBLIC KEY-----\n" + \
                         json.loads(response.text).get("public_key") + \
                         "\n-----END PUBLIC KEY-----"

            access_token_json = jwt.decode(token, public_key, options={"verify_signature": False})
            user_name = access_token_json.get("preferred_username")
            if user_name:
                token_valid = True
            else:
                token_valid = False
        except:
            token_valid = False

        # print(f"_validate_token: {token_valid}")
        return token_valid

    def authorize(self, fl_ctx: FLContext) -> Tuple[bool, str]:
        command = fl_ctx.get_prop(FLContextKey.COMMAND_NAME)
        if command in ["check_resources", "submit_job"]:
            security_items = fl_ctx.get_prop(FLContextKey.SECURITY_ITEMS)
            job_meta = security_items.get(FLContextKey.JOB_META)
            auth_tokens = job_meta.get(JobMetaKey.CUSTOM_PROPS).get("auth_tokens")
            site_name = fl_ctx.get_identity_name()
            site_auth_token = auth_tokens.get(site_name).split(":")[1]

            if not self._validate_token(site_auth_token):
                return False, f"Not authorized to execute command: {command}"
            else:
                return True, ""
        else:
            return True, ""
