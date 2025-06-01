# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.edge.web.models.base_model import BaseModel, EdgeProtoKey


class UserInfo(BaseModel):
    def __init__(
        self,
        user_id: str = None,
        user_name: str = None,
        access_token: str = None,
        auth_token: str = None,
        auth_session: str = None,
        **kwargs,
    ):
        super().__init__()
        self.user_id = user_id
        self.user_name = user_name
        self.access_token = access_token
        self.auth_token = auth_token
        self.auth_session = auth_session

        if kwargs:
            self.update(kwargs)

    @staticmethod
    def extract_from_dict(d: dict):
        error = ""
        user_info_dict = d.pop(EdgeProtoKey.USER_INFO, None)
        if user_info_dict:
            user_info = UserInfo()
            user_info.update(user_info_dict)
        else:
            error = "missing user_info"
            user_info = None
        return error, user_info
