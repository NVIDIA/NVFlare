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

import time


class Client:
    def __init__(self, name, token) -> None:
        """Init Client.

        Represents a client, and is managed by the client manager.
        The token is a uuid used for authorization.

        Args:
            name: client name
            token: client token
        """
        self.name = name
        self.token = token
        self.last_connect_time = time.time()
        self.props = {}

    def set_token(self, token):
        self.token = token

    def get_token(self):
        return self.token

    def set_prop(self, name, value):
        self.props[name] = value

    def get_prop(self, name, default=None):
        return self.props.get(name, default)
