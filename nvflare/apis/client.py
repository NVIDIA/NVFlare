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

import time


class ClientPropKey:

    FQCN = "fqcn"  # Fully Qualified Cell Name: position in Cellnet
    FQSN = "fqsn"  # Fully Qualified Site Name: position in client hierarchy
    IS_LEAF = "is_leaf"  # Whether the client is a leaf node in client hierarchy


class Client:
    def __init__(self, name, token) -> None:
        """Init Client.

        Represents a client, and is managed by the client manager.
        The token is an uuid used for authorization.

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

    def set_fqcn(self, value: str):
        self.set_prop(ClientPropKey.FQCN, value)

    def get_fqcn(self):
        return self.get_prop(ClientPropKey.FQCN)

    def set_fqsn(self, value: str):
        self.set_prop(ClientPropKey.FQSN, value)

    def get_fqsn(self):
        return self.get_prop(ClientPropKey.FQSN)

    def set_is_leaf(self, value: bool):
        self.set_prop(ClientPropKey.IS_LEAF, value)

    def get_is_leaf(self):
        return self.get_prop(ClientPropKey.IS_LEAF)
