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


class ClientDictKey:
    NAME = "name"
    FQCN = "fqcn"
    FQSN = "fqsn"
    IS_LEAF = "is_leaf"


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
        self.props = {ClientPropKey.FQCN: name, ClientPropKey.FQSN: name, ClientPropKey.IS_LEAF: True}

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

    def to_dict(self) -> dict:
        """Convert the Client object to a dict representation.
        This dict could be used included into a job's metadata.

        Returns: dict that contains essential info of the client.

        Note that the client's token is not included in the result since it is authentication data.

        """
        r = {ClientDictKey.NAME: self.name}

        fqcn = self.get_fqcn()
        if fqcn != self.name:
            r[ClientDictKey.FQCN] = fqcn

        fqsn = self.get_fqsn()
        if fqsn != self.name:
            r[ClientDictKey.FQSN] = fqsn

        is_leaf = self.get_is_leaf()
        if not is_leaf:
            r[ClientDictKey.IS_LEAF] = False

        return r


def from_dict(d: dict) -> Client:
    """Create a Client object from the data in the specified dict.

    Args:
        d: the dict that contains Client data. This dict should be the result of to_dict() of a Client object.

    Returns: a Client object

    """
    if not isinstance(d, dict):
        raise ValueError(f"expect client dict to be a dict but got {type(d)}")

    name = d.get(ClientDictKey.NAME)
    if not name:
        raise ValueError(f"missing '{ClientDictKey.NAME}' from client dict")

    c = Client(name=name, token="")

    # If FQCN is missing, default to name
    fqcn = d.get(ClientDictKey.FQCN, name)
    c.set_fqcn(fqcn)

    # If FQSN is missing, default to name
    fqsn = d.get(ClientDictKey.FQSN, name)
    c.set_fqsn(fqsn)

    # If IS_LEAF is missing, default to True
    is_leaf = d.get(ClientDictKey.IS_LEAF, True)
    c.set_is_leaf(is_leaf)

    return c
