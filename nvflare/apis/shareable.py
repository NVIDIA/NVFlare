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
import copy

from ..fuel.utils import fobs
from .fl_constant import ReservedKey, ReturnCode


class ReservedHeaderKey(object):

    HEADERS = "__headers__"
    TOPIC = "__topic__"
    RC = ReservedKey.RC
    COOKIE_JAR = ReservedKey.COOKIE_JAR
    PEER_PROPS = "__peer_props__"
    REPLY_IS_LATE = "__reply_is_late__"
    TASK_NAME = ReservedKey.TASK_NAME
    TASK_ID = ReservedKey.TASK_ID
    WORKFLOW = ReservedKey.WORKFLOW
    AUDIT_EVENT_ID = ReservedKey.AUDIT_EVENT_ID
    CONTENT_TYPE = "__content_type__"


class Shareable(dict):
    """The information communicated between server and client.

    Shareable is just a dict that can have any keys and values, defined by developers and users.
    It is recommended that keys are strings. Values must be serializable.
    """

    def __init__(self):
        """Init the Shareable."""
        super().__init__()
        self[ReservedHeaderKey.HEADERS] = {}

    def set_header(self, key: str, value):
        header = self.get(ReservedHeaderKey.HEADERS, None)
        if not header:
            header = {}
            self[ReservedHeaderKey.HEADERS] = header
        header[key] = value

    def get_header(self, key: str, default=None):
        header = self.get(ReservedHeaderKey.HEADERS, None)
        if not header:
            return default
        else:
            if not isinstance(header, dict):
                raise ValueError("header object must be a dict, but got {}".format(type(header)))
            return header.get(key, default)

    # some convenience methods
    def get_return_code(self, default=ReturnCode.OK):
        return self.get_header(ReservedHeaderKey.RC, default)

    def set_return_code(self, rc):
        self.set_header(ReservedHeaderKey.RC, rc)

    def add_cookie(self, name: str, data):
        """Add a cookie that is to be sent to the client and echoed back in response.

        This method is intended to be called by the Server side.

        Args:
            name: the name of the cookie
            data: the data of the cookie, which must be serializable

        """
        cookie_jar = self.get_cookie_jar()
        if not cookie_jar:
            cookie_jar = {}
            self.set_header(key=ReservedHeaderKey.COOKIE_JAR, value=cookie_jar)
        cookie_jar[name] = data

    def get_cookie_jar(self):
        return self.get_header(key=ReservedHeaderKey.COOKIE_JAR, default=None)

    def set_cookie_jar(self, jar):
        self.set_header(key=ReservedHeaderKey.COOKIE_JAR, value=jar)

    def get_cookie(self, name: str, default=None):
        jar = self.get_cookie_jar()
        if not jar:
            return default
        return jar.get(name, default)

    def set_peer_props(self, props: dict):
        self.set_header(ReservedHeaderKey.PEER_PROPS, props)

    def get_peer_props(self):
        return self.get_header(ReservedHeaderKey.PEER_PROPS, None)

    def get_peer_prop(self, key: str, default):
        props = self.get_peer_props()
        if not isinstance(props, dict):
            return default
        return props.get(key, default)

    def to_bytes(self) -> bytes:
        """Serialize the Model object into bytes.

        Returns:
            object serialized in bytes.

        """
        return fobs.dumps(self)

    @classmethod
    def from_bytes(cls, data: bytes):
        """Convert the data bytes into Model object.

        Args:
            data: a bytes object

        Returns:
            an object loaded by FOBS from data

        """
        return fobs.loads(data)


# some convenience functions
def make_reply(rc, headers=None) -> Shareable:
    reply = Shareable()
    reply.set_return_code(rc)
    if headers and isinstance(headers, dict):
        for k, v in headers.items():
            reply.set_header(k, v)
    return reply


def make_copy(source: Shareable) -> Shareable:
    """
    Make a copy from the source.
    The content (non-headers) will be kept intact. Headers will be deep-copied into the new instance.
    """
    assert isinstance(source, Shareable)
    c = copy.copy(source)
    headers = source.get(ReservedHeaderKey.HEADERS, None)
    if headers:
        new_headers = copy.deepcopy(headers)
    else:
        new_headers = {}
    c[ReservedHeaderKey.HEADERS] = new_headers
    return c
