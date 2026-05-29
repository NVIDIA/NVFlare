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

try:
    import numpy as np
except ImportError:
    np = None

from ..fuel.utils import fobs
from .fl_constant import ReservedKey, ReturnCode, ServerCommandKey

_NO_COPY_VALUE_TYPES = (bytes, bytearray, memoryview)
if np is not None:
    _NO_COPY_VALUE_TYPES = _NO_COPY_VALUE_TYPES + (np.ndarray,)


class ReservedHeaderKey:

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
    TASK_OPERATOR = "__task_operator__"
    ERROR = "__error__"
    PEER_CTX = ServerCommandKey.PEER_FL_CONTEXT
    MSG_ROOT_ID = "__msg_root_id__"
    MSG_ROOT_TTL = "__msg_root_ttl__"  # TTL = time to live
    PASS_THROUGH = "__pass_through__"  # request PASS_THROUGH decode at receiving CJ


class Shareable(dict):
    """The information communicated between server and client.

    Shareable is just a dict that can have any keys and values, defined by developers and users.
    It is recommended that keys are strings. Values must be serializable.
    """

    def __init__(self, data: dict | None = None):
        """Init the Shareable."""
        super().__init__()
        if data:
            self.update(data)
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
                raise ValueError(f"header object must be a dict, but got {type(header)}")
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

    def set_peer_context(self, peer_ctx):
        self.set_header(ReservedHeaderKey.PEER_CTX, peer_ctx)

    def get_peer_context(self):
        return self.get_header(ReservedHeaderKey.PEER_CTX)

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


def _collect_no_copy_values(value, no_copy_value_types: tuple, memo: dict, seen: set):
    value_id = id(value)
    if value_id in seen:
        return

    seen.add(value_id)
    if isinstance(value, no_copy_value_types):
        memo[value_id] = value
        return

    if isinstance(value, dict):
        for k, v in value.items():
            _collect_no_copy_values(k, no_copy_value_types, memo, seen)
            _collect_no_copy_values(v, no_copy_value_types, memo, seen)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _collect_no_copy_values(item, no_copy_value_types, memo, seen)
    else:
        attrs = getattr(value, "__dict__", None)
        if attrs:
            _collect_no_copy_values(attrs, no_copy_value_types, memo, seen)


def _normalize_no_copy_types(no_copy_types) -> tuple:
    if no_copy_types is None:
        return _NO_COPY_VALUE_TYPES
    if isinstance(no_copy_types, type):
        no_copy_types = (no_copy_types,)
    return _NO_COPY_VALUE_TYPES + tuple(no_copy_types)


def _make_no_copy_memo(source: Shareable, no_copy_types) -> dict:
    memo = {}
    _collect_no_copy_values(source, _normalize_no_copy_types(no_copy_types), memo, set())
    return memo


def make_copy(source: Shareable, exclude_headers: list = None, no_copy_types=None) -> Shareable:
    """Make a copy from the source.

    The content and headers will be deep-copied into the new instance, but large binary/array values are reused.
    Built-in no-copy values are bytes, bytearray, memoryview, and numpy.ndarray. Additional no-copy types can be
    supplied with no_copy_types. For example, to reuse PyTorch tensors by identity:

        import torch

        copied = make_copy(source, no_copy_types=(torch.Tensor,))

    Args:
        source: Shareable to copy.
        exclude_headers: Header keys to remove from the returned copy.
        no_copy_types: Additional value types to reuse by identity during deepcopy.
    """
    assert isinstance(source, Shareable)
    c = copy.deepcopy(source, memo=_make_no_copy_memo(source, no_copy_types))
    headers = c.get(ReservedHeaderKey.HEADERS)
    if headers:
        if exclude_headers:
            for k in exclude_headers:
                headers.pop(k, None)
    else:
        headers = {}
    c[ReservedHeaderKey.HEADERS] = headers
    return c
