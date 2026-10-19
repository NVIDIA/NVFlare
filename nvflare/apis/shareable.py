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
from .fl_constant import ReservedKey, ReturnCode, ServerCommandKey, TaskResultReceipt


class ReservedHeaderKey:

    HEADERS = "__headers__"
    TOPIC = "__topic__"
    RC = ReservedKey.RC
    COOKIE_JAR = ReservedKey.COOKIE_JAR
    PEER_PROPS = "__peer_props__"
    REPLY_IS_LATE = "__reply_is_late__"
    TASK_NAME = ReservedKey.TASK_NAME
    TASK_ID = ReservedKey.TASK_ID
    TASK_ATTEMPT_ID = ReservedKey.TASK_ATTEMPT_ID
    TASK_ATTEMPT_REQUIRED = ReservedKey.TASK_ATTEMPT_REQUIRED
    TASK_RESULT_RECEIPT = "__task_result_receipt__"
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

    def get_task_result_receipt(self, task_id: str, attempt_id: str, workflow_id: str) -> str | None:
        """Return only a receipt bound to the exact submitted assignment."""
        expected = (
            (ReservedHeaderKey.TASK_ID, task_id),
            (ReservedHeaderKey.TASK_ATTEMPT_ID, attempt_id),
            (ReservedHeaderKey.WORKFLOW, workflow_id),
        )
        try:
            if any(not isinstance(value, str) or not value or self.get_header(key) != value for key, value in expected):
                return None
            receipt = self.get_header(ReservedHeaderKey.TASK_RESULT_RECEIPT)
            if isinstance(receipt, str) and receipt in (
                TaskResultReceipt.RECEIVED,
                TaskResultReceipt.TASK_CLOSED,
                TaskResultReceipt.RETRY,
            ):
                return receipt
        except ValueError:
            pass
        return None

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

    def get_task_attempt_id(self):
        """Return the scheduling authority's attempt ID, rejecting wire conflicts.

        Cookies let existing job-based clients echo the assignment identity
        unchanged. New clients also echo it in the header. Neither representation
        may contradict the other or downgrade an attempt-fenced assignment.
        ``None`` is reserved for legacy, unfenced task protocols.
        """
        cookie_jar = self.get_cookie_jar()
        if cookie_jar is not None and not isinstance(cookie_jar, dict):
            raise ValueError("task attempt cookie jar must be a dict")
        header = self.get_header(ReservedHeaderKey.TASK_ATTEMPT_ID)
        cookie = self.get_cookie(ReservedHeaderKey.TASK_ATTEMPT_ID)
        if header is not None and cookie is not None and header != cookie:
            raise ValueError("conflicting task attempt identities")
        attempt_id = cookie if cookie is not None else header
        if attempt_id is not None and (
            not isinstance(attempt_id, str) or not attempt_id.strip() or "\x00" in attempt_id
        ):
            raise ValueError("task attempt ID must be a non-empty string without NUL")
        required = False
        for value in (
            self.get_header(ReservedHeaderKey.TASK_ATTEMPT_REQUIRED),
            self.get_cookie(ReservedHeaderKey.TASK_ATTEMPT_REQUIRED),
        ):
            if value is not None and not isinstance(value, bool):
                raise ValueError("task attempt requirement must be a bool")
            required = required or value is True
        if required and attempt_id is None:
            raise ValueError("attempt-fenced task assignment requires a task attempt ID")
        return attempt_id

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


def make_copy(source: Shareable, exclude_headers: list = None) -> Shareable:
    """
    Make a copy from the source.
    The content (non-headers) will be kept intact. Headers will be deep-copied into the new instance.
    """
    assert isinstance(source, Shareable)
    c = copy.copy(source)
    headers = source.get(ReservedHeaderKey.HEADERS)
    if headers:
        new_headers = copy.deepcopy(headers)
        if exclude_headers:
            for k in exclude_headers:
                new_headers.pop(k, None)
    else:
        new_headers = {}
    c[ReservedHeaderKey.HEADERS] = new_headers
    return c
