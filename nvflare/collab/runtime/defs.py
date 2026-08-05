# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
SYNC_TASK_NAME = "sync"
SETUP_TASK_NAME = "setup"

MSG_CHANNEL = "collab"
MSG_TOPIC = "call"

DIST_CHANNEL = "collab_distributed"


class DistributedTopic:
    HELLO = "hello"
    READY = "ready"
    INVOKE = "invoke"
    FINALIZE = "finalize"
    OUTBOUND = "outbound"
    CLOSE = "close"


class DistributedKey:
    PROTOCOL_VERSION = "protocol_version"
    PROTOCOL_ID = "protocol_id"
    AUTH_TOKEN = "auth_token"
    SESSION_ID = "session_id"
    PARENT_FQCN = "parent_fqcn"
    PARENT_URL = "parent_url"
    WORKER_FQCN = "worker_fqcn"
    SECURE_SUPPORTED = "secure_supported"
    STARTUP_TIMEOUT = "startup_timeout"
    OK = "ok"
    ERROR = "error"
    WORLD_SIZE = "world_size"
    TARGET = "target"
    REQUEST = "request"
    TIMEOUT = "timeout"
    SECURE = "secure"
    OPTIONAL = "optional"
    EXPECT_RESULT = "expect_result"
    TOPIC = "topic"
    PAYLOAD = "payload"
    HEADERS = "headers"


class SyncKey:
    COLLAB_INTERFACE = "collab_interface"
    CLIENT_INTERFACES = "client_interfaces"
    SERVER_FQCN = "server_fqcn"


class ObjectCallKey:
    CALLER = "caller"
    TARGET_NAME = "target_name"
    METHOD_NAME = "method_name"
    ARGS = "args"
    KWARGS = "kwargs"
    TIMEOUT = "timeout"


class CallReplyKey:
    ERROR = "error"
    ERROR_TYPE = "error_type"
    ERROR_TRACEBACK = "error_traceback"
    RESULT = "result"


def encode_message(message):
    """Convert an F3 Message to FOBS-safe wire data."""
    from nvflare.fuel.f3.message import Message

    if not isinstance(message, Message):
        raise TypeError(f"expected Message but got {type(message).__name__}")
    return {
        DistributedKey.HEADERS: message.headers or {},
        DistributedKey.PAYLOAD: message.payload,
    }


def decode_message(data):
    """Reconstruct an F3 Message from FOBS-safe wire data."""
    from nvflare.fuel.f3.message import Message

    if not isinstance(data, dict) or DistributedKey.HEADERS not in data:
        raise TypeError(f"expected encoded Message data but got {type(data).__name__}")
    headers = data[DistributedKey.HEADERS]
    if not isinstance(headers, dict):
        raise TypeError(f"encoded Message headers must be a dict but got {type(headers).__name__}")
    return Message(headers=headers, payload=data.get(DistributedKey.PAYLOAD))
