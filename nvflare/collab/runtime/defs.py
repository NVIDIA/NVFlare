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
CALL_PROTOCOL_VERSION = 1


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
    BLOCKING = "blocking"


class CallHeaderKey:
    PROTOCOL_VERSION = "collab__protocol_version"
    TARGET_NAME = "collab__target_name"
    METHOD_NAME = "collab__method_name"
    # This header is overwritten by the receiver-side authorizer and is never
    # sent as an assertion of identity by CellDispatcher.
    AUTHENTICATED_CALLER = "collab__authenticated_caller"


class CallReplyKey:
    ERROR = "error"
    ERROR_TYPE = "error_type"
    ERROR_TRACEBACK = "error_traceback"
    RESULT = "result"
