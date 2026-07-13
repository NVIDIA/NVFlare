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
SYNC_TASK_NAME = "sync"

MSG_CHANNEL = "collab"
MSG_TOPIC = "call"


class SyncKey:
    COLLAB_INTERFACE = "collab_interface"


class ObjectCallKey:
    CALLER = "caller"
    TARGET_NAME = "target_name"
    METHOD_NAME = "method_name"
    ARGS = "args"
    KWARGS = "kwargs"
    TIMEOUT = "timeout"
    BLOCKING = "blocking"


class CallReplyKey:
    ERROR = "error"
    RESULT = "result"
