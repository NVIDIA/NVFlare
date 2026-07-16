# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Protocol vocabulary for the Client API control protocol over Cell.

This module is the control-protocol vocabulary implemented by the trainer-side Cell
engine and the external-process backend. It is a pure vocabulary module: constants
only, no Cell/cellnet imports and no I/O.
"""

# Cell channel used for all Client API control protocol messages.
# Clearly namespaced so it cannot collide with the legacy FlareAgent channel
# ("flare_agent" in nvflare/client/ipc/defs.py) or other Cell channels.
# This is a frozen wire constant; its exact value is part of the cross-track contract.
CHANNEL = "client_api"

# The Client API control protocol version carried in HELLO.
# V1 supports exactly one protocol version; the field exists so later versions
# can define a compatibility window.
PROTOCOL_VERSION = 1


class Topic:
    """Topics of the Client API control protocol messages over the Cell CHANNEL.

    Reply-type messages (HELLO_ACCEPTED, TASK_ACCEPTED, RESULT_ACCEPTED and
    RESULT_REJECTED) are modeled as Cell request replies rather than separate sends.

    Values are prefixed with "client_api." so they never collide with legacy topic values
    (e.g. "hello"/"heartbeat"/"abort"/"bye" in nvflare/client/ipc/defs.py).
    """

    # Session setup (all out-of-process modes)
    HELLO = "client_api.hello"
    HELLO_ACCEPTED = "client_api.hello_accepted"
    # Distinct from ERROR (a protocol/transport error): HELLO_REJECTED is a clean, semantic
    # auth/handshake refusal (bad proof, wrong scope, consumed/expired nonce, single-session),
    # so consumers can tell a recoverable auth failure from a protocol fault.
    HELLO_REJECTED = "client_api.hello_rejected"

    # Per task (every round)
    TASK_READY = "client_api.task_ready"
    TASK_ACCEPTED = "client_api.task_accepted"
    TASK_FAILED = "client_api.task_failed"
    RESULT_READY = "client_api.result_ready"
    RESULT_ACCEPTED = "client_api.result_accepted"
    RESULT_REJECTED = "client_api.result_rejected"

    # Throughout the session
    LOG = "client_api.log"
    HEARTBEAT = "client_api.heartbeat"

    # Teardown / failure
    ABORT = "client_api.abort"
    SHUTDOWN = "client_api.shutdown"
    ERROR = "client_api.error"


class MsgKey:
    """Keys for Client API control protocol message payloads.

    These string values are the implemented wire contract; renaming a value is a protocol
    break, not a refactor.
    """

    SESSION_ID = "session_id"
    JOB_ID = "job_id"
    SITE_NAME = "site_name"
    TRAINER_FQCN = "trainer_fqcn"
    RANK = "rank"
    PROTOCOL_VERSION = "protocol_version"
    PROOF = "proof"
    REASON = "reason"
    TASK_ID = "task_id"
    # TASK_READY carries the task name alongside the task id.
    TASK_NAME = "task_name"
    # TASK_READY carries the task Shareable. Cell/FOBS decides whether its payload is inline
    # or represented by a ViaDownloader reference.
    MODEL = "model"
    # The task/result Shareables ride directly in their Cell requests. Cell's FOBS
    # encoder selects inline encoding or ViaDownloader references as appropriate.
    RESULT = "result"
    # SHUTDOWN reply truth: True while flare.send() still owns the accepted-result
    # publication barrier, including the RESULT_ACCEPTED reply race for an inline result;
    # False once the send barrier has cleared. The CJ may then use ordinary bounded
    # process-exit grace before signaling the process group. This closes the
    # final-send/END_RUN race without a timer or a second control topic.
    RESULT_SOURCE_LIVE = "result_source_live"
    # Authenticated session policy returned in HELLO_ACCEPTED. A zero timeout disables
    # heartbeat lease enforcement for compatibility with legacy jobs that explicitly
    # disabled PipeHandler heartbeat checking.
    HEARTBEAT_INTERVAL = "heartbeat_interval"
    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    # Reply-type messages (see the Topic docstring) ride Cell request replies; the reply
    # body carries its protocol topic under this key so state machines can tell e.g.
    # HELLO_ACCEPTED from HELLO_REJECTED without a separate wire message.
    REPLY_TOPIC = "reply_topic"
