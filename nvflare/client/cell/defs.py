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

This module is interface freeze #1 of the Client API Execution Modes design
(docs/design/client_api_execution_modes.md, "Control Protocol"). It is consumed by the
trainer-side Cell engine, the external_process backend, and the attach backend. It is a
pure vocabulary module: constants only, no Cell/cellnet imports, no I/O.
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

    Reply-type messages (HELLO_CHALLENGE, HELLO_ACCEPTED, TASK_ACCEPTED, RESULT_ACCEPTED,
    RESULT_REJECTED) may be modeled as Cell request replies at runtime rather than separate
    sends; the constants still name them so state machines, logs, and tests share one
    vocabulary.

    Values are prefixed with "client_api." so they never collide with legacy topic values
    (e.g. "hello"/"heartbeat"/"abort"/"bye" in nvflare/client/ipc/defs.py).
    """

    # Session setup (all out-of-process modes)
    HELLO = "client_api.hello"
    HELLO_CHALLENGE = "client_api.hello_challenge"
    HELLO_PROOF = "client_api.hello_proof"
    HELLO_ACCEPTED = "client_api.hello_accepted"

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
    BYE = "client_api.bye"
    ERROR = "client_api.error"


class MsgKey:
    """Keys for Client API control protocol message payloads.

    These string values are the frozen wire contract shared across tracks; renaming a value is
    a protocol break, not a refactor.
    """

    SESSION_ID = "session_id"
    ATTACH_ID = "attach_id"
    JOB_ID = "job_id"
    SITE_NAME = "site_name"
    TRAINER_FQCN = "trainer_fqcn"
    TARGET_FQCN = "target_fqcn"
    RANK = "rank"
    # Rank policy the session is scoped to (a TokenScope / HELLO_PROOF-covered field);
    # distinct from RANK, which is the concrete rank a trainer reports in HELLO.
    RANK_POLICY = "rank_policy"
    PROTOCOL_VERSION = "protocol_version"
    NONCE = "nonce"
    PROOF = "proof"
    REASON = "reason"
    TASK_ID = "task_id"
    # TASK_READY carries the task name alongside the task id.
    TASK_NAME = "task_name"
    # TASK_READY carries the FLModel reference and its params (lazy refs, not materialized bytes).
    MODEL = "model"
    PARAMS = "params"
    RESULT_ID = "result_id"
    TRANSFER_ID = "transfer_id"
    # RESULT_READY carries the payload manifest describing the result envelope.
    MANIFEST = "manifest"
