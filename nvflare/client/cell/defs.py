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
"""Stable Client API Cell protocol constants shared by trainer and backend."""

# Frozen wire channel, namespaced away from legacy "flare_agent".
CHANNEL = "client_api"

# V1 supports one version; HELLO carries it for future negotiation.
PROTOCOL_VERSION = 1


class Topic:
    """Cell protocol topics.

    Reply topics use Cell request replies; prefixes avoid legacy topic collisions.
    """

    # Session setup (all out-of-process modes)
    HELLO = "client_api.hello"
    HELLO_ACCEPTED = "client_api.hello_accepted"
    # Semantic handshake rejection, distinct from transport/protocol ERROR.
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
    """Stable payload keys; changing their values breaks the wire protocol."""

    SESSION_ID = "session_id"
    JOB_ID = "job_id"
    SITE_NAME = "site_name"
    TRAINER_FQCN = "trainer_fqcn"
    RANK = "rank"
    PROTOCOL_VERSION = "protocol_version"
    PROOF = "proof"
    REASON = "reason"
    TASK_ID = "task_id"
    TASK_NAME = "task_name"
    # Task Shareable; FOBS chooses inline or ViaDownloader encoding.
    MODEL = "model"
    # Result Shareable with the same FOBS encoding policy.
    RESULT = "result"
    # True while send() owns an accepted source, including the RESULT_ACCEPTED race;
    # SHUTDOWN may stop the process only after this becomes False.
    RESULT_SOURCE_LIVE = "result_source_live"
    # A zero timeout disables heartbeat lease enforcement for legacy compatibility.
    HEARTBEAT_INTERVAL = "heartbeat_interval"
    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    # Identifies the protocol reply carried inside a Cell reply body.
    REPLY_TOPIC = "reply_topic"
