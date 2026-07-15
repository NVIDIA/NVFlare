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

from nvflare.client.cell.defs import CHANNEL, PROTOCOL_VERSION, MsgKey, Topic
from nvflare.client.ipc import defs as ipc_defs


def _public_str_values(clazz) -> dict:
    return {name: value for name, value in vars(clazz).items() if not name.startswith("_") and isinstance(value, str)}


class TestDefs:
    def test_protocol_version(self):
        assert PROTOCOL_VERSION == 1

    def test_expected_topics_present(self):
        expected = {
            "HELLO",
            "HELLO_ACCEPTED",
            "HELLO_REJECTED",
            "TASK_READY",
            "TASK_ACCEPTED",
            "TASK_FAILED",
            "RESULT_READY",
            "RESULT_ACCEPTED",
            "RESULT_REJECTED",
            "LOG",
            "HEARTBEAT",
            "ABORT",
            "SHUTDOWN",
            "ERROR",
        }
        assert expected == set(_public_str_values(Topic).keys())

    def test_topic_values_unique(self):
        values = list(_public_str_values(Topic).values())
        assert len(values) == len(set(values))

    def test_msg_key_values_unique(self):
        values = list(_public_str_values(MsgKey).values())
        assert len(values) == len(set(values))

    def test_no_collision_with_legacy_ipc_defs(self):
        # the Client API control protocol must not collide with the legacy FlareAgent
        # channel/topic values in nvflare/client/ipc/defs.py
        assert CHANNEL != ipc_defs.CHANNEL
        legacy_topics = {
            value for name, value in vars(ipc_defs).items() if name.startswith("TOPIC_") and isinstance(value, str)
        }
        assert not legacy_topics.intersection(_public_str_values(Topic).values())

    def test_channel_wire_value(self):
        # frozen cross-track wire constant: renaming this value is a protocol break
        assert CHANNEL == "client_api"

    def test_topic_wire_values(self):
        # exact frozen wire strings for every topic; a value rename must fail CI
        expected = {
            "HELLO": "client_api.hello",
            "HELLO_ACCEPTED": "client_api.hello_accepted",
            "HELLO_REJECTED": "client_api.hello_rejected",
            "TASK_READY": "client_api.task_ready",
            "TASK_ACCEPTED": "client_api.task_accepted",
            "TASK_FAILED": "client_api.task_failed",
            "RESULT_READY": "client_api.result_ready",
            "RESULT_ACCEPTED": "client_api.result_accepted",
            "RESULT_REJECTED": "client_api.result_rejected",
            "LOG": "client_api.log",
            "HEARTBEAT": "client_api.heartbeat",
            "ABORT": "client_api.abort",
            "SHUTDOWN": "client_api.shutdown",
            "ERROR": "client_api.error",
        }
        assert _public_str_values(Topic) == expected
        # every topic is namespaced under the channel prefix
        assert all(value.startswith("client_api.") for value in expected.values())

    def test_msg_key_wire_values(self):
        # exact frozen wire strings for every message key; a value rename must fail CI
        expected = {
            "SESSION_ID": "session_id",
            "JOB_ID": "job_id",
            "SITE_NAME": "site_name",
            "TRAINER_FQCN": "trainer_fqcn",
            "RANK": "rank",
            "PROTOCOL_VERSION": "protocol_version",
            "PROOF": "proof",
            "REASON": "reason",
            "TASK_ID": "task_id",
            "TASK_NAME": "task_name",
            "MODEL": "model",
            "RESULT_ID": "result_id",
            "RESULT": "result",
            "RESULT_SOURCE_LIVE": "result_source_live",
            "HEARTBEAT_INTERVAL": "heartbeat_interval",
            "HEARTBEAT_TIMEOUT": "heartbeat_timeout",
            "REPLY_TOPIC": "reply_topic",
        }
        assert _public_str_values(MsgKey) == expected

    def test_msg_key_names_present(self):
        for name in ("TASK_NAME", "MODEL", "PROOF"):
            assert hasattr(MsgKey, name)
