# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.flower.metrics import FlowerMetricsReceiver, _FlowerMetricsBackend, _FlowerSession
from nvflare.client.cell.defs import PROTOCOL_VERSION, MsgKey, Topic
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message


def _hello_request(trainer_fqcn: str, token: str, **overrides) -> Message:
    payload = {
        MsgKey.TRAINER_FQCN: trainer_fqcn,
        MsgKey.PROOF: token,
        MsgKey.PROTOCOL_VERSION: PROTOCOL_VERSION,
        MsgKey.JOB_ID: "job-1",
        MsgKey.SITE_NAME: "site-1",
        MsgKey.RANK: 0,
    }
    payload.update(overrides)
    return Message({MessageHeaderKey.ORIGIN: trainer_fqcn}, payload)


def _metrics_backend() -> tuple[_FlowerMetricsBackend, _FlowerSession]:
    backend = _FlowerMetricsBackend("client_api_config.json")
    session = _FlowerSession("launch-token", "site-1.job-1.flower_client_api")
    backend._session = session
    backend._job_id = "job-1"
    backend._site_name = "site-1"
    backend._context = SimpleNamespace(heartbeat_interval=5.0, heartbeat_timeout=30.0)
    return backend, session


@pytest.mark.parametrize("name", ["", "../client_api_config.json", "/tmp/client_api_config.json"])
def test_flower_metrics_receiver_rejects_unsafe_config_file_name(name):
    with pytest.raises(ValueError, match="config_file_name"):
        FlowerMetricsReceiver(name)


def test_flower_metrics_hello_authenticates_and_is_idempotent():
    backend, session = _metrics_backend()

    first = backend._handle_hello(_hello_request(session.trainer_fqcn, session.token))
    second = backend._handle_hello(_hello_request(session.trainer_fqcn, session.token))

    assert first.payload[MsgKey.REPLY_TOPIC] == Topic.HELLO_ACCEPTED
    assert first.payload[MsgKey.SESSION_ID]
    assert second.payload[MsgKey.SESSION_ID] == first.payload[MsgKey.SESSION_ID]
    assert first.payload[MsgKey.HEARTBEAT_INTERVAL] == 5.0
    assert first.payload[MsgKey.HEARTBEAT_TIMEOUT] == 30.0


def test_flower_metrics_hello_rejects_bad_token():
    backend, session = _metrics_backend()

    reply = backend._handle_hello(_hello_request(session.trainer_fqcn, "wrong-token"))

    assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.HELLO_REJECTED
    assert "token mismatch" in reply.payload[MsgKey.REASON]
    assert not session.ready.is_set()


def test_flower_metrics_finalize_removes_bootstrap_and_closes_session(tmp_path):
    backend, session = _metrics_backend()
    bootstrap = tmp_path / "client_api_config.json"
    bootstrap.write_text("{}")
    backend._bootstrap_path = str(bootstrap)
    backend._cell = MagicMock()
    session.session_id = "session-1"
    session.ready.set()

    backend.finalize(FLContext())

    assert not bootstrap.exists()
    backend._cell.fire_and_forget.assert_called_once()
    assert backend._session is None
