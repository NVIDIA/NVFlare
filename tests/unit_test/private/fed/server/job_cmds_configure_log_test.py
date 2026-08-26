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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue, ProtoKey
from nvflare.private.admin_defs import MsgHeader, error_reply, ok_reply
from nvflare.private.fed.server import job_cmds as job_cmds_module
from nvflare.private.fed.server.job_cmds import JobCommandModule
from nvflare.private.fed.server.message_send import ClientReply


class _FakeContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _FakeJobManager:
    def get_job(self, job_id, fl_ctx):
        return SimpleNamespace(meta={JobMetaKey.STATUS: RunStatus.RUNNING})


class _FakeServerEngine:
    def __init__(self):
        self.job_def_manager = _FakeJobManager()
        self.configure_job_log = MagicMock(return_value=None)

    def new_context(self):
        return _FakeContext()


def _run_client_config(monkeypatch, replies, expected_clients=None, target_type="client"):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    engine = _FakeServerEngine()
    conn = Connection(
        app_ctx=engine,
        props={JobCommandModule.TARGET_CLIENTS: expected_clients or {}},
    )
    module = JobCommandModule()
    monkeypatch.setattr(module, "send_request_to_clients", lambda conn, message: replies)

    module.configure_job_log(conn, ["configure_job_log", "job-1", target_type, "site-a", "DEBUG"])
    return conn, engine


def _reply_with_return_code(return_code, body):
    reply = ok_reply(body=body)
    reply.set_header(MsgHeader.RETURN_CODE, return_code)
    return reply


@pytest.mark.parametrize(
    "reply, expected_info",
    [
        (error_reply("log configuration refused"), "log configuration refused"),
        (_reply_with_return_code("timeout", "request timed out"), "request timed out"),
        (None, "no reply"),
    ],
)
def test_configure_job_log_client_failure_sets_error_meta(monkeypatch, reply, expected_info):
    client_reply = ClientReply(client_token="token-a", client_name="site-a", req=None, reply=reply)

    conn, _ = _run_client_config(monkeypatch, [client_reply])

    assert conn.buffer.meta[MetaKey.STATUS] == MetaStatusValue.ERROR
    assert "site-a" in conn.buffer.meta[MetaKey.INFO]
    assert expected_info in conn.buffer.meta[MetaKey.INFO]
    assert conn.buffer.data[-1] == {
        ProtoKey.TYPE: ProtoKey.ERROR,
        ProtoKey.DATA: conn.buffer.meta[MetaKey.INFO],
    }


def test_configure_job_log_no_client_responses_sets_error_meta(monkeypatch):
    conn, _ = _run_client_config(monkeypatch, [], expected_clients={"token-a": "site-a"})

    assert conn.buffer.meta == {
        MetaKey.STATUS: MetaStatusValue.ERROR,
        MetaKey.INFO: "site-a: no reply",
    }
    assert conn.buffer.data[-1] == {
        ProtoKey.TYPE: ProtoKey.ERROR,
        ProtoKey.DATA: "site-a: no reply",
    }


def test_configure_job_log_partial_client_responses_set_error_meta(monkeypatch):
    reply = ClientReply(client_token="token-a", client_name="site-a", req=None, reply=ok_reply())

    conn, _ = _run_client_config(
        monkeypatch,
        [reply],
        expected_clients={"token-a": "site-a", "token-b": "site-b"},
    )

    assert conn.buffer.meta == {
        MetaKey.STATUS: MetaStatusValue.ERROR,
        MetaKey.INFO: "site-b: no reply",
    }
    assert conn.buffer.data[-1] == {
        ProtoKey.TYPE: ProtoKey.ERROR,
        ProtoKey.DATA: "site-b: no reply",
    }


def test_configure_job_log_success_does_not_set_error_meta(monkeypatch):
    reply = ClientReply(client_token="token-a", client_name="site-a", req=None, reply=ok_reply())

    conn, _ = _run_client_config(monkeypatch, [reply], expected_clients={"token-a": "site-a"})

    assert MetaKey.STATUS not in conn.buffer.meta


def test_configure_job_log_all_with_no_clients_remains_successful(monkeypatch):
    conn, engine = _run_client_config(monkeypatch, [], target_type="all")

    engine.configure_job_log.assert_called_once_with("job-1", "DEBUG")
    assert MetaKey.STATUS not in conn.buffer.meta
