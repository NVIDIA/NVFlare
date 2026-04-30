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

from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue
from nvflare.private.fed.server import training_cmds as training_cmds_module
from nvflare.private.fed.server.training_cmds import TrainingCommandModule


class _FakeConnection:
    def __init__(self, engine):
        self.app_ctx = engine
        self.errors = []
        self.dicts = []

    def append_error(self, msg, meta=None):
        self.errors.append((msg, meta))

    def append_dict(self, data, meta=None):
        self.dicts.append((data, meta))


class _FakeEngine:
    def __init__(self, disable_error=None, enable_error=None):
        self.disable_error = disable_error
        self.enable_error = enable_error

    def disable_clients(self, client_names):
        if self.disable_error:
            raise self.disable_error
        return {"disabled": client_names}

    def enable_clients(self, client_names):
        if self.enable_error:
            raise self.enable_error
        return {"enabled": client_names}


def test_disable_client_reports_engine_exception_as_structured_error(monkeypatch):
    monkeypatch.setattr(training_cmds_module, "ServerEngineInternalSpec", _FakeEngine)
    conn = _FakeConnection(_FakeEngine(disable_error=RuntimeError("persist failed")))

    TrainingCommandModule().disable_client(conn, ["disable_client", "site-1"])

    assert conn.dicts == []
    assert len(conn.errors) == 1
    assert "persist failed" in conn.errors[0][0]
    assert conn.errors[0][1][MetaKey.STATUS] == MetaStatusValue.INTERNAL_ERROR


def test_enable_client_reports_engine_exception_as_structured_error(monkeypatch):
    monkeypatch.setattr(training_cmds_module, "ServerEngineInternalSpec", _FakeEngine)
    conn = _FakeConnection(_FakeEngine(enable_error=RuntimeError("persist failed")))

    TrainingCommandModule().enable_client(conn, ["enable_client", "site-1"])

    assert conn.dicts == []
    assert len(conn.errors) == 1
    assert "persist failed" in conn.errors[0][0]
    assert conn.errors[0][1][MetaKey.STATUS] == MetaStatusValue.INTERNAL_ERROR
