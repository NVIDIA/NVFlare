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

from nvflare.private.fed.server.server_engine import ServerEngine


class _FakeClientManager:
    def __init__(self):
        self.disabled = set()
        self.disable_errors = {}
        self.enable_errors = {}

    def is_client_disabled(self, client_name):
        return client_name in self.disabled

    def disable_client(self, client_name):
        if client_name in self.disable_errors:
            raise self.disable_errors[client_name]
        self.disabled.add(client_name)
        return []

    def enable_client(self, client_name):
        if client_name in self.enable_errors:
            raise self.enable_errors[client_name]
        was_disabled = client_name in self.disabled
        self.disabled.discard(client_name)
        return was_disabled


class _FakeServer:
    def __init__(self):
        self.client_manager = _FakeClientManager()
        self.admin_server = None
        self.removed_tokens = []

    def remove_client_data(self, token):
        self.removed_tokens.append(token)


def test_disable_clients_reports_already_disabled_state():
    server = _FakeServer()
    engine = ServerEngine.__new__(ServerEngine)
    engine.server = server

    first = engine.disable_clients(["site-1"])
    second = engine.disable_clients(["site-1"])

    assert first["clients"][0]["already_disabled"] is False
    assert second["clients"][0]["already_disabled"] is True


def test_disable_clients_continues_after_per_client_error():
    server = _FakeServer()
    server.client_manager.disable_errors["site-2"] = RuntimeError("persist failed")
    engine = ServerEngine.__new__(ServerEngine)
    engine.server = server

    result = engine.disable_clients(["site-1", "site-2", "site-3"])

    assert [client["client_name"] for client in result["clients"]] == ["site-1", "site-2", "site-3"]
    assert [client["state"] for client in result["clients"]] == ["disabled", "error", "disabled"]
    assert result["clients"][1]["error"] == "persist failed"
    assert server.client_manager.disabled == {"site-1", "site-3"}


def test_disable_clients_single_error_still_raises():
    server = _FakeServer()
    server.client_manager.disable_errors["site-1"] = RuntimeError("persist failed")
    engine = ServerEngine.__new__(ServerEngine)
    engine.server = server

    try:
        engine.disable_clients(["site-1"])
    except RuntimeError as e:
        assert str(e) == "persist failed"
    else:
        raise AssertionError("expected RuntimeError")


def test_enable_clients_continues_after_per_client_error():
    server = _FakeServer()
    server.client_manager.disabled.update({"site-1", "site-2", "site-3"})
    server.client_manager.enable_errors["site-2"] = RuntimeError("persist failed")
    engine = ServerEngine.__new__(ServerEngine)
    engine.server = server

    result = engine.enable_clients(["site-1", "site-2", "site-3"])

    assert [client["client_name"] for client in result["clients"]] == ["site-1", "site-2", "site-3"]
    assert [client["state"] for client in result["clients"]] == ["enabled", "error", "enabled"]
    assert result["clients"][1]["error"] == "persist failed"
    assert server.client_manager.disabled == {"site-2"}


def test_enable_clients_single_error_still_raises():
    server = _FakeServer()
    server.client_manager.disabled.add("site-1")
    server.client_manager.enable_errors["site-1"] = RuntimeError("persist failed")
    engine = ServerEngine.__new__(ServerEngine)
    engine.server = server

    try:
        engine.enable_clients(["site-1"])
    except RuntimeError as e:
        assert str(e) == "persist failed"
    else:
        raise AssertionError("expected RuntimeError")
